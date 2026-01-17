from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb

DIMENSION_NAMES = ("vol_ratio", "curvature", "drawdown", "burstiness", "acorr_1", "var_ratio")


def quantile_label(point: float) -> str:
    label = int(round(point * 100))
    return f"p{label:02d}"


class SummaryStorage:
    def __init__(
        self,
        db_path: Path | str,
        spec: object,
        *,
        parquet_out: Path | str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.table = "quotient_summaries"
        self.con = duckdb.connect(str(self.db_path))
        self.parquet_out = Path(parquet_out) if parquet_out else None
        self._columns = self._build_columns()
        self._ensure_table()
        self._ensure_legitimacy_view()
        InfluenceTensorStorage(self.con, self.table, max(1, getattr(self.spec, "window_seconds", 60)))

    def _build_columns(self) -> list[str]:
        columns = [
            "symbol",
            "window_start",
            "window_end",
            "window_seconds",
            "count",
            "legitimacy_density",
            "unknown_frac",
            "event_marker",
        ]
        for name in DIMENSION_NAMES:
            columns.extend(
                [
                    f"mean_{name}",
                    f"median_{name}",
                    f"mad_{name}",
                    f"run_length_pos_{name}",
                    f"run_length_neg_{name}",
                ]
            )
        quantile_dims = getattr(self.spec, "quantile_dims", ())
        quantile_points = getattr(self.spec, "quantile_points", ())
        for dim in quantile_dims:
            if 0 <= dim < len(DIMENSION_NAMES):
                name = DIMENSION_NAMES[dim]
                for point in quantile_points:
                    label = quantile_label(point)
                    columns.append(f"quant_{name}_{label}")
        columns.append("artifact_ts")
        return columns

    def _column_type(self, column: str) -> str:
        if column in {"symbol"}:
            return "VARCHAR"
        if column in {"window_start", "window_end", "artifact_ts"}:
            return "TIMESTAMP"
        if column in {"window_seconds", "count"} or column.startswith("run_length"):
            return "INTEGER"
        if column == "event_marker":
            return "BOOLEAN"
        return "DOUBLE"

    def _ensure_table(self) -> None:
        column_defs = [f"{col} {self._column_type(col)}" for col in self._columns]
        ddl = f"CREATE TABLE IF NOT EXISTS {self.table} ({', '.join(column_defs)});"
        self.con.execute(ddl)

    def _ensure_legitimacy_view(self) -> None:
        self.con.execute(
            f"""
            CREATE VIEW IF NOT EXISTS legitimacy_stream AS
            SELECT symbol, window_start, window_end, legitimacy_density, unknown_frac, artifact_ts
            FROM {self.table}
            ORDER BY symbol, window_start
            """
        )

    def append(self, record: dict[str, object]) -> None:
        if not self._columns:
            return
        values = [record.get(col) for col in self._columns]
        placeholders = ", ".join(["?"] * len(self._columns))
        column_list = ", ".join(self._columns)
        self.con.execute(
            f"INSERT INTO {self.table} ({column_list}) VALUES ({placeholders})",
            values,
        )
        if self.parquet_out:
            self.parquet_out.parent.mkdir(parents=True, exist_ok=True)
            self.con.execute(f"COPY {self.table} TO '{self.parquet_out}' (FORMAT PARQUET);")


class InfluenceTensorStorage:
    def __init__(self, con: duckdb.DuckDBPyConnection, summary_table: str, window_seconds: int) -> None:
        self.con = con
        self.summary_table = summary_table
        self.window_seconds = max(1, window_seconds)
        self._ensure_delta_view()
        self._ensure_cross_lag_view()
        self._ensure_gate_view()
        self._ensure_tensor_table()

    def _ensure_delta_view(self) -> None:
        delta_expressions = [
            f"(mean_{name} - LAG(mean_{name}) OVER (PARTITION BY symbol ORDER BY window_start)) AS delta_{name}"
            for name in DIMENSION_NAMES
        ]
        delta_expressions.append(
            "(COALESCE(legitimacy_density, 0.0) - LAG(COALESCE(legitimacy_density, 0.0)) "
            "OVER (PARTITION BY symbol ORDER BY window_start)) AS delta_legitimacy"
        )
        select_clause = ", ".join(delta_expressions)
        self.con.execute(
            f"""
            CREATE VIEW IF NOT EXISTS quotient_deltas AS
            SELECT symbol, window_start, window_end, {select_clause}
            FROM {self.summary_table}
            ORDER BY symbol, window_start
            """
        )

    def _ensure_cross_lag_view(self) -> None:
        delta_clauses = ", ".join(
            f"(target.mean_{name} - source.mean_{name}) AS delta_{name}" for name in DIMENSION_NAMES
        )
        time_diff = "EXTRACT(EPOCH FROM target.window_start - source.window_start)"
        lag_expr = f"CAST(({time_diff}) / {self.window_seconds} AS INTEGER)"
        self.con.execute(
            f"""
            CREATE VIEW IF NOT EXISTS quotient_cross_lag_inputs AS
            SELECT
                target.symbol AS target_symbol,
                source.symbol AS source_symbol,
                {lag_expr} AS lag,
                target.window_start AS target_window,
                source.window_start AS source_window,
                {delta_clauses},
                (COALESCE(target.legitimacy_density, 0.0) - COALESCE(source.legitimacy_density, 0.0)) AS delta_legitimacy,
                COALESCE(target.legitimacy_density, 0.0) AS target_legitimacy,
                COALESCE(source.legitimacy_density, 0.0) AS source_legitimacy,
                (COALESCE(target.legitimacy_density, 0.0) * COALESCE(source.legitimacy_density, 0.0)) AS legitimacy_weight,
                target.artifact_ts AS artifact_ts
            FROM {self.summary_table} AS target
            JOIN {self.summary_table} AS source
                ON target.window_start > source.window_start
                AND MOD(CAST({time_diff} AS BIGINT), {self.window_seconds}) = 0
            WHERE {lag_expr} >= 1
            """
        )

    def _ensure_gate_view(self) -> None:
        self.con.execute(
            """
            CREATE VIEW IF NOT EXISTS influence_gate_feed AS
            SELECT target_symbol, source_symbol, lag, legitimacy_weight, delta_legitimacy, artifact_ts
            FROM quotient_cross_lag_inputs
            ORDER BY artifact_ts DESC
            """
        )

    def _ensure_tensor_table(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS influence_tensor_entries (
                target_symbol VARCHAR,
                source_symbol VARCHAR,
                lag INTEGER,
                source_dim VARCHAR,
                target_dim VARCHAR,
                weight DOUBLE,
                legitimacy_weight DOUBLE,
                artifact_ts TIMESTAMP,
                entry_ts TIMESTAMP DEFAULT current_timestamp
            );
            """
        )

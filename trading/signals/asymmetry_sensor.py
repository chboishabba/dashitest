from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import duckdb


class InfluenceTensorMonitor:
    """Load the latest influence tensor log so controllers can query triadic recommendations."""

    def __init__(
        self,
        log_dir: str | Path = "logs/asymmetry",
        duckdb_path: str | Path | None = "logs/research/quotient_summaries.duckdb",
    ):
        self.log_dir = Path(log_dir)
        self._con = self._connect_duckdb(duckdb_path)
        self.entries = list(self._iter_entries())

    def _connect_duckdb(self, path: str | Path | None) -> duckdb.DuckDBPyConnection | None:
        if path is None:
            return None
        db_path = Path(path)
        if not db_path.exists():
            return None
        return duckdb.connect(str(db_path))

    def _latest_file(self) -> Path | None:
        files = sorted(self.log_dir.glob("influence_tensor_*.jsonl"))
        return files[-1] if files else None

    def _iter_entries(self) -> Iterable[dict]:
        latest = self._latest_file()
        if latest is None:
            return
        with latest.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _latest_legitimacy(self, symbol: str) -> float | None:
        if self._con is None:
            return None
        try:
            row = self._con.execute(
                """
                SELECT legitimacy_density
                FROM legitimacy_stream
                WHERE symbol = ?
                ORDER BY window_start DESC
                LIMIT 1
                """,
                [symbol],
            ).fetchone()
        except duckdb.Error:
            return None
        if not row:
            return None
        value = row[0]
        if value is None:
            return None
        return float(value) if math.isfinite(value) else None

    def should_escalate(self, symbol: str, direction: int) -> bool:
        """Return True when the latest tensor indicates positively aligned influence for the requested direction."""
        if direction == 0 or not self.entries:
            return False
        desired = "+1" if direction > 0 else "-1"
        for entry in self.entries:
            if entry.get("target") != symbol:
                continue
            if entry.get("symbol") == desired:
                legit = self._latest_legitimacy(symbol)
                if legit is None or legit <= 0.0:
                    return False
                return True
        return False

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    duckdb = None


BAN_STATUSES = frozenset({"structural_boundary", "outside"})
ACTIONABILITY_BINS = (-1e-9, 0.2, 0.5, 0.8, float("inf"))
ACTIONABILITY_LABELS = ("very_low", "low_mid", "mid_high", "high")


@dataclass(frozen=True)
class CertificationWindow:
    symbol: str
    window_id: int
    ts_start: int
    ts_end: int
    row_count: int
    trade_certified_count: int
    preserve_certified_count: int
    ban_required_count: int
    ban_correct_count: int
    ban_missed_count: int
    forced_ban_count: int
    proposed_expected_surplus_sum: float
    executed_expected_surplus_sum: float
    realized_surplus_sum: float
    realized_efficiency: float
    realized_efficiency_abs: float
    executed_expected_surplus_active: bool
    aligned_expected_surplus_sum: float
    aligned_realized_surplus_sum: float
    aligned_realized_efficiency: float
    aligned_realized_efficiency_abs: float
    aligned_expected_surplus_active: bool
    execution_cost_realized_sum: float
    execution_cost_gap_sum: float
    execution_fill_ratio_mean: float
    hazard_contextual_aligned_drag_sum: float
    hazard_synthetic_aligned_drag_sum: float
    hazard_contextual_aligned_drag_share: float
    hazard_synthetic_aligned_drag_share: float
    hazard_contextual_aligned_efficiency: float
    hazard_synthetic_aligned_efficiency: float
    capital_mean: float
    kappa_mean: float
    hazard_active_count: int
    hazard_contextual_active_count: int
    hazard_contextual_tightened_count: int
    hazard_synthetic_tightened_count: int
    hazard_ban_count: int
    hazard_preserve_count: int
    hazard_trade_count: int
    hazard_active_share: float
    hazard_contextual_active_share: float
    hazard_contextual_tightened_share: float
    hazard_synthetic_tightened_share: float
    hazard_ban_share: float
    hazard_preserve_share: float
    hazard_trade_share: float
    hazard_name_mode: str
    hazard_contextual_mode: str
    hazard_reason_mode: str
    family_mode: str
    family_reason_mode: str
    family_capability: str
    refusal_mode: str
    window_class: str
    trade_certified_share: float
    preserve_certified_share: float
    ban_required_share: float
    ban_correct_share: float
    ban_coverage: float


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _coerce_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _mode_value(series: pd.Series, default: str = "") -> str:
    if series.empty:
        return default
    mode = series.mode(dropna=True)
    if mode.empty:
        return default
    return str(mode.iloc[0])


def _realized_efficiency_metrics(
    *,
    executed_expected_surplus_sum: float,
    realized_surplus_sum: float,
    tiny_surplus_floor: float = 1e-9,
) -> tuple[float, float, bool]:
    denom = float(executed_expected_surplus_sum)
    numer = float(realized_surplus_sum)
    active = abs(denom) > tiny_surplus_floor
    if not active:
        efficiency = 0.0
    else:
        efficiency = numer / denom
    return efficiency, abs(efficiency), active


def _drag_metrics(
    *,
    expected_surplus_sum: float,
    realized_surplus_sum: float,
) -> tuple[float, float]:
    drag_sum = max(float(expected_surplus_sum) - float(realized_surplus_sum), 0.0)
    drag_share = 0.0
    if abs(float(expected_surplus_sum)) > 1e-9:
        drag_share = drag_sum / abs(float(expected_surplus_sum))
    return drag_sum, drag_share


def _classify_window(
    *,
    row_count: int,
    trade_count: int,
    preserve_count: int,
    ban_required_count: int,
    ban_missed_count: int,
    forced_ban_count: int,
    proposed_expected_surplus_sum: float,
    executed_expected_surplus_sum: float,
    realized_surplus_sum: float,
) -> str:
    if row_count <= 0:
        return "empty"
    trade_share = float(trade_count) / float(row_count)
    preserve_share = float(preserve_count) / float(row_count)
    ban_share = float(ban_required_count) / float(row_count)
    forced_ban_share = float(forced_ban_count) / float(row_count)

    if (
        trade_count > 0
        and realized_surplus_sum > 0.0
        and ban_missed_count == 0
        and trade_share >= preserve_share
        and ban_share < 0.50
    ):
        return "tradeable"
    if preserve_count > 0 and trade_count == 0 and ban_missed_count == 0:
        return "preserve_only"
    if (
        ban_required_count > 0
        and trade_count == 0
        and (
            ban_share >= max(preserve_share, 0.25)
            or forced_ban_share >= 0.10
            or executed_expected_surplus_sum <= 0.0
        )
    ):
        return "ban_dominated"
    return "mixed"


def _family_capability(
    *,
    trade_count: int,
    preserve_count: int,
    ban_required_count: int,
) -> str:
    if trade_count > 0:
        return "trade_capable"
    if preserve_count > 0:
        return "preserve_capable"
    if ban_required_count > 0:
        return "ban_only"
    return "unclear"


def _clean_label(series: pd.Series, *, default: str = "unknown") -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.mask(cleaned.eq(""), default)


def _actionability_band(series: pd.Series) -> pd.Series:
    numeric = _coerce_numeric(series, default=0.0)
    band = pd.cut(
        numeric,
        bins=ACTIONABILITY_BINS,
        labels=ACTIONABILITY_LABELS,
        include_lowest=True,
        ordered=True,
    )
    return _clean_label(band.astype("object"), default="unknown")


def _has_informative_value(series: pd.Series, *, empty: str = "unknown") -> bool:
    cleaned = _clean_label(series, default=empty)
    return bool(cleaned.ne(empty).any())


def load_certification_frame(
    path: Path,
    *,
    step_table: str = "nashi_steps",
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".duckdb", ".db"}:
        if duckdb is None:
            raise ModuleNotFoundError("duckdb is required to read DuckDB certification inputs")
        con = duckdb.connect(str(path), read_only=True)
        try:
            return con.execute(f"SELECT * FROM {step_table}").fetchdf()
        finally:
            con.close()
    raise ValueError(f"unsupported certification input format: {path}")


def prepare_certification_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "symbol" not in prepared.columns:
        raise KeyError("certification input must include 'symbol'")
    if "ts" not in prepared.columns:
        if "timestamp" in prepared.columns:
            ts = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")
            prepared["ts"] = ts.apply(
                lambda value: int(value.value // 1_000_000) if not pd.isna(value) else 0
            ).astype("int64")
        else:
            raise KeyError("certification input must include 'ts' or 'timestamp'")

    prepared["ts"] = _coerce_numeric(prepared["ts"], default=0).astype("int64")
    prepared["action"] = _coerce_numeric(prepared.get("action", pd.Series(0, index=prepared.index)), default=0).astype(int)
    prepared["hold"] = _coerce_bool(prepared.get("hold", pd.Series(False, index=prepared.index)))
    prepared["mw_forced_ban"] = _coerce_bool(prepared.get("mw_forced_ban", pd.Series(False, index=prepared.index)))
    prepared["mw_refusal_level"] = prepared.get("mw_refusal_level", pd.Series("", index=prepared.index)).fillna("").astype(str)
    prepared["nashi_status"] = prepared.get("nashi_status", pd.Series("", index=prepared.index)).fillna("").astype(str)
    prepared["proposed_expected_surplus"] = _coerce_numeric(
        prepared.get(
            "proposed_expected_surplus",
            prepared.get("expected_surplus", pd.Series(0.0, index=prepared.index)),
        )
    )
    prepared["expected_surplus"] = _coerce_numeric(prepared.get("expected_surplus", pd.Series(0.0, index=prepared.index)))
    prepared["realized_surplus"] = _coerce_numeric(prepared.get("realized_surplus", pd.Series(0.0, index=prepared.index)))
    prepared["aligned_expected_surplus"] = _coerce_numeric(
        prepared.get("aligned_expected_surplus", prepared.get("expected_surplus", pd.Series(0.0, index=prepared.index)))
    )
    prepared["aligned_realized_surplus"] = _coerce_numeric(
        prepared.get("aligned_realized_surplus", prepared.get("realized_surplus", pd.Series(0.0, index=prepared.index)))
    )
    prepared["execution_cost_realized"] = _coerce_numeric(
        prepared.get("execution_cost_realized", prepared.get("fee", pd.Series(0.0, index=prepared.index)))
    )
    prepared["execution_cost_gap"] = _coerce_numeric(
        prepared.get("execution_cost_gap", pd.Series(0.0, index=prepared.index))
    )
    prepared["execution_fill_ratio"] = _coerce_numeric(
        prepared.get("execution_fill_ratio", pd.Series(0.0, index=prepared.index))
    )
    prepared["capital_C"] = _coerce_numeric(prepared.get("capital_C", pd.Series(0.0, index=prepared.index)))
    prepared["kappa_t"] = _coerce_numeric(prepared.get("kappa_t", pd.Series(0.0, index=prepared.index)))
    prepared["actionability"] = _coerce_numeric(prepared.get("actionability", pd.Series(0.0, index=prepared.index)))
    prepared["actionability_band"] = _actionability_band(prepared["actionability"])
    prepared["nashi_spread_regime"] = _clean_label(
        prepared.get("nashi_spread_regime", pd.Series("", index=prepared.index)),
        default="unknown",
    )
    prepared["hazard_active"] = _coerce_bool(prepared.get("hazard_active", pd.Series(False, index=prepared.index)))
    prepared["hazard_contextual_active"] = _coerce_bool(
        prepared.get("hazard_contextual_active", pd.Series(False, index=prepared.index))
    )
    prepared["hazard_forced_hold"] = _coerce_bool(prepared.get("hazard_forced_hold", pd.Series(False, index=prepared.index)))
    prepared["hazard_forced_ban"] = _coerce_bool(prepared.get("hazard_forced_ban", pd.Series(False, index=prepared.index)))
    prepared["hazard_contextual_pressure"] = _coerce_numeric(
        prepared.get("hazard_contextual_pressure", pd.Series(0.0, index=prepared.index))
    )
    prepared["hazard_name"] = _clean_label(
        prepared.get("hazard_name", pd.Series("", index=prepared.index)),
        default="unknown",
    )
    prepared["hazard_contextual_label"] = _clean_label(
        prepared.get("hazard_contextual_label", pd.Series("", index=prepared.index)),
        default="unknown",
    )
    prepared["hazard_reason"] = _clean_label(
        prepared.get("hazard_reason", pd.Series("", index=prepared.index)),
        default="unknown",
    )
    prepared["hazard_source"] = _clean_label(
        prepared.get("hazard_source", pd.Series("", index=prepared.index)),
        default="none",
    )
    prepared["hazard_tightened_source"] = _clean_label(
        prepared.get("hazard_tightened_source", pd.Series("", index=prepared.index)),
        default="none",
    )
    prepared["nashi_family_reasons"] = _clean_label(
        prepared.get("nashi_family_reasons", pd.Series("", index=prepared.index)),
        default="unknown",
    )

    family_trade = prepared.get("nashi_family_trade_certified", pd.Series(False, index=prepared.index))
    family_preserve = prepared.get("nashi_family_preserve_certified", pd.Series(False, index=prepared.index))
    prepared["nashi_family_trade_certified"] = _coerce_bool(family_trade)
    prepared["nashi_family_preserve_certified"] = _coerce_bool(family_preserve)
    prepared["nashi_family_class"] = _clean_label(
        prepared.get("nashi_family_class", pd.Series("", index=prepared.index)),
        default="unknown",
    )
    prepared["mw_refusal_level"] = _clean_label(prepared["mw_refusal_level"], default="unknown")
    prepared["mw_reason"] = _clean_label(prepared.get("mw_reason", pd.Series("", index=prepared.index)), default="unknown")

    ban_required = (
        prepared["mw_forced_ban"]
        | prepared["mw_refusal_level"].str.lower().eq("ban")
        | prepared["nashi_status"].isin(BAN_STATUSES)
    )
    ban_correct = ban_required & (prepared["hold"] | prepared["action"].eq(0))

    prepared.loc[
        prepared["hazard_source"].eq("none") & prepared["hazard_contextual_active"],
        "hazard_source",
    ] = "contextual"
    prepared.loc[
        prepared["hazard_source"].eq("none") & prepared["hazard_active"] & ~prepared["hazard_contextual_active"],
        "hazard_source",
    ] = "synthetic_only"
    prepared.loc[
        prepared["hazard_active"] & prepared["hazard_tightened_source"].eq("none"),
        "hazard_tightened_source",
    ] = prepared.loc[
        prepared["hazard_active"] & prepared["hazard_tightened_source"].eq("none"),
        "hazard_source",
    ]

    prepared["ban_required"] = ban_required
    prepared["ban_correct"] = ban_correct
    prepared["ban_missed"] = ban_required & ~ban_correct
    return prepared.sort_values(["symbol", "ts", "action"], kind="stable").reset_index(drop=True)


def _window_index(frame: pd.DataFrame, *, window_rows: int | None, window_ms: int | None) -> pd.Series:
    if window_rows is not None and window_rows > 0:
        return frame.groupby("symbol").cumcount() // int(window_rows)
    if window_ms is not None and window_ms > 0:
        return ((frame["ts"] - frame.groupby("symbol")["ts"].transform("min")) // int(window_ms)).astype(int)
    raise ValueError("one of window_rows or window_ms must be positive")


def certification_census(
    frame: pd.DataFrame,
    *,
    window_rows: int | None = 256,
    window_ms: int | None = None,
) -> pd.DataFrame:
    prepared = prepare_certification_frame(frame)
    prepared["window_id"] = _window_index(prepared, window_rows=window_rows, window_ms=window_ms)

    rows: list[CertificationWindow] = []
    for (symbol, window_id), subset in prepared.groupby(["symbol", "window_id"], sort=True):
        row_count = int(len(subset))
        trade_count = int(subset["nashi_family_trade_certified"].sum())
        preserve_count = int(subset["nashi_family_preserve_certified"].sum())
        ban_required_count = int(subset["ban_required"].sum())
        ban_correct_count = int(subset["ban_correct"].sum())
        ban_missed_count = int(subset["ban_missed"].sum())
        forced_ban_count = int(subset["mw_forced_ban"].sum())
        hazard_active_count = int(subset["hazard_active"].sum())
        hazard_contextual_active_count = int(subset["hazard_contextual_active"].sum())
        hazard_contextual_tightened_count = int(
            (subset["hazard_active"] & subset["hazard_contextual_active"]).sum()
        )
        hazard_synthetic_tightened_count = int(
            (subset["hazard_active"] & ~subset["hazard_contextual_active"]).sum()
        )
        hazard_ban_count = int((subset["hazard_active"] & subset["hazard_forced_ban"]).sum())
        hazard_preserve_count = int((subset["hazard_active"] & subset["hazard_forced_hold"] & ~subset["hazard_forced_ban"]).sum())
        hazard_trade_count = int((subset["hazard_active"] & subset["action"].ne(0)).sum())

        proposed_expected_surplus_sum = float(subset["proposed_expected_surplus"].sum())
        executed_expected_surplus_sum = float(subset["expected_surplus"].sum())
        realized_surplus_sum = float(subset["realized_surplus"].sum())
        realized_efficiency, realized_efficiency_abs, executed_expected_surplus_active = _realized_efficiency_metrics(
            executed_expected_surplus_sum=executed_expected_surplus_sum,
            realized_surplus_sum=realized_surplus_sum,
        )
        aligned_expected_surplus_sum = float(subset["aligned_expected_surplus"].sum())
        aligned_realized_surplus_sum = float(subset["aligned_realized_surplus"].sum())
        aligned_realized_efficiency, aligned_realized_efficiency_abs, aligned_expected_surplus_active = _realized_efficiency_metrics(
            executed_expected_surplus_sum=aligned_expected_surplus_sum,
            realized_surplus_sum=aligned_realized_surplus_sum,
        )
        contextual_subset = subset[subset["hazard_tightened_source"].eq("contextual")]
        synthetic_subset = subset[subset["hazard_tightened_source"].eq("synthetic_only")]
        contextual_aligned_expected_sum = float(contextual_subset["aligned_expected_surplus"].sum())
        contextual_aligned_realized_sum = float(contextual_subset["aligned_realized_surplus"].sum())
        contextual_aligned_drag_sum, contextual_aligned_drag_share = _drag_metrics(
            expected_surplus_sum=contextual_aligned_expected_sum,
            realized_surplus_sum=contextual_aligned_realized_sum,
        )
        synthetic_aligned_expected_sum = float(synthetic_subset["aligned_expected_surplus"].sum())
        synthetic_aligned_realized_sum = float(synthetic_subset["aligned_realized_surplus"].sum())
        synthetic_aligned_drag_sum, synthetic_aligned_drag_share = _drag_metrics(
            expected_surplus_sum=synthetic_aligned_expected_sum,
            realized_surplus_sum=synthetic_aligned_realized_sum,
        )
        contextual_aligned_efficiency, _, _ = _realized_efficiency_metrics(
            executed_expected_surplus_sum=contextual_aligned_expected_sum,
            realized_surplus_sum=contextual_aligned_realized_sum,
        )
        synthetic_aligned_efficiency, _, _ = _realized_efficiency_metrics(
            executed_expected_surplus_sum=synthetic_aligned_expected_sum,
            realized_surplus_sum=synthetic_aligned_realized_sum,
        )

        rows.append(
            CertificationWindow(
                symbol=str(symbol),
                window_id=int(window_id),
                ts_start=int(subset["ts"].iloc[0]),
                ts_end=int(subset["ts"].iloc[-1]),
                row_count=row_count,
                trade_certified_count=trade_count,
                preserve_certified_count=preserve_count,
                ban_required_count=ban_required_count,
                ban_correct_count=ban_correct_count,
                ban_missed_count=ban_missed_count,
                forced_ban_count=forced_ban_count,
                proposed_expected_surplus_sum=proposed_expected_surplus_sum,
                executed_expected_surplus_sum=executed_expected_surplus_sum,
                realized_surplus_sum=realized_surplus_sum,
                realized_efficiency=realized_efficiency,
                realized_efficiency_abs=realized_efficiency_abs,
                executed_expected_surplus_active=executed_expected_surplus_active,
                aligned_expected_surplus_sum=aligned_expected_surplus_sum,
                aligned_realized_surplus_sum=aligned_realized_surplus_sum,
                aligned_realized_efficiency=aligned_realized_efficiency,
                aligned_realized_efficiency_abs=aligned_realized_efficiency_abs,
                aligned_expected_surplus_active=aligned_expected_surplus_active,
                execution_cost_realized_sum=float(subset["execution_cost_realized"].sum()),
                execution_cost_gap_sum=float(subset["execution_cost_gap"].sum()),
                execution_fill_ratio_mean=float(subset["execution_fill_ratio"].mean()),
                hazard_contextual_aligned_drag_sum=contextual_aligned_drag_sum,
                hazard_synthetic_aligned_drag_sum=synthetic_aligned_drag_sum,
                hazard_contextual_aligned_drag_share=contextual_aligned_drag_share,
                hazard_synthetic_aligned_drag_share=synthetic_aligned_drag_share,
                hazard_contextual_aligned_efficiency=contextual_aligned_efficiency,
                hazard_synthetic_aligned_efficiency=synthetic_aligned_efficiency,
                capital_mean=float(subset["capital_C"].mean()),
                kappa_mean=float(subset["kappa_t"].mean()),
                hazard_active_count=hazard_active_count,
                hazard_contextual_active_count=hazard_contextual_active_count,
                hazard_contextual_tightened_count=hazard_contextual_tightened_count,
                hazard_synthetic_tightened_count=hazard_synthetic_tightened_count,
                hazard_ban_count=hazard_ban_count,
                hazard_preserve_count=hazard_preserve_count,
                hazard_trade_count=hazard_trade_count,
                hazard_active_share=(float(hazard_active_count) / row_count) if row_count else 0.0,
                hazard_contextual_active_share=(float(hazard_contextual_active_count) / row_count) if row_count else 0.0,
                hazard_contextual_tightened_share=(float(hazard_contextual_tightened_count) / row_count) if row_count else 0.0,
                hazard_synthetic_tightened_share=(float(hazard_synthetic_tightened_count) / row_count) if row_count else 0.0,
                hazard_ban_share=(float(hazard_ban_count) / row_count) if row_count else 0.0,
                hazard_preserve_share=(float(hazard_preserve_count) / row_count) if row_count else 0.0,
                hazard_trade_share=(float(hazard_trade_count) / row_count) if row_count else 0.0,
                hazard_name_mode=_mode_value(subset["hazard_name"], default="unknown"),
                hazard_contextual_mode=_mode_value(subset["hazard_contextual_label"], default="unknown"),
                hazard_reason_mode=_mode_value(subset["hazard_reason"], default="unknown"),
                family_mode=_mode_value(subset["nashi_family_class"], default=""),
                family_reason_mode=_mode_value(subset.get("nashi_family_reasons", pd.Series("", index=subset.index)), default=""),
                family_capability=_family_capability(
                    trade_count=trade_count,
                    preserve_count=preserve_count,
                    ban_required_count=ban_required_count,
                ),
                refusal_mode=_mode_value(subset["mw_refusal_level"], default=""),
                window_class=_classify_window(
                    row_count=row_count,
                    trade_count=trade_count,
                    preserve_count=preserve_count,
                    ban_required_count=ban_required_count,
                    ban_missed_count=ban_missed_count,
                    forced_ban_count=forced_ban_count,
                    proposed_expected_surplus_sum=proposed_expected_surplus_sum,
                    executed_expected_surplus_sum=executed_expected_surplus_sum,
                    realized_surplus_sum=realized_surplus_sum,
                ),
                trade_certified_share=(float(trade_count) / row_count) if row_count else 0.0,
                preserve_certified_share=(float(preserve_count) / row_count) if row_count else 0.0,
                ban_required_share=(float(ban_required_count) / row_count) if row_count else 0.0,
                ban_correct_share=(float(ban_correct_count) / row_count) if row_count else 0.0,
                ban_coverage=(float(ban_correct_count) / ban_required_count) if ban_required_count else 1.0,
            )
        )

    return pd.DataFrame([asdict(row) for row in rows])


def summarize_census(census: pd.DataFrame) -> dict[str, Any]:
    if census.empty:
        return {
            "window_count": 0,
            "symbol_count": 0,
            "trade_certified_windows": 0,
            "preserve_certified_windows": 0,
            "ban_required_windows": 0,
            "ban_correct_windows": 0,
            "ban_missed_windows": 0,
            "hazard_active_windows": 0,
            "hazard_contextual_active_windows": 0,
            "hazard_contextual_tightened_windows": 0,
            "hazard_synthetic_tightened_windows": 0,
            "hazard_ban_windows": 0,
            "hazard_preserve_windows": 0,
            "hazard_trade_windows": 0,
            "trade_certified_steps": 0,
            "preserve_certified_steps": 0,
            "ban_required_steps": 0,
            "ban_correct_steps": 0,
            "ban_missed_steps": 0,
            "hazard_active_steps": 0,
            "hazard_contextual_active_steps": 0,
            "hazard_contextual_tightened_steps": 0,
            "hazard_synthetic_tightened_steps": 0,
            "hazard_ban_steps": 0,
            "hazard_preserve_steps": 0,
            "hazard_trade_steps": 0,
            "hazard_split_summary": {},
            "hazard_drag_slice_summary": {},
            "hazard_contextual_label_summary": {},
            "hazard_summary": {},
            "window_classes": {},
        }

    trade_windows = int((census["trade_certified_count"] > 0).sum())
    preserve_windows = int((census["preserve_certified_count"] > 0).sum())
    ban_required_windows = int((census["ban_required_count"] > 0).sum())
    ban_correct_windows = int(((census["ban_required_count"] > 0) & (census["ban_missed_count"] == 0)).sum())
    ban_missed_windows = int((census["ban_missed_count"] > 0).sum())
    hazard_active_windows = int((census["hazard_active_count"] > 0).sum()) if "hazard_active_count" in census.columns else 0
    hazard_contextual_active_windows = int((census["hazard_contextual_active_count"] > 0).sum()) if "hazard_contextual_active_count" in census.columns else 0
    hazard_contextual_tightened_windows = int((census["hazard_contextual_tightened_count"] > 0).sum()) if "hazard_contextual_tightened_count" in census.columns else 0
    hazard_synthetic_tightened_windows = int((census["hazard_synthetic_tightened_count"] > 0).sum()) if "hazard_synthetic_tightened_count" in census.columns else 0
    hazard_ban_windows = int((census["hazard_ban_count"] > 0).sum()) if "hazard_ban_count" in census.columns else 0
    hazard_preserve_windows = int((census["hazard_preserve_count"] > 0).sum()) if "hazard_preserve_count" in census.columns else 0
    hazard_trade_windows = int((census["hazard_trade_count"] > 0).sum()) if "hazard_trade_count" in census.columns else 0
    window_count = int(len(census))

    def _window_split_summary(
        mask: pd.Series,
        *,
        drag_sum_col: str,
        drag_share_col: str,
        efficiency_col: str,
        step_count_col: str,
    ) -> dict[str, Any]:
        subset = census.loc[mask]
        if subset.empty:
            return {
                "window_count": 0,
                "window_share": 0.0,
                "step_count": 0,
                "aligned_drag_sum": 0.0,
                "mean_aligned_drag_share": 0.0,
                "mean_aligned_efficiency": 0.0,
            }
        return {
            "window_count": int(len(subset)),
            "window_share": float(len(subset)) / float(window_count) if window_count else 0.0,
            "step_count": int(subset[step_count_col].sum()) if step_count_col in subset.columns else 0,
            "aligned_drag_sum": float(subset[drag_sum_col].sum()) if drag_sum_col in subset.columns else 0.0,
            "mean_aligned_drag_share": float(subset[drag_share_col].mean()) if drag_share_col in subset.columns else 0.0,
            "mean_aligned_efficiency": float(subset[efficiency_col].mean()) if efficiency_col in subset.columns else 0.0,
        }

    contextual_window_mask = census["hazard_contextual_tightened_count"] > 0 if "hazard_contextual_tightened_count" in census.columns else pd.Series(False, index=census.index)
    synthetic_window_mask = census["hazard_synthetic_tightened_count"] > 0 if "hazard_synthetic_tightened_count" in census.columns else pd.Series(False, index=census.index)
    hazard_split_summary = {
        "contextual": _window_split_summary(
            contextual_window_mask,
            drag_sum_col="hazard_contextual_aligned_drag_sum",
            drag_share_col="hazard_contextual_aligned_drag_share",
            efficiency_col="hazard_contextual_aligned_efficiency",
            step_count_col="hazard_contextual_tightened_count",
        ),
        "synthetic_only": _window_split_summary(
            synthetic_window_mask,
            drag_sum_col="hazard_synthetic_aligned_drag_sum",
            drag_share_col="hazard_synthetic_aligned_drag_share",
            efficiency_col="hazard_synthetic_aligned_efficiency",
            step_count_col="hazard_synthetic_tightened_count",
        ),
    }

    hazard_drag_slice_summary: dict[str, Any] = {}
    synthetic_slice = census.loc[synthetic_window_mask]
    if not synthetic_slice.empty:
        hazard_drag_slice_summary["synthetic_only"] = {
            "window_count": int(len(synthetic_slice)),
            "aligned_drag_sum": float(synthetic_slice["hazard_synthetic_aligned_drag_sum"].sum()),
            "mean_aligned_drag_share": float(synthetic_slice["hazard_synthetic_aligned_drag_share"].mean()),
            "mean_aligned_efficiency": float(synthetic_slice["hazard_synthetic_aligned_efficiency"].mean()),
        }
    contextual_labels = census.loc[contextual_window_mask, "hazard_contextual_mode"].fillna("unknown").astype(str).str.strip()
    contextual_labels = contextual_labels.mask(contextual_labels.eq(""), "unknown")
    for label, subset in census.loc[contextual_window_mask].assign(_context_label=contextual_labels).groupby("_context_label", sort=True):
        if str(label) == "unknown":
            continue
        hazard_drag_slice_summary[f"contextual:{label}"] = {
            "window_count": int(len(subset)),
            "aligned_drag_sum": float(subset["hazard_contextual_aligned_drag_sum"].sum()),
            "mean_aligned_drag_share": float(subset["hazard_contextual_aligned_drag_share"].mean()),
            "mean_aligned_efficiency": float(subset["hazard_contextual_aligned_efficiency"].mean()),
        }

    hazard_contextual_label_summary = (
        contextual_labels[contextual_labels.ne("unknown")].value_counts().head(10).to_dict()
        if contextual_window_mask.any()
        else {}
    )

    return {
        "window_count": window_count,
        "symbol_count": int(census["symbol"].nunique()),
        "trade_certified_windows": trade_windows,
        "preserve_certified_windows": preserve_windows,
        "ban_required_windows": ban_required_windows,
        "ban_correct_windows": ban_correct_windows,
        "ban_missed_windows": ban_missed_windows,
        "hazard_active_windows": hazard_active_windows,
        "hazard_contextual_active_windows": hazard_contextual_active_windows,
        "hazard_contextual_tightened_windows": hazard_contextual_tightened_windows,
        "hazard_synthetic_tightened_windows": hazard_synthetic_tightened_windows,
        "hazard_ban_windows": hazard_ban_windows,
        "hazard_preserve_windows": hazard_preserve_windows,
        "hazard_trade_windows": hazard_trade_windows,
        "trade_certified_steps": int(census["trade_certified_count"].sum()),
        "preserve_certified_steps": int(census["preserve_certified_count"].sum()),
        "ban_required_steps": int(census["ban_required_count"].sum()),
        "ban_correct_steps": int(census["ban_correct_count"].sum()),
        "ban_missed_steps": int(census["ban_missed_count"].sum()),
        "hazard_active_steps": int(census["hazard_active_count"].sum()) if "hazard_active_count" in census.columns else 0,
        "hazard_contextual_active_steps": int(census["hazard_contextual_active_count"].sum()) if "hazard_contextual_active_count" in census.columns else 0,
        "hazard_contextual_tightened_steps": int(census["hazard_contextual_tightened_count"].sum()) if "hazard_contextual_tightened_count" in census.columns else 0,
        "hazard_synthetic_tightened_steps": int(census["hazard_synthetic_tightened_count"].sum()) if "hazard_synthetic_tightened_count" in census.columns else 0,
        "hazard_ban_steps": int(census["hazard_ban_count"].sum()) if "hazard_ban_count" in census.columns else 0,
        "hazard_preserve_steps": int(census["hazard_preserve_count"].sum()) if "hazard_preserve_count" in census.columns else 0,
        "hazard_trade_steps": int(census["hazard_trade_count"].sum()) if "hazard_trade_count" in census.columns else 0,
        "proposed_expected_surplus_sum": float(census["proposed_expected_surplus_sum"].sum()),
        "executed_expected_surplus_sum": float(census["executed_expected_surplus_sum"].sum()),
        "realized_surplus_sum": float(census["realized_surplus_sum"].sum()),
        "mean_realized_efficiency": float(census["realized_efficiency"].mean()),
        "median_realized_efficiency": float(census["realized_efficiency"].median()),
        "positive_realized_efficiency_windows": int((census["realized_efficiency"] > 0.0).sum()),
        "executed_expected_surplus_active_windows": int(census["executed_expected_surplus_active"].sum()),
        "aligned_expected_surplus_sum": float(census["aligned_expected_surplus_sum"].sum()),
        "aligned_realized_surplus_sum": float(census["aligned_realized_surplus_sum"].sum()),
        "mean_aligned_realized_efficiency": float(census["aligned_realized_efficiency"].mean()),
        "median_aligned_realized_efficiency": float(census["aligned_realized_efficiency"].median()),
        "positive_aligned_realized_efficiency_windows": int((census["aligned_realized_efficiency"] > 0.0).sum()),
        "aligned_expected_surplus_active_windows": int(census["aligned_expected_surplus_active"].sum()),
            "execution_cost_realized_sum": float(census["execution_cost_realized_sum"].sum()),
            "execution_cost_gap_sum": float(census["execution_cost_gap_sum"].sum()),
            "mean_execution_fill_ratio": float(census["execution_fill_ratio_mean"].mean()),
            "hazard_contextual_aligned_drag_sum": float(census["hazard_contextual_aligned_drag_sum"].sum()),
            "hazard_synthetic_aligned_drag_sum": float(census["hazard_synthetic_aligned_drag_sum"].sum()),
        "mean_hazard_contextual_aligned_drag_share": float(census["hazard_contextual_aligned_drag_share"].mean()),
        "mean_hazard_synthetic_aligned_drag_share": float(census["hazard_synthetic_aligned_drag_share"].mean()),
        "mean_hazard_contextual_aligned_efficiency": float(census["hazard_contextual_aligned_efficiency"].mean()),
        "mean_hazard_synthetic_aligned_efficiency": float(census["hazard_synthetic_aligned_efficiency"].mean()),
        "mean_ban_coverage": float(census["ban_coverage"].mean()),
        "hazard_split_summary": hazard_split_summary,
        "hazard_drag_slice_summary": hazard_drag_slice_summary,
        "hazard_contextual_label_summary": hazard_contextual_label_summary,
        "hazard_summary": {
            "hazard_name_modes": census["hazard_name_mode"].value_counts().head(5).to_dict()
            if "hazard_name_mode" in census.columns and _has_informative_value(census["hazard_name_mode"])
            else {},
            "hazard_contextual_modes": census["hazard_contextual_mode"].value_counts().head(5).to_dict()
            if "hazard_contextual_mode" in census.columns and _has_informative_value(census["hazard_contextual_mode"])
            else {},
            "hazard_reason_modes": census["hazard_reason_mode"].value_counts().head(5).to_dict()
            if "hazard_reason_mode" in census.columns and _has_informative_value(census["hazard_reason_mode"])
            else {},
            "hazard_source_steps": {
                "synthetic_only": int(
                    census["hazard_synthetic_tightened_count"].sum()
                ) if "hazard_synthetic_tightened_count" in census.columns else 0,
                "contextual": int(
                    census["hazard_contextual_tightened_count"].sum()
                ) if "hazard_contextual_tightened_count" in census.columns else 0,
            },
        },
        "window_classes": census["window_class"].value_counts().to_dict() if "window_class" in census.columns else {},
    }


def attribution_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_certification_frame(frame)
    axes: tuple[tuple[str, str], ...] = (
        ("refusal_mode", "mw_refusal_level"),
        ("refusal_reason", "mw_reason"),
        ("actionability_band", "actionability_band"),
        ("spread_regime", "nashi_spread_regime"),
        ("hazard_source", "hazard_source"),
        ("hazard_tightened_source", "hazard_tightened_source"),
        ("hazard_drag_slice", "hazard_drag_slice"),
        ("hazard_name", "hazard_name"),
        ("hazard_contextual_label", "hazard_contextual_label"),
        ("hazard_reason", "hazard_reason"),
        ("family_class", "nashi_family_class"),
        ("family_reason", "nashi_family_reasons"),
    )
    prepared["hazard_source"] = "none"
    prepared.loc[prepared["hazard_active"] & ~prepared["hazard_contextual_active"], "hazard_source"] = "synthetic_only"
    prepared.loc[prepared["hazard_contextual_active"], "hazard_source"] = "contextual"
    prepared["hazard_drag_slice"] = "none"
    prepared.loc[
        prepared["hazard_tightened_source"].eq("synthetic_only"),
        "hazard_drag_slice",
    ] = "synthetic_only"
    prepared.loc[
        prepared["hazard_tightened_source"].eq("contextual"),
        "hazard_drag_slice",
    ] = "contextual:" + prepared["hazard_contextual_label"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    total_drag = float((prepared["expected_surplus"] - prepared["realized_surplus"]).clip(lower=0.0).sum())
    total_aligned_drag = float((prepared["aligned_expected_surplus"] - prepared["aligned_realized_surplus"]).clip(lower=0.0).sum())
    rows: list[dict[str, Any]] = []
    for axis_name, column in axes:
        if column not in prepared.columns:
            continue
        if column.startswith("hazard_") and not _has_informative_value(prepared[column]):
            continue
        grouped = prepared.groupby(column, sort=True, dropna=False)
        for label, subset in grouped:
            executed_expected_surplus_sum = float(subset["expected_surplus"].sum())
            realized_surplus_sum = float(subset["realized_surplus"].sum())
            realized_efficiency, realized_efficiency_abs, executed_expected_surplus_active = _realized_efficiency_metrics(
                executed_expected_surplus_sum=executed_expected_surplus_sum,
                realized_surplus_sum=realized_surplus_sum,
            )
            aligned_expected_surplus_sum = float(subset["aligned_expected_surplus"].sum())
            aligned_realized_surplus_sum = float(subset["aligned_realized_surplus"].sum())
            aligned_realized_efficiency, aligned_realized_efficiency_abs, aligned_expected_surplus_active = _realized_efficiency_metrics(
                executed_expected_surplus_sum=aligned_expected_surplus_sum,
                realized_surplus_sum=aligned_realized_surplus_sum,
            )
            drag_surplus_sum = max(executed_expected_surplus_sum - realized_surplus_sum, 0.0)
            aligned_drag_surplus_sum = max(aligned_expected_surplus_sum - aligned_realized_surplus_sum, 0.0)
            rows.append(
                {
                    "axis": axis_name,
                    "label": str(label),
                    "row_count": int(len(subset)),
                    "proposed_expected_surplus_sum": float(subset["proposed_expected_surplus"].sum()),
                    "executed_expected_surplus_sum": executed_expected_surplus_sum,
                    "realized_surplus_sum": realized_surplus_sum,
                    "drag_surplus_sum": drag_surplus_sum,
                    "drag_share": (drag_surplus_sum / total_drag) if total_drag > 0.0 else 0.0,
                    "realized_efficiency": realized_efficiency,
                    "realized_efficiency_abs": realized_efficiency_abs,
                    "executed_expected_surplus_active": executed_expected_surplus_active,
                    "aligned_expected_surplus_sum": aligned_expected_surplus_sum,
                    "aligned_realized_surplus_sum": aligned_realized_surplus_sum,
                    "aligned_drag_surplus_sum": aligned_drag_surplus_sum,
                    "aligned_drag_share": (aligned_drag_surplus_sum / total_aligned_drag) if total_aligned_drag > 0.0 else 0.0,
                    "aligned_realized_efficiency": aligned_realized_efficiency,
                    "aligned_realized_efficiency_abs": aligned_realized_efficiency_abs,
                    "aligned_expected_surplus_active": aligned_expected_surplus_active,
                    "execution_cost_realized_sum": float(subset["execution_cost_realized"].sum()),
                    "execution_cost_gap_sum": float(subset["execution_cost_gap"].sum()),
                    "execution_fill_ratio_mean": float(subset["execution_fill_ratio"].mean()),
                    "hazard_active_share": float(subset["hazard_active"].mean()) if "hazard_active" in subset.columns else 0.0,
                    "hazard_contextual_active_share": float(subset["hazard_contextual_active"].mean()) if "hazard_contextual_active" in subset.columns else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "axis",
                "label",
                "row_count",
                "proposed_expected_surplus_sum",
                "executed_expected_surplus_sum",
                "realized_surplus_sum",
                "drag_surplus_sum",
                "drag_share",
                "realized_efficiency",
                "realized_efficiency_abs",
                "executed_expected_surplus_active",
                "aligned_expected_surplus_sum",
                "aligned_realized_surplus_sum",
                "aligned_drag_surplus_sum",
                "aligned_drag_share",
                "aligned_realized_efficiency",
                "aligned_realized_efficiency_abs",
                "aligned_expected_surplus_active",
                "execution_cost_realized_sum",
                "execution_cost_gap_sum",
                "execution_fill_ratio_mean",
                "hazard_active_share",
                "hazard_contextual_active_share",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["axis", "drag_surplus_sum", "executed_expected_surplus_sum", "row_count", "label"],
        ascending=[True, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def summarize_attribution(attribution: pd.DataFrame, *, top_k: int = 3) -> dict[str, Any]:
    if attribution.empty:
        return {}
    summary: dict[str, Any] = {}
    for axis, subset in attribution.groupby("axis", sort=True):
        top = subset.head(top_k)
        summary[str(axis)] = [
            {
                "label": str(row["label"]),
                "drag_share": float(row["drag_share"]),
                "drag_surplus_sum": float(row["drag_surplus_sum"]),
                "executed_expected_surplus_sum": float(row["executed_expected_surplus_sum"]),
                "realized_surplus_sum": float(row["realized_surplus_sum"]),
                "realized_efficiency": float(row["realized_efficiency"]),
                "aligned_drag_surplus_sum": float(row["aligned_drag_surplus_sum"]),
                "aligned_expected_surplus_sum": float(row["aligned_expected_surplus_sum"]),
                "aligned_realized_surplus_sum": float(row["aligned_realized_surplus_sum"]),
                "aligned_realized_efficiency": float(row["aligned_realized_efficiency"]),
                "execution_cost_realized_sum": float(row["execution_cost_realized_sum"]),
                "execution_cost_gap_sum": float(row["execution_cost_gap_sum"]),
                "execution_fill_ratio_mean": float(row["execution_fill_ratio_mean"]),
            }
            for _, row in top.iterrows()
        ]
    return summary


def write_census_outputs(
    census: pd.DataFrame,
    *,
    output_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    summary = summarize_census(census)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        census.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

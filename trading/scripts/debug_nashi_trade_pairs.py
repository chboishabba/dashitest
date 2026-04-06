#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nashi.certification import load_certification_frame, prepare_certification_frame  # noqa: E402


ContinuationFamily = Literal[
    "interior_persistent",
    "interior_softening",
    "tail_boundary_stiff",
    "boundary_break",
    "adverse_continuation",
    "flatten_transition",
    "uncertain",
]

LeadSignal = Literal[
    "none",
    "edge_persistence_decay",
    "edge_shock_spike",
    "microstructure_pressure_rise",
    "hazard_rise",
    "cost_survival_decay",
    "continuation_support_collapse",
    "negative_efficiency_drift",
    "flatten_event",
]


@dataclass(frozen=True)
class ContinuationFamilyRow:
    t: int
    family: ContinuationFamily
    lead_signal: LeadSignal
    score_interior: float
    score_boundary: float
    score_adverse: float
    continuation_candidate_active: bool
    continuation_transition_row: bool
    continuation_post_exec_active: bool
    persistence_low: bool
    shock_high: bool
    pressure_high: bool
    hazard_high: bool
    cost_survival_bad: bool
    negative_efficiency: bool
    support_collapsed: bool
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV or DuckDB step-log artifact")
    parser.add_argument("--top-k", type=int, default=5, help="How many worst trade episodes to print")
    parser.add_argument("--context-rows", type=int, default=2, help="How many rows of context to include around each episode")
    parser.add_argument("--exposure-eps", type=float, default=1e-9, help="Exposure threshold considered flat")
    parser.add_argument("--output-csv", help="Optional output path for ranked episode summary CSV")
    parser.add_argument("--rows-csv", help="Optional output path for episode row drilldown CSV")
    parser.add_argument("--summary-json", help="Optional output path for episode summary JSON")
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _coerce(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(name, pd.Series(default, index=frame.index)), errors="coerce").fillna(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _find_trade_episodes(frame: pd.DataFrame, *, exposure_eps: float) -> pd.DataFrame:
    prepared = prepare_certification_frame(frame).copy()
    if "t" in prepared.columns:
        prepared["sort_t"] = _coerce(prepared, "t", default=0).astype(int)
    else:
        prepared["sort_t"] = prepared.groupby("symbol").cumcount().astype(int)
    prepared = prepared.sort_values(["symbol", "sort_t", "ts"], kind="stable").reset_index(drop=True)
    prepared["fill_abs"] = _coerce(prepared, "fill", default=0.0).abs()
    prepared["exposure_num"] = _coerce(prepared, "exposure", default=0.0)

    episodes: list[dict[str, Any]] = []
    for symbol, subset in prepared.groupby("symbol", sort=True):
        subset = subset.reset_index(drop=True)
        prev_exposure = 0.0
        open_idx: int | None = None
        open_sign = 0
        episode_id = 0
        for idx, row in subset.iterrows():
            curr_exposure = float(row["exposure_num"])
            curr_sign = _sign(curr_exposure, exposure_eps)
            prev_sign = _sign(prev_exposure, exposure_eps)
            entered = prev_sign == 0 and curr_sign != 0 and float(row["fill_abs"]) > exposure_eps
            exited = open_idx is not None and curr_sign == 0
            flipped = open_idx is not None and curr_sign != 0 and curr_sign != open_sign

            if open_idx is None and entered:
                open_idx = idx
                open_sign = curr_sign

            if open_idx is not None and (exited or flipped):
                start = int(open_idx)
                end = int(idx)
                episode = subset.iloc[start : end + 1].copy()
                max_abs_exposure = float(episode["exposure_num"].abs().max())
                executed_expected_sum = float(_coerce(episode, "executed_expected_surplus").sum())
                realized_sum = float(_coerce(episode, "realized_surplus").sum())
                aligned_expected_sum = float(_coerce(episode, "aligned_expected_surplus").sum())
                aligned_realized_sum = float(_coerce(episode, "aligned_realized_surplus").sum())
                cost_sum = float(_coerce(episode, "execution_cost_realized").sum())
                loss_sum = max(-realized_sum, 0.0)
                drag_sum = max(executed_expected_sum - realized_sum, 0.0)
                realized_eff = 0.0
                if abs(executed_expected_sum) > 1e-9:
                    realized_eff = realized_sum / executed_expected_sum
                aligned_eff = 0.0
                if abs(aligned_expected_sum) > 1e-9:
                    aligned_eff = aligned_realized_sum / aligned_expected_sum
                episode_type = "roundtrip"
                if end - start <= 1:
                    episode_type = "immediate_unwind"
                if flipped:
                    episode_type = "sign_flip"
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "symbol": str(symbol),
                        "t_open": int(episode["sort_t"].iloc[0]),
                        "t_close": int(episode["sort_t"].iloc[-1]),
                        "ts_open": int(episode["ts"].iloc[0]),
                        "ts_close": int(episode["ts"].iloc[-1]),
                        "row_count": int(len(episode)),
                        "duration_rows": int(end - start + 1),
                        "entry_sign": int(open_sign),
                        "episode_type": episode_type,
                        "max_abs_exposure": max_abs_exposure,
                        "fill_sum_abs": float(episode["fill_abs"].sum()),
                        "executed_expected_surplus_sum": executed_expected_sum,
                        "realized_surplus_sum": realized_sum,
                        "realized_loss_sum": loss_sum,
                        "drag_sum": drag_sum,
                        "realized_efficiency": realized_eff,
                        "aligned_expected_surplus_sum": aligned_expected_sum,
                        "aligned_realized_surplus_sum": aligned_realized_sum,
                        "aligned_realized_efficiency": aligned_eff,
                        "execution_cost_realized_sum": cost_sum,
                        "entry_reason": str(episode["mw_reason"].iloc[0]),
                        "exit_reason": str(episode["mw_reason"].iloc[-1]),
                        "entry_family_class": str(episode["nashi_family_class"].iloc[0]),
                        "exit_family_class": str(episode["nashi_family_class"].iloc[-1]),
                        "entry_spread_regime": str(episode["nashi_spread_regime"].iloc[0]),
                        "exit_spread_regime": str(episode["nashi_spread_regime"].iloc[-1]),
                        "entry_hazard_label": str(episode["hazard_contextual_label"].iloc[0]),
                        "exit_hazard_label": str(episode["hazard_contextual_label"].iloc[-1]),
                        "entry_hazard_source": str(episode["hazard_source"].iloc[0]),
                        "exit_hazard_source": str(episode["hazard_source"].iloc[-1]),
                        "entry_hazard_tightened_source": str(episode["hazard_tightened_source"].iloc[0]),
                        "exit_hazard_tightened_source": str(episode["hazard_tightened_source"].iloc[-1]),
                        "entry_actionability": float(_coerce(episode.iloc[[0]], "actionability").iloc[0]),
                        "exit_actionability": float(_coerce(episode.iloc[[-1]], "actionability").iloc[0]),
                        "entry_edge": float(_coerce(episode.iloc[[0]], "edge").iloc[0]),
                        "exit_edge": float(_coerce(episode.iloc[[-1]], "edge").iloc[0]),
                        "forensic_score": 4.0 * loss_sum + 2.0 * drag_sum + 5000.0 * max(-realized_eff, 0.0),
                    }
                )
                episode_id += 1
                open_idx = None
                open_sign = 0
                if flipped and float(row["fill_abs"]) > exposure_eps:
                    open_idx = idx
                    open_sign = curr_sign
            prev_exposure = curr_exposure
    if not episodes:
        return pd.DataFrame()
    ranked = pd.DataFrame(episodes)
    return ranked.sort_values(
        ["forensic_score", "realized_loss_sum", "drag_sum"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _episode_rows(
    frame: pd.DataFrame,
    *,
    symbol: str,
    t_open: int,
    t_close: int,
    context_rows: int,
) -> pd.DataFrame:
    prepared = prepare_certification_frame(frame).copy()
    if "t" in prepared.columns:
        prepared["sort_t"] = _coerce(prepared, "t", default=0).astype(int)
    else:
        prepared["sort_t"] = prepared.groupby("symbol").cumcount().astype(int)
    prepared = prepared[prepared["symbol"].astype(str) == str(symbol)].sort_values(["sort_t", "ts"], kind="stable")
    lo = int(t_open) - int(context_rows)
    hi = int(t_close) + int(context_rows)
    subset = prepared[(prepared["sort_t"] >= lo) & (prepared["sort_t"] <= hi)].copy()
    subset["row_drag"] = (_coerce(subset, "executed_expected_surplus") - _coerce(subset, "realized_surplus")).clip(lower=0.0)
    subset["row_loss"] = (-_coerce(subset, "realized_surplus")).clip(lower=0.0)
    subset["in_episode"] = ((subset["sort_t"] >= int(t_open)) & (subset["sort_t"] <= int(t_close))).astype(int)
    subset["exposure_pre"] = _coerce(subset, "exposure", default=0.0).shift(1).fillna(0.0)
    subset["exposure_post"] = _coerce(subset, "exposure", default=0.0)
    keep = [
        "symbol",
        "sort_t",
        "ts",
        "action",
        "hold",
        "fill",
        "exposure",
        "edge",
        "edge_persistence",
        "edge_shock",
        "actionability",
        "spread_bps",
        "microstructure_pressure",
        "cost_survival_ratio",
        "family_cooldown",
        "executed_expected_surplus",
        "proposed_survivability_score",
        "proposed_survivability_viable",
        "aligned_expected_surplus",
        "realized_surplus",
        "aligned_realized_surplus",
        "realized_efficiency",
        "execution_cost_realized",
        "execution_cost_gap",
        "hazard_active",
        "hazard",
        "hazard_source",
        "hazard_tightened_source",
        "hazard_contextual_active",
        "hazard_contextual_label",
        "hazard_reason",
        "nashi_family_class",
        "nashi_family_reasons",
        "mw_reason",
        "mw_refusal_level",
        "nashi_spread_regime",
        "nashi_candidate_reason",
        "justification_chain",
        "exposure_pre",
        "exposure_post",
        "row_drag",
        "row_loss",
        "in_episode",
    ]
    present = [name for name in keep if name in subset.columns]
    return subset[present].reset_index(drop=True)


def _forensic_trace_analysis(rows: pd.DataFrame, *, exposure_eps: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    episode = rows[rows["in_episode"].astype(bool)].copy().reset_index(drop=True)
    if episode.empty:
        return rows, {}

    labels: list[ContinuationFamilyRow] = []
    prev_interior_like = True
    prev_support_scalar: float | None = None
    prev_pressure: float | None = None
    prev_hazard: float | None = None
    prev_cost_survival: float | None = None

    for _, row in episode.iterrows():
        t_value = int(_safe_float(row.get("sort_t"), 0.0))
        exposure_pre = _safe_float(row.get("exposure_pre"))
        exposure_post = _safe_float(row.get("exposure_post"))
        fill = _safe_float(row.get("fill"))
        edge = _safe_float(row.get("edge"))
        edge_abs = abs(edge)
        edge_persistence = _clip01(_safe_float(row.get("edge_persistence")))
        edge_shock = _clip01(_safe_float(row.get("edge_shock")))
        actionability = _clip01(_safe_float(row.get("actionability")))
        microstructure_pressure = _clip01(_safe_float(row.get("microstructure_pressure")))
        hazard_level = _clip01(_safe_float(row.get("hazard")))
        cost_survival_ratio = max(0.0, _safe_float(row.get("cost_survival_ratio")))
        executed_expected_surplus = _safe_float(row.get("executed_expected_surplus"))
        realized_surplus = _safe_float(row.get("realized_surplus"))
        realized_efficiency = _safe_float(row.get("realized_efficiency"))

        continuation_candidate_active = abs(exposure_pre) > exposure_eps or abs(exposure_post) > exposure_eps
        continuation_transition_row = abs(exposure_pre) > exposure_eps and abs(exposure_post) <= exposure_eps and abs(fill) > exposure_eps
        continuation_post_exec_active = abs(exposure_post) > exposure_eps

        support_scalar = edge_abs * actionability
        support_collapsed = continuation_candidate_active and executed_expected_surplus <= 1e-9
        persistence_low = edge_persistence <= 0.20
        persistence_soft = edge_persistence <= 0.45
        shock_high = edge_shock >= 0.70
        pressure_high = microstructure_pressure >= 0.65
        hazard_high = hazard_level >= 0.65
        cost_survival_bad = 0.0 < cost_survival_ratio < 1.0
        cost_survival_tail = 0.0 < cost_survival_ratio < 1.15
        negative_efficiency = realized_surplus < -1e-9 and (
            realized_efficiency < -1e-9 or support_collapsed or executed_expected_surplus <= 1e-9
        )

        persistence_decay = prev_support_scalar is not None and support_scalar < prev_support_scalar * 0.35
        pressure_rise = prev_pressure is not None and microstructure_pressure > prev_pressure + 0.10
        hazard_rise = prev_hazard is not None and hazard_level > prev_hazard + 0.10
        cost_survival_decay = prev_cost_survival is not None and cost_survival_ratio < prev_cost_survival - 0.10

        score_interior = _clip01(
            0.35 * edge_persistence
            + 0.20 * actionability
            + 0.15 * min(edge_abs, 1.0)
            + 0.15 * (1.0 - min(microstructure_pressure, 1.0))
            + 0.15 * (1.0 if executed_expected_surplus > 1e-9 else 0.0)
        )
        score_boundary = _clip01(
            0.35 * edge_shock
            + 0.20 * (1.0 if persistence_low else 0.0)
            + 0.20 * min(microstructure_pressure, 1.0)
            + 0.10 * hazard_level
            + 0.15 * (1.0 if support_collapsed else 0.0)
        )
        score_adverse = _clip01(
            0.35 * (1.0 if negative_efficiency else 0.0)
            + 0.25 * (1.0 if support_collapsed else 0.0)
            + 0.20 * (1.0 if cost_survival_tail else 0.0)
            + 0.10 * min(microstructure_pressure, 1.0)
            + 0.10 * hazard_level
        )

        family: ContinuationFamily
        lead_signal: LeadSignal
        note: str
        if continuation_transition_row:
            family = "flatten_transition"
            lead_signal = "flatten_event"
            note = "position flattened on this row"
        elif shock_high and persistence_low:
            family = "boundary_break"
            lead_signal = "edge_shock_spike"
            note = "shock-dominated persistence break"
        elif cost_survival_bad and persistence_soft and not shock_high:
            family = "tail_boundary_stiff"
            lead_signal = "cost_survival_decay"
            note = "economically fragile tail without sharp break"
        elif support_collapsed and persistence_soft:
            family = "adverse_continuation"
            lead_signal = "continuation_support_collapse"
            note = "held position lost continuation support while still open"
        elif negative_efficiency and persistence_soft and not shock_high:
            family = "adverse_continuation"
            lead_signal = "negative_efficiency_drift"
            note = "continuation drift turned economically adverse"
        elif persistence_soft or pressure_high or hazard_high or persistence_decay:
            family = "interior_softening"
            if support_collapsed:
                lead_signal = "continuation_support_collapse"
            elif persistence_decay:
                lead_signal = "edge_persistence_decay"
            elif pressure_rise:
                lead_signal = "microstructure_pressure_rise"
            elif hazard_rise:
                lead_signal = "hazard_rise"
            elif cost_survival_decay:
                lead_signal = "cost_survival_decay"
            else:
                lead_signal = "none"
            note = "interior quality is softening but not yet a hard break"
        elif score_interior >= 0.50 and executed_expected_surplus > 1e-9:
            family = "interior_persistent"
            lead_signal = "none"
            note = "stable continuation"
        else:
            family = "uncertain"
            lead_signal = "none"
            note = "mixed continuation signals"

        labels.append(
            ContinuationFamilyRow(
                t=t_value,
                family=family,
                lead_signal=lead_signal,
                score_interior=score_interior,
                score_boundary=score_boundary,
                score_adverse=score_adverse,
                continuation_candidate_active=continuation_candidate_active,
                continuation_transition_row=continuation_transition_row,
                continuation_post_exec_active=continuation_post_exec_active,
                persistence_low=persistence_low,
                shock_high=shock_high,
                pressure_high=pressure_high,
                hazard_high=hazard_high,
                cost_survival_bad=cost_survival_bad,
                negative_efficiency=negative_efficiency,
                support_collapsed=support_collapsed,
                note=note,
            )
        )
        prev_interior_like = family in {"interior_persistent", "interior_softening"}
        prev_support_scalar = support_scalar
        prev_pressure = microstructure_pressure
        prev_hazard = hazard_level
        prev_cost_survival = cost_survival_ratio

    label_df = pd.DataFrame(asdict(label) for label in labels)
    out = rows.merge(label_df, how="left", left_on="sort_t", right_on="t")
    episode_labels = label_df[label_df["continuation_candidate_active"].astype(bool)].copy()
    non_interior = episode_labels[~episode_labels["family"].isin(["interior_persistent"])]
    adverse = episode_labels[episode_labels["family"].isin(["tail_boundary_stiff", "boundary_break", "adverse_continuation", "flatten_transition"])]
    lead_signal_counts = (
        episode_labels["lead_signal"].value_counts(dropna=False).sort_values(ascending=False).to_dict()
        if not episode_labels.empty
        else {}
    )
    family_counts = (
        episode_labels["family"].value_counts(dropna=False).sort_values(ascending=False).to_dict()
        if not episode_labels.empty
        else {}
    )
    summary = {
        "first_non_interior_t": None if non_interior.empty else int(non_interior["t"].iloc[0]),
        "first_non_interior_family": None if non_interior.empty else str(non_interior["family"].iloc[0]),
        "first_non_interior_lead_signal": None if non_interior.empty else str(non_interior["lead_signal"].iloc[0]),
        "first_adverse_t": None if adverse.empty else int(adverse["t"].iloc[0]),
        "first_adverse_family": None if adverse.empty else str(adverse["family"].iloc[0]),
        "first_adverse_lead_signal": None if adverse.empty else str(adverse["lead_signal"].iloc[0]),
        "terminal_family": str(label_df["family"].iloc[-1]),
        "terminal_lead_signal": str(label_df["lead_signal"].iloc[-1]),
        "family_counts": {str(key): int(value) for key, value in family_counts.items()},
        "lead_signal_counts": {str(key): int(value) for key, value in lead_signal_counts.items()},
    }
    return out, summary


def _episode_axis_summary(rows: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    episode = rows[rows["in_episode"].astype(bool)].copy()
    if episode.empty:
        return {}
    episode["actionability_band"] = pd.cut(
        _coerce(episode, "actionability"),
        bins=[-1e-9, 0.2, 0.5, 0.8, float("inf")],
        labels=["very_low", "low_mid", "mid_high", "high"],
        include_lowest=True,
        ordered=True,
    ).astype("object").fillna("unknown")
    axes = {
        "mw_reason": "mw_reason",
        "family_class": "nashi_family_class",
        "spread_regime": "nashi_spread_regime",
        "hazard_source": "hazard_tightened_source",
        "hazard_label": "hazard_contextual_label",
        "actionability_band": "actionability_band",
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for axis, column in axes.items():
        grouped: list[dict[str, Any]] = []
        labels = episode.get(column, pd.Series("unknown", index=episode.index)).fillna("unknown").astype(str)
        for label, subset in episode.groupby(labels, sort=False):
            grouped.append(
                {
                    "label": str(label),
                    "row_count": int(len(subset)),
                    "executed_expected_surplus_sum": float(_coerce(subset, "executed_expected_surplus").sum()),
                    "realized_surplus_sum": float(_coerce(subset, "realized_surplus").sum()),
                    "execution_cost_realized_sum": float(_coerce(subset, "execution_cost_realized").sum()),
                }
            )
        grouped.sort(key=lambda row: (row["realized_surplus_sum"], -row["executed_expected_surplus_sum"]))
        out[axis] = grouped[:6]
    return out


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    frame = load_certification_frame(path)
    ranked = _find_trade_episodes(frame, exposure_eps=float(args.exposure_eps))

    if ranked.empty:
        summary = {"input": _display_path(path), "episode_count": 0, "message": "no trade episodes found"}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    selected = ranked.head(int(args.top_k)).copy()
    row_drills: list[pd.DataFrame] = []
    selected_json: list[dict[str, Any]] = []
    for _, episode in selected.iterrows():
        rows = _episode_rows(
            frame,
            symbol=str(episode["symbol"]),
            t_open=int(episode["t_open"]),
            t_close=int(episode["t_close"]),
            context_rows=int(args.context_rows),
        )
        continuation_summary: dict[str, Any] = {}
        if not rows.empty:
            rows, continuation_summary = _forensic_trace_analysis(rows, exposure_eps=float(args.exposure_eps))
        if not rows.empty:
            rows = rows.copy()
            rows.insert(0, "episode_id", int(episode["episode_id"]))
            row_drills.append(rows)
        selected_json.append(
            {
                key: (float(value) if isinstance(value, float) else int(value) if isinstance(value, int) else value)
                for key, value in episode.to_dict().items()
            }
            | {
                "axis_summary": _episode_axis_summary(rows),
                "continuation_forensics": continuation_summary,
                "rows": rows.to_dict(orient="records"),
            }
        )

    summary = {
        "input": _display_path(path),
        "episode_count": int(len(ranked)),
        "selected_episode_count": int(len(selected)),
        "worst_episode_realized_surplus_sum": float(selected["realized_surplus_sum"].min()),
        "worst_episode_drag_sum": float(selected["drag_sum"].max()),
        "selected": selected_json,
    }

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out, index=False)
    if args.rows_csv and row_drills:
        out = Path(args.rows_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(row_drills, ignore_index=True).to_csv(out, index=False)
    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nashi.certification import certification_census, load_certification_frame, prepare_certification_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV or DuckDB step-log artifact")
    parser.add_argument("--window-rows", type=int, default=64, help="Rows per forensic window")
    parser.add_argument("--top-k", type=int, default=5, help="How many bad windows to print")
    parser.add_argument("--row-top-k", type=int, default=8, help="How many worst rows to print per selected window")
    parser.add_argument("--output-csv", help="Optional output path for ranked forensic windows")
    parser.add_argument("--rows-csv", help="Optional output path for worst-row drilldown")
    parser.add_argument("--summary-json", help="Optional output path for forensic summary JSON")
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _window_metrics(frame: pd.DataFrame, *, window_rows: int) -> pd.DataFrame:
    prepared = prepare_certification_frame(frame)
    census = certification_census(prepared, window_rows=window_rows)
    if census.empty:
        return census
    drag_sum = (census["executed_expected_surplus_sum"] - census["realized_surplus_sum"]).clip(lower=0.0)
    drag_share = pd.Series(0.0, index=census.index, dtype=float)
    positive = census["executed_expected_surplus_sum"].abs() > 1e-9
    drag_share.loc[positive] = drag_sum.loc[positive] / census.loc[positive, "executed_expected_surplus_sum"].abs()
    census = census.copy()
    census["drag_sum"] = drag_sum
    census["drag_share"] = drag_share
    census["window_loss"] = (-census["realized_surplus_sum"]).clip(lower=0.0)
    census["forensic_score"] = (
        3.0 * census["window_loss"]
        + 1.5 * census["drag_sum"]
        + 5000.0 * (-census["realized_efficiency"]).clip(lower=0.0)
        + 0.50 * census["hazard_contextual_tightened_count"]
        + 0.25 * census["hazard_synthetic_tightened_count"]
        + 0.10 * census["ban_required_count"]
    )
    return census.sort_values(
        ["forensic_score", "window_loss", "drag_sum", "hazard_contextual_tightened_count"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _window_rows(prepared: pd.DataFrame, *, symbol: str, window_id: int, window_rows: int) -> pd.DataFrame:
    subset = prepared[prepared["symbol"].astype(str) == str(symbol)].copy()
    subset["window_id"] = subset.groupby("symbol").cumcount() // int(window_rows)
    return subset[subset["window_id"] == int(window_id)].copy()


def _row_drilldown(window: pd.DataFrame, *, row_top_k: int) -> pd.DataFrame:
    subset = window.copy()
    subset["row_drag"] = (subset["executed_expected_surplus"] - subset["realized_surplus"]).clip(lower=0.0)
    subset["row_loss"] = (-subset["realized_surplus"]).clip(lower=0.0)
    subset["row_forensic_score"] = (
        3.0 * subset["row_loss"]
        + 1.5 * subset["row_drag"]
        + 5000.0 * (-subset["realized_efficiency"]).clip(lower=0.0)
        + 0.50 * subset["hazard_active"].astype(float)
        + 0.35 * subset["hazard_contextual_active"].astype(float)
        + 0.25 * subset["hazard_forced_hold"].astype(float)
        + 0.25 * subset["hazard_forced_ban"].astype(float)
    )
    keep = [
        "t",
        "ts",
        "symbol",
        "price",
        "action",
        "hold",
        "fill",
        "exposure",
        "edge",
        "actionability",
        "spread_bps",
        "microstructure_pressure",
        "cost_survival_ratio",
        "expected_surplus",
        "executed_expected_surplus",
        "realized_surplus",
        "realized_efficiency",
        "execution_fill_ratio",
        "execution_cost_realized",
        "execution_cost_gap",
        "hazard",
        "hazard_regime",
        "hazard_active",
        "hazard_source",
        "hazard_tightened_source",
        "hazard_contextual_active",
        "hazard_contextual_label",
        "hazard_reason",
        "hazard_forced_hold",
        "hazard_forced_ban",
        "nashi_family_class",
        "nashi_family_reasons",
        "mw_reason",
        "mw_refusal_level",
        "nashi_status",
        "nashi_candidate_id",
        "nashi_candidate_reason",
        "nashi_spread_regime",
        "justification_chain",
        "row_drag",
        "row_loss",
        "row_forensic_score",
    ]
    present = [name for name in keep if name in subset.columns]
    return subset.sort_values(
        ["row_forensic_score", "row_loss", "row_drag"],
        ascending=[False, False, False],
        kind="stable",
    )[present].head(int(row_top_k)).reset_index(drop=True)


def _axis_summary(window: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    axes: dict[str, list[dict[str, Any]]] = {}
    mappings = {
        "hazard_tightened_source": "hazard_tightened_source",
        "hazard_contextual_label": "hazard_contextual_label",
        "family_class": "nashi_family_class",
        "family_reason": "nashi_family_reasons",
        "refusal_reason": "mw_reason",
        "spread_regime": "nashi_spread_regime",
    }
    working = window.copy()
    working["actionability_band"] = pd.cut(
        pd.to_numeric(working.get("actionability", 0.0), errors="coerce").fillna(0.0),
        bins=[-1e-9, 0.2, 0.5, 0.8, float("inf")],
        labels=["very_low", "low_mid", "mid_high", "high"],
        include_lowest=True,
        ordered=True,
    ).astype("object").fillna("unknown")
    mappings["actionability_band"] = "actionability_band"

    for axis, column in mappings.items():
        grouped = []
        labels = working.get(column, pd.Series("unknown", index=working.index)).fillna("unknown").astype(str)
        for label, subset in working.groupby(labels, sort=False):
            grouped.append(
                {
                    "label": str(label),
                    "row_count": int(len(subset)),
                    "realized_surplus_sum": float(subset.get("realized_surplus", 0.0).sum()),
                    "executed_expected_surplus_sum": float(subset.get("executed_expected_surplus", 0.0).sum()),
                    "drag_sum": float((subset.get("executed_expected_surplus", 0.0) - subset.get("realized_surplus", 0.0)).clip(lower=0.0).sum()),
                    "mean_realized_efficiency": float(pd.to_numeric(subset.get("realized_efficiency", 0.0), errors="coerce").fillna(0.0).mean()),
                }
            )
        grouped.sort(key=lambda row: (row["realized_surplus_sum"], -row["drag_sum"]))
        axes[axis] = grouped[:6]
    return axes


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    frame = load_certification_frame(path)
    prepared = prepare_certification_frame(frame)
    ranked = _window_metrics(prepared, window_rows=int(args.window_rows))

    if ranked.empty:
        summary = {
            "input": _display_path(path),
            "window_count": 0,
            "message": "no forensic windows available",
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    selected = ranked.head(int(args.top_k)).copy()
    drill_rows: list[pd.DataFrame] = []
    drill_json: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        window = _window_rows(prepared, symbol=str(row["symbol"]), window_id=int(row["window_id"]), window_rows=int(args.window_rows))
        worst_rows = _row_drilldown(window, row_top_k=int(args.row_top_k))
        if not worst_rows.empty:
            worst_rows = worst_rows.copy()
            worst_rows.insert(0, "window_id", int(row["window_id"]))
            drill_rows.append(worst_rows)
        drill_json.append(
            {
                "symbol": str(row["symbol"]),
                "window_id": int(row["window_id"]),
                "ts_start": int(row["ts_start"]),
                "ts_end": int(row["ts_end"]),
                "window_class": str(row["window_class"]),
                "realized_surplus_sum": float(row["realized_surplus_sum"]),
                "executed_expected_surplus_sum": float(row["executed_expected_surplus_sum"]),
                "drag_sum": float(row["drag_sum"]),
                "realized_efficiency": float(row["realized_efficiency"]),
                "hazard_contextual_tightened_count": int(row["hazard_contextual_tightened_count"]),
                "hazard_synthetic_tightened_count": int(row["hazard_synthetic_tightened_count"]),
                "family_mode": str(row["family_mode"]),
                "family_reason_mode": str(row["family_reason_mode"]),
                "hazard_name_mode": str(row["hazard_name_mode"]),
                "hazard_contextual_mode": str(row["hazard_contextual_mode"]),
                "refusal_mode": str(row["refusal_mode"]),
                "axis_summary": _axis_summary(window),
                "worst_rows": worst_rows.to_dict(orient="records"),
            }
        )

    summary = {
        "input": _display_path(path),
        "window_rows": int(args.window_rows),
        "top_k": int(args.top_k),
        "window_count": int(len(ranked)),
        "selected_window_count": int(len(selected)),
        "worst_window_realized_surplus_sum": float(selected["realized_surplus_sum"].min()),
        "worst_window_drag_sum": float(selected["drag_sum"].max()),
        "selected": drill_json,
    }

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out, index=False)
    if args.rows_csv and drill_rows:
        rows_path = Path(args.rows_csv)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(drill_rows, ignore_index=True).to_csv(rows_path, index=False)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

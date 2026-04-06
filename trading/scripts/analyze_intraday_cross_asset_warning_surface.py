#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, help="Summary JSON from scripts/forensic_cross_asset_context.py")
    parser.add_argument("--rows-csv", required=True, help="Annotated in-episode rows CSV from scripts/forensic_cross_asset_context.py")
    parser.add_argument("--panel-csv", required=True, help="Aligned intraday panel CSV")
    parser.add_argument("--panel-time-col", default="timestamp", help="Timestamp column in the panel CSV")
    parser.add_argument("--timestamp-tolerance", default="5min", help="Merge tolerance for warning/response rows")
    parser.add_argument("--output-csv", help="Optional CSV for per-artifact warning/response features")
    parser.add_argument("--summary-out", help="Optional JSON for grouped summary")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _sign_label(x: float) -> str:
    if x > 1e-12:
        return "pos"
    if x < -1e-12:
        return "neg"
    return "flat"


def _panel_with_features(panel_path: Path, time_col: str) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    if time_col not in panel.columns:
        raise SystemExit(f"missing panel time column {time_col!r}")
    panel[time_col] = pd.to_datetime(panel[time_col], utc=True, errors="coerce")
    panel = panel.dropna(subset=[time_col]).sort_values(time_col, kind="stable").reset_index(drop=True)

    return_cols = [col for col in panel.columns if col.endswith("__return")]
    if len(return_cols) < 2:
        raise SystemExit("panel needs at least two *__return columns")

    peer_cols = [col for col in return_cols if not col.startswith("BTC__")]
    if not peer_cols:
        raise SystemExit("panel needs at least one non-BTC return column")

    panel["peer_mean_return"] = panel[peer_cols].mean(axis=1)
    panel["peer_abs_mean_return"] = panel[peer_cols].abs().mean(axis=1)
    panel["peer_dispersion"] = panel[peer_cols].std(axis=1).fillna(0.0)
    panel["peer_consensus"] = panel[peer_cols].apply(lambda row: abs(np.mean(np.sign(row.to_numpy(dtype=float)))), axis=1)
    panel["btc_peer_gap"] = panel["BTC__return"] - panel["peer_mean_return"]
    panel["btc_agrees_with_peer_mean"] = (
        np.sign(panel["BTC__return"]).fillna(0.0) == np.sign(panel["peer_mean_return"]).fillna(0.0)
    )
    panel["sign_pattern"] = panel[return_cols].apply(
        lambda row: "|".join(_sign_label(float(v)) for v in row.to_numpy(dtype=float)),
        axis=1,
    )
    panel["btc_return_abs"] = panel["BTC__return"].abs()
    return panel


def _merge_event_rows(event_rows: pd.DataFrame, panel: pd.DataFrame, *, time_col: str, tolerance: str) -> pd.DataFrame:
    out = event_rows.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return pd.merge_asof(
        out,
        panel.sort_values(time_col, kind="stable"),
        left_on="timestamp",
        right_on=time_col,
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = pd.read_csv(args.rows_csv)
    panel = _panel_with_features(Path(args.panel_csv), args.panel_time_col)

    event_rows: list[dict[str, Any]] = []
    for record in summary.get("artifacts", []):
        label = str(record.get("label", ""))
        pair = record.get("cross_asset_row_pair") or {}
        warning_t = pair.get("warning_t")
        response_t = pair.get("response_t")
        klass = str(pair.get("cross_asset_warning_class", ""))

        subset = rows[rows["label"] == label].copy()
        for role, sort_t in (("warning", warning_t), ("response", response_t)):
            if sort_t is None:
                continue
            hit = subset[subset["sort_t"] == int(sort_t)]
            if hit.empty:
                continue
            row = hit.iloc[0].to_dict()
            row["row_role"] = role
            row["cross_asset_warning_class"] = klass
            row["artifact"] = str(record.get("artifact", ""))
            row["episode_type"] = str(record.get("episode_type", ""))
            row["entry_reason"] = str(record.get("entry_reason", ""))
            row["exit_reason"] = str(record.get("exit_reason", ""))
            event_rows.append(row)

    if not event_rows:
        raise SystemExit("no warning/response rows found")

    merged = _merge_event_rows(pd.DataFrame(event_rows), panel, time_col=args.panel_time_col, tolerance=args.timestamp_tolerance)

    keep_cols = [
        "label",
        "artifact",
        "row_role",
        "sort_t",
        "timestamp",
        "cross_asset_warning_class",
        "episode_type",
        "entry_reason",
        "exit_reason",
        "family",
        "lead_signal",
        "executed_expected_surplus",
        "realized_surplus",
        "BTC__return",
        "ES__return",
        "NQ__return",
        "peer_mean_return",
        "peer_abs_mean_return",
        "peer_dispersion",
        "peer_consensus",
        "btc_peer_gap",
        "btc_agrees_with_peer_mean",
        "sign_pattern",
        "common_factor_regime",
        "fragmented_regime",
        "recovery_friendly_regime",
    ]
    available_cols = [col for col in keep_cols if col in merged.columns]
    trimmed = merged[available_cols].copy()

    class_summary: dict[str, Any] = {}
    for klass, chunk in trimmed.groupby("cross_asset_warning_class", dropna=False):
        by_role = {}
        for role, role_chunk in chunk.groupby("row_role", dropna=False):
            by_role[str(role)] = {
                "count": int(len(role_chunk)),
                "btc_return_mean": float(role_chunk["BTC__return"].mean()) if "BTC__return" in role_chunk else None,
                "peer_mean_return_mean": float(role_chunk["peer_mean_return"].mean()) if "peer_mean_return" in role_chunk else None,
                "btc_peer_gap_mean": float(role_chunk["btc_peer_gap"].mean()) if "btc_peer_gap" in role_chunk else None,
                "peer_consensus_mean": float(role_chunk["peer_consensus"].mean()) if "peer_consensus" in role_chunk else None,
                "peer_dispersion_mean": float(role_chunk["peer_dispersion"].mean()) if "peer_dispersion" in role_chunk else None,
                "btc_agree_rate": float(role_chunk["btc_agrees_with_peer_mean"].mean()) if "btc_agrees_with_peer_mean" in role_chunk else None,
                "sign_patterns": {
                    str(k): int(v) for k, v in role_chunk["sign_pattern"].value_counts(dropna=False).head(5).to_dict().items()
                }
                if "sign_pattern" in role_chunk
                else {},
            }
        class_summary[str(klass)] = by_role

    out_summary = {
        "artifact_count": int(len(summary.get("artifacts", []))),
        "event_row_count": int(len(trimmed)),
        "class_summary": class_summary,
    }

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        trimmed.to_csv(out, index=False)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(out_summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(out_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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
    parser.add_argument("--timestamp-tolerance", default="5min", help="Backward merge tolerance for event rows")
    parser.add_argument("--window-radius", type=int, default=2, help="Temporal radius in panel rows around each matched event row")
    parser.add_argument("--output-csv", help="Optional CSV for per-event temporal features")
    parser.add_argument("--summary-out", help="Optional JSON for grouped temporal summary")
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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    if float(np.std(x)) <= 1e-18 or float(np.std(y)) <= 1e-18:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if corr != corr:
        return 0.0
    return corr


def _lag_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag > 0:
        return _safe_corr(x[lag:], y[:-lag])
    if lag < 0:
        shift = -lag
        return _safe_corr(x[:-shift], y[shift:])
    return _safe_corr(x, y)


def _load_panel(path: Path, time_col: str) -> pd.DataFrame:
    panel = pd.read_csv(path)
    if time_col not in panel.columns:
        raise SystemExit(f"missing panel time column {time_col!r}")
    panel[time_col] = pd.to_datetime(panel[time_col], utc=True, errors="coerce")
    panel = panel.dropna(subset=[time_col]).sort_values(time_col, kind="stable").reset_index(drop=True)

    return_cols = [col for col in panel.columns if col.endswith("__return")]
    if len(return_cols) < 2:
        raise SystemExit("panel needs at least two *__return columns")
    if "BTC__return" not in panel.columns:
        raise SystemExit("panel needs BTC__return")

    peer_cols = [col for col in return_cols if col != "BTC__return"]
    if not peer_cols:
        raise SystemExit("panel needs at least one peer return column")

    panel["peer_mean_return"] = panel[peer_cols].mean(axis=1)
    panel["peer_dispersion"] = panel[peer_cols].std(axis=1).fillna(0.0)
    panel["peer_sign_pattern"] = panel[peer_cols].apply(
        lambda row: "|".join(_sign_label(float(v)) for v in row.to_numpy(dtype=float)),
        axis=1,
    )
    panel["btc_peer_gap"] = panel["BTC__return"] - panel["peer_mean_return"]
    panel["btc_sign"] = panel["BTC__return"].map(_sign_label)
    panel["peer_sign"] = panel["peer_mean_return"].map(_sign_label)
    panel["btc_peer_pattern"] = panel["btc_sign"] + "|" + panel["peer_sign_pattern"]
    panel["panel_index"] = np.arange(len(panel))
    return panel


def _extract_event_rows(summary: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for artifact in summary.get("artifacts", []):
        label = str(artifact.get("label", ""))
        pair = artifact.get("cross_asset_row_pair") or {}
        klass = str(pair.get("cross_asset_warning_class", ""))
        subset = rows[rows["label"] == label].copy()
        for role, sort_t in (("warning", pair.get("warning_t")), ("response", pair.get("response_t"))):
            if sort_t is None:
                continue
            hit = subset[subset["sort_t"] == int(sort_t)]
            if hit.empty:
                continue
            row = hit.iloc[0].to_dict()
            row["row_role"] = role
            row["cross_asset_warning_class"] = klass
            row["artifact"] = str(artifact.get("artifact", ""))
            row["episode_type"] = str(artifact.get("episode_type", ""))
            row["entry_reason"] = str(artifact.get("entry_reason", ""))
            row["exit_reason"] = str(artifact.get("exit_reason", ""))
            records.append(row)
    if not records:
        raise SystemExit("no warning/response event rows found")
    out = pd.DataFrame(records)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    return out


def _match_to_panel(events: pd.DataFrame, panel: pd.DataFrame, *, time_col: str, tolerance: str) -> pd.DataFrame:
    return pd.merge_asof(
        events,
        panel[[time_col, "panel_index"]].sort_values(time_col, kind="stable"),
        left_on="timestamp",
        right_on=time_col,
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )


def _compute_temporal_features(panel: pd.DataFrame, panel_index: int, radius: int) -> dict[str, Any]:
    start = max(0, panel_index - radius)
    stop = min(len(panel), panel_index + radius + 1)
    window = panel.iloc[start:stop].copy().reset_index(drop=True)
    center = panel_index - start

    btc = window["BTC__return"].to_numpy(dtype=float)
    peer = window["peer_mean_return"].to_numpy(dtype=float)
    gap = window["btc_peer_gap"].to_numpy(dtype=float)

    cur_btc = float(btc[center])
    cur_peer = float(peer[center])
    cur_gap = float(gap[center])

    prev_btc = float(btc[center - 1]) if center - 1 >= 0 else np.nan
    next_btc = float(btc[center + 1]) if center + 1 < len(window) else np.nan
    prev_peer = float(peer[center - 1]) if center - 1 >= 0 else np.nan
    next_peer = float(peer[center + 1]) if center + 1 < len(window) else np.nan
    prev_gap = float(gap[center - 1]) if center - 1 >= 0 else np.nan
    next_gap = float(gap[center + 1]) if center + 1 < len(window) else np.nan

    lag_corrs = {f"lag_corr_{lag}": _lag_corr(btc, peer, lag) for lag in (-2, -1, 0, 1, 2)}
    best_lag = max(lag_corrs, key=lambda k: abs(lag_corrs[k]))
    best_lag_value = int(best_lag.rsplit("_", 1)[1])

    pattern_window = ",".join(window["btc_peer_pattern"].astype(str).tolist())

    return {
        "matched_panel_index": int(panel_index),
        "window_row_count": int(len(window)),
        "btc_return_t": cur_btc,
        "peer_mean_return_t": cur_peer,
        "btc_peer_gap_t": cur_gap,
        "btc_return_prev": None if prev_btc != prev_btc else prev_btc,
        "btc_return_next": None if next_btc != next_btc else next_btc,
        "peer_mean_return_prev": None if prev_peer != prev_peer else prev_peer,
        "peer_mean_return_next": None if next_peer != next_peer else next_peer,
        "btc_peer_gap_prev": None if prev_gap != prev_gap else prev_gap,
        "btc_peer_gap_next": None if next_gap != next_gap else next_gap,
        "btc_return_delta_prev": None if prev_btc != prev_btc else cur_btc - prev_btc,
        "btc_return_delta_next": None if next_btc != next_btc else next_btc - cur_btc,
        "peer_mean_delta_prev": None if prev_peer != prev_peer else cur_peer - prev_peer,
        "peer_mean_delta_next": None if next_peer != next_peer else next_peer - cur_peer,
        "btc_peer_gap_delta_prev": None if prev_gap != prev_gap else cur_gap - prev_gap,
        "btc_peer_gap_delta_next": None if next_gap != next_gap else next_gap - cur_gap,
        "window_btc_mean": float(np.mean(btc)),
        "window_peer_mean": float(np.mean(peer)),
        "window_gap_mean": float(np.mean(gap)),
        "window_gap_std": float(np.std(gap)),
        "window_peer_dispersion_mean": float(window["peer_dispersion"].mean()),
        "window_peer_sign_pattern_mode": str(window["peer_sign_pattern"].mode().iloc[0]),
        "window_btc_peer_pattern": pattern_window,
        "best_abs_lag_corr": float(lag_corrs[best_lag]),
        "best_abs_lag": best_lag_value,
        "lag_corr_m2": float(lag_corrs["lag_corr_-2"]),
        "lag_corr_m1": float(lag_corrs["lag_corr_-1"]),
        "lag_corr_0": float(lag_corrs["lag_corr_0"]),
        "lag_corr_p1": float(lag_corrs["lag_corr_1"]),
        "lag_corr_p2": float(lag_corrs["lag_corr_2"]),
        "lead_minus_lag_corr": float(lag_corrs["lag_corr_1"] - lag_corrs["lag_corr_-1"]),
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = pd.read_csv(args.rows_csv)
    panel = _load_panel(Path(args.panel_csv), args.panel_time_col)
    events = _extract_event_rows(summary, rows)
    matched = _match_to_panel(events, panel, time_col=args.panel_time_col, tolerance=args.timestamp_tolerance)

    feature_rows: list[dict[str, Any]] = []
    for row in matched.to_dict(orient="records"):
        panel_index = row.get("panel_index")
        if panel_index != panel_index:
            continue
        features = _compute_temporal_features(panel, int(panel_index), args.window_radius)
        feature_rows.append({**row, **features})

    if not feature_rows:
        raise SystemExit("no event rows matched to panel indices")

    feature_df = pd.DataFrame(feature_rows)

    summary_by_class: dict[str, Any] = {}
    group_cols = [
        "btc_return_t",
        "peer_mean_return_t",
        "btc_peer_gap_t",
        "btc_return_delta_prev",
        "btc_return_delta_next",
        "peer_mean_delta_prev",
        "peer_mean_delta_next",
        "btc_peer_gap_delta_prev",
        "btc_peer_gap_delta_next",
        "window_gap_mean",
        "window_gap_std",
        "window_peer_dispersion_mean",
        "best_abs_lag_corr",
        "best_abs_lag",
        "lag_corr_m2",
        "lag_corr_m1",
        "lag_corr_0",
        "lag_corr_p1",
        "lag_corr_p2",
        "lead_minus_lag_corr",
    ]

    for klass, klass_chunk in feature_df.groupby("cross_asset_warning_class", dropna=False):
        role_summary: dict[str, Any] = {}
        for role, role_chunk in klass_chunk.groupby("row_role", dropna=False):
            means = {}
            for col in group_cols:
                if col not in role_chunk.columns:
                    continue
                series = pd.to_numeric(role_chunk[col], errors="coerce")
                means[col] = None if series.dropna().empty else float(series.mean())
            role_summary[str(role)] = {
                "count": int(len(role_chunk)),
                "means": means,
                "pattern_modes": {
                    str(k): int(v)
                    for k, v in role_chunk["window_btc_peer_pattern"].value_counts(dropna=False).head(5).to_dict().items()
                },
            }
        summary_by_class[str(klass)] = role_summary

    out_summary = {
        "artifact_count": int(len(summary.get("artifacts", []))),
        "matched_event_row_count": int(len(feature_df)),
        "window_radius": int(args.window_radius),
        "class_summary": summary_by_class,
    }

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(out, index=False)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(out_summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(out_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

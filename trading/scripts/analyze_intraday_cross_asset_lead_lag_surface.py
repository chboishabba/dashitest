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
    parser.add_argument("--output-csv", help="Optional CSV for per-pair lead/lag features")
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


def _sign_label(x: float) -> str:
    if x > 1e-12:
        return "pos"
    if x < -1e-12:
        return "neg"
    return "flat"


def _load_panel(path: Path, time_col: str) -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_csv(path)
    if time_col not in panel.columns:
        raise SystemExit(f"missing panel time column {time_col!r}")
    panel[time_col] = pd.to_datetime(panel[time_col], utc=True, errors="coerce")
    panel = panel.dropna(subset=[time_col]).sort_values(time_col, kind="stable").reset_index(drop=True)

    return_cols = [col for col in panel.columns if col.endswith("__return")]
    if "BTC__return" not in panel.columns:
        raise SystemExit("panel needs BTC__return")
    peer_cols = [col for col in return_cols if col != "BTC__return"]
    if not peer_cols:
        raise SystemExit("panel needs at least one non-BTC *__return column")

    panel["peer_mean_return"] = panel[peer_cols].mean(axis=1)
    panel["peer_sign_pattern"] = panel[peer_cols].apply(
        lambda row: "|".join(_sign_label(float(v)) for v in row.to_numpy(dtype=float)),
        axis=1,
    )
    panel["panel_index"] = np.arange(len(panel))
    return panel, peer_cols


def _extract_pair_rows(summary: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for artifact in summary.get("artifacts", []):
        label = str(artifact.get("label", ""))
        pair = artifact.get("cross_asset_row_pair") or {}
        klass = str(pair.get("cross_asset_warning_class", ""))
        warning_t = pair.get("warning_t")
        response_t = pair.get("response_t")
        subset = rows[rows["label"] == label].copy()

        record: dict[str, Any] = {
            "label": label,
            "artifact": str(artifact.get("artifact", "")),
            "episode_type": str(artifact.get("episode_type", "")),
            "entry_reason": str(artifact.get("entry_reason", "")),
            "exit_reason": str(artifact.get("exit_reason", "")),
            "cross_asset_warning_class": klass,
            "warning_t": warning_t,
            "response_t": response_t,
        }

        for prefix, sort_t in (("warning", warning_t), ("response", response_t)):
            if sort_t is None:
                continue
            hit = subset[subset["sort_t"] == int(sort_t)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            record[f"{prefix}_timestamp"] = row.get("timestamp")
            record[f"{prefix}_family"] = str(row.get("family", ""))
            record[f"{prefix}_lead_signal"] = str(row.get("lead_signal", ""))
            record[f"{prefix}_executed_expected_surplus"] = _safe_float(row.get("executed_expected_surplus"), 0.0)
            record[f"{prefix}_realized_surplus"] = _safe_float(row.get("realized_surplus"), 0.0)
            record[f"{prefix}_sort_t"] = int(row.get("sort_t"))

        records.append(record)

    if not records:
        raise SystemExit("no pair rows found")
    out = pd.DataFrame(records)
    for col in ("warning_timestamp", "response_timestamp"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def _merge_pair_times(pair_rows: pd.DataFrame, panel: pd.DataFrame, *, time_col: str, tolerance: str) -> pd.DataFrame:
    matched = pair_rows.copy()
    for prefix in ("warning", "response"):
        ts_col = f"{prefix}_timestamp"
        if ts_col not in matched.columns:
            continue
        event_times = matched[[ts_col]].rename(columns={ts_col: "event_timestamp"})
        event_times = event_times.reset_index().rename(columns={"index": "_row_index"})
        valid = event_times.dropna(subset=["event_timestamp"]).sort_values("event_timestamp", kind="stable")
        matched[f"{prefix}_panel_index"] = np.nan
        if valid.empty:
            continue
        joined = pd.merge_asof(
            valid,
            panel[[time_col, "panel_index"]].sort_values(time_col, kind="stable"),
            left_on="event_timestamp",
            right_on=time_col,
            direction="backward",
            tolerance=pd.Timedelta(tolerance),
        )
        matched.loc[joined["_row_index"].to_numpy(dtype=int), f"{prefix}_panel_index"] = joined["panel_index"].to_numpy()
    return matched


def _window(panel: pd.DataFrame, center_index: int, radius: int) -> pd.DataFrame:
    start = max(0, center_index - radius)
    stop = min(len(panel), center_index + radius + 1)
    return panel.iloc[start:stop].copy().reset_index(drop=True)


def _support_reforms(record: dict[str, Any]) -> bool:
    warning_exec = _safe_float(record.get("warning_executed_expected_surplus"), 0.0)
    response_exec = _safe_float(record.get("response_executed_expected_surplus"), 0.0)
    response_family = str(record.get("response_family", ""))
    response_lead = str(record.get("response_lead_signal", ""))
    return (
        response_exec > max(warning_exec, 1e-9)
        and response_family == "interior_persistent"
        and response_lead in {"", "none"}
    )


def _support_stays_dead(record: dict[str, Any]) -> bool:
    response_exec = _safe_float(record.get("response_executed_expected_surplus"), 0.0)
    response_family = str(record.get("response_family", ""))
    response_lead = str(record.get("response_lead_signal", ""))
    return (
        response_exec <= 1e-9
        and response_family in {"adverse_continuation", "boundary_break", "interior_softening"}
        and response_lead in {"continuation_support_collapse", "edge_shock_spike", "negative_efficiency_drift"}
    )


def _compute_pair_features(record: dict[str, Any], panel: pd.DataFrame, peer_cols: list[str], radius: int) -> dict[str, Any]:
    warning_index = record.get("warning_panel_index")
    response_index = record.get("response_panel_index")
    if warning_index != warning_index:
        raise ValueError("warning panel index missing")

    warning_window = _window(panel, int(warning_index), radius)
    warning_center = min(radius, len(warning_window) - 1)
    warning_btc = warning_window["BTC__return"].to_numpy(dtype=float)
    warning_peer_mean = warning_window["peer_mean_return"].to_numpy(dtype=float)

    pair_features: dict[str, Any] = {
        "warning_panel_index": int(warning_index),
        "response_panel_index": None if response_index != response_index else int(response_index),
        "support_reforms": _support_reforms(record),
        "support_stays_dead": _support_stays_dead(record),
        "warning_peer_pattern_window": ",".join(
            (
                warning_window["BTC__return"].map(_sign_label)
                + "|"
                + warning_window["peer_sign_pattern"].astype(str)
            ).tolist()
        ),
    }

    lag_values = (-2, -1, 0, 1, 2)
    warning_lag_corrs = {lag: _lag_corr(warning_btc, warning_peer_mean, lag) for lag in lag_values}
    best_warning_lag = max(warning_lag_corrs, key=lambda lag: abs(warning_lag_corrs[lag]))
    pair_features.update(
        {
            "warning_best_lag": int(best_warning_lag),
            "warning_best_lag_corr": float(warning_lag_corrs[best_warning_lag]),
            "warning_lead_minus_lag_corr": float(warning_lag_corrs[1] - warning_lag_corrs[-1]),
            "warning_btc_return_t": float(warning_btc[warning_center]),
            "warning_peer_mean_t": float(warning_peer_mean[warning_center]),
            "warning_peer_mean_next": None if warning_center + 1 >= len(warning_peer_mean) else float(warning_peer_mean[warning_center + 1]),
            "warning_peer_mean_prev": None if warning_center - 1 < 0 else float(warning_peer_mean[warning_center - 1]),
        }
    )

    if response_index == response_index:
        response_window = _window(panel, int(response_index), radius)
        response_center = min(radius, len(response_window) - 1)
        response_btc = response_window["BTC__return"].to_numpy(dtype=float)
        response_peer_mean = response_window["peer_mean_return"].to_numpy(dtype=float)
        response_lag_corrs = {lag: _lag_corr(response_btc, response_peer_mean, lag) for lag in lag_values}
        best_response_lag = max(response_lag_corrs, key=lambda lag: abs(response_lag_corrs[lag]))
        pair_features.update(
            {
                "response_best_lag": int(best_response_lag),
                "response_best_lag_corr": float(response_lag_corrs[best_response_lag]),
                "response_lead_minus_lag_corr": float(response_lag_corrs[1] - response_lag_corrs[-1]),
                "response_btc_return_t": float(response_btc[response_center]),
                "response_peer_mean_t": float(response_peer_mean[response_center]),
                "response_peer_mean_next": None if response_center + 1 >= len(response_peer_mean) else float(response_peer_mean[response_center + 1]),
                "response_peer_mean_prev": None if response_center - 1 < 0 else float(response_peer_mean[response_center - 1]),
                "warning_to_response_panel_delta": int(response_index) - int(warning_index),
                "response_peer_pattern_window": ",".join(
                    (
                        response_window["BTC__return"].map(_sign_label)
                        + "|"
                        + response_window["peer_sign_pattern"].astype(str)
                    ).tolist()
                ),
            }
        )
    else:
        pair_features.update(
            {
                "response_best_lag": None,
                "response_best_lag_corr": None,
                "response_lead_minus_lag_corr": None,
                "response_btc_return_t": None,
                "response_peer_mean_t": None,
                "response_peer_mean_next": None,
                "response_peer_mean_prev": None,
                "warning_to_response_panel_delta": None,
                "response_peer_pattern_window": None,
            }
        )

    # Per-peer lag asymmetry at warning row.
    for col in peer_cols:
        peer_name = col[:-8]
        peer_values = warning_window[col].to_numpy(dtype=float)
        lag_corrs = {lag: _lag_corr(warning_btc, peer_values, lag) for lag in lag_values}
        best_lag = max(lag_corrs, key=lambda lag: abs(lag_corrs[lag]))
        pair_features[f"{peer_name.lower()}_warning_best_lag"] = int(best_lag)
        pair_features[f"{peer_name.lower()}_warning_best_lag_corr"] = float(lag_corrs[best_lag])
        pair_features[f"{peer_name.lower()}_warning_lead_minus_lag_corr"] = float(lag_corrs[1] - lag_corrs[-1])

    return pair_features


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    rows = pd.read_csv(args.rows_csv)
    panel, peer_cols = _load_panel(Path(args.panel_csv), args.panel_time_col)
    pair_rows = _extract_pair_rows(summary, rows)
    matched = _merge_pair_times(pair_rows, panel, time_col=args.panel_time_col, tolerance=args.timestamp_tolerance)

    feature_rows: list[dict[str, Any]] = []
    for record in matched.to_dict(orient="records"):
        try:
            features = _compute_pair_features(record, panel, peer_cols, args.window_radius)
        except ValueError:
            continue
        feature_rows.append({**record, **features})

    if not feature_rows:
        raise SystemExit("no pair rows matched to panel indices")

    feature_df = pd.DataFrame(feature_rows)
    summary_by_class: dict[str, Any] = {}
    numeric_cols = [
        "warning_best_lag",
        "warning_best_lag_corr",
        "warning_lead_minus_lag_corr",
        "response_best_lag",
        "response_best_lag_corr",
        "response_lead_minus_lag_corr",
        "warning_to_response_panel_delta",
        "eth_warning_best_lag",
        "eth_warning_best_lag_corr",
        "eth_warning_lead_minus_lag_corr",
        "sol_warning_best_lag",
        "sol_warning_best_lag_corr",
        "sol_warning_lead_minus_lag_corr",
    ]
    bool_cols = ["support_reforms", "support_stays_dead"]

    for klass, chunk in feature_df.groupby("cross_asset_warning_class", dropna=False):
        means: dict[str, Any] = {}
        for col in numeric_cols:
            if col not in chunk.columns:
                continue
            series = pd.to_numeric(chunk[col], errors="coerce")
            means[col] = None if series.dropna().empty else float(series.mean())
        for col in bool_cols:
            if col in chunk.columns:
                means[col] = float(pd.Series(chunk[col]).astype(float).mean())
        summary_by_class[str(klass)] = {
            "count": int(len(chunk)),
            "means": means,
            "warning_pattern_windows": {
                str(k): int(v)
                for k, v in chunk["warning_peer_pattern_window"].value_counts(dropna=False).head(5).to_dict().items()
            },
            "response_pattern_windows": {
                str(k): int(v)
                for k, v in chunk["response_peer_pattern_window"].value_counts(dropna=False).head(5).to_dict().items()
            }
            if "response_peer_pattern_window" in chunk.columns
            else {},
        }

    out_summary = {
        "artifact_count": int(len(summary.get("artifacts", []))),
        "matched_pair_count": int(len(feature_df)),
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

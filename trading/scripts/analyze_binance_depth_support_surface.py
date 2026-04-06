#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-ndjson", required=True, help="NDJSON file from scripts/download_binance_depth_snapshots.py")
    parser.add_argument("--summary-json", help="Optional summary JSON containing warning/response pair metadata")
    parser.add_argument("--rows-csv", help="Optional annotated rows CSV containing timestamps and sort_t")
    parser.add_argument("--timestamp-tolerance", default="2s", help="Backward merge tolerance for event rows")
    parser.add_argument("--near-levels", type=int, default=5, help="Depth levels considered near-touch")
    parser.add_argument("--depth-levels", type=int, default=10, help="Depth levels used for support features")
    parser.add_argument("--path-horizon", type=int, default=3, help="Future snapshot horizon for support persistence / follow-through")
    parser.add_argument("--output-csv", help="Optional CSV for per-pair L2 features")
    parser.add_argument("--summary-out", help="Optional JSON summary output")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _safe_norm(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _entropy(values: np.ndarray) -> float:
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    p = np.clip(values / total, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_sum = float(np.sum(p))
    q_sum = float(np.sum(q))
    if p_sum <= 0.0 or q_sum <= 0.0:
        return 0.0
    p = np.clip(p / p_sum, 1e-12, 1.0)
    q = np.clip(q / q_sum, 1e-12, 1.0)
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * (np.log(p) - np.log(m)))
        + 0.5 * np.sum(q * (np.log(q) - np.log(m)))
    )


def _obi(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> float:
    bid_total = float(np.sum(bid_sizes))
    ask_total = float(np.sum(ask_sizes))
    denom = bid_total + ask_total
    if denom <= 0.0:
        return 0.0
    return (bid_total - ask_total) / denom


def _parse_side(entries: Any, levels: int) -> tuple[np.ndarray, np.ndarray]:
    prices: list[float] = []
    sizes: list[float] = []
    if not isinstance(entries, list):
        return np.array([], dtype=float), np.array([], dtype=float)
    for entry in entries[:levels]:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        prices.append(_safe_float(entry[0]))
        sizes.append(_safe_float(entry[1]))
    return np.asarray(prices, dtype=float), np.asarray(sizes, dtype=float)


def _weighted_mass(prices: np.ndarray, sizes: np.ndarray, mid: float) -> float:
    if len(prices) == 0 or len(sizes) == 0:
        return 0.0
    dist = np.abs(prices - mid)
    weights = 1.0 / (1.0 + dist)
    return float(np.sum(sizes * weights))


def _support_features(row: pd.Series, near_levels: int, depth_levels: int) -> dict[str, Any]:
    bid_prices, bid_sizes = _parse_side(row.get("bids"), depth_levels)
    ask_prices, ask_sizes = _parse_side(row.get("asks"), depth_levels)
    if len(bid_prices) == 0 or len(ask_prices) == 0:
        raise ValueError("snapshot missing bid/ask levels")

    best_bid = float(bid_prices[0])
    best_ask = float(ask_prices[0])
    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid

    k1 = min(1, len(bid_sizes), len(ask_sizes))
    k5 = min(5, len(bid_sizes), len(ask_sizes))
    k10 = min(depth_levels, len(bid_sizes), len(ask_sizes))
    near = min(near_levels, len(bid_sizes), len(ask_sizes))

    bid_dist = np.abs(bid_prices[:k10] - mid)
    ask_dist = np.abs(ask_prices[:k10] - mid)
    bid_total = float(np.sum(bid_sizes[:k10]))
    ask_total = float(np.sum(ask_sizes[:k10]))

    bid_com = float(np.sum(bid_dist * bid_sizes[:k10]) / (bid_total + 1e-12))
    ask_com = float(np.sum(ask_dist * ask_sizes[:k10]) / (ask_total + 1e-12))
    bid_var = float(np.sum(((bid_dist - bid_com) ** 2) * bid_sizes[:k10]) / (bid_total + 1e-12))
    ask_var = float(np.sum(((ask_dist - ask_com) ** 2) * ask_sizes[:k10]) / (ask_total + 1e-12))

    wb = _weighted_mass(bid_prices[:k10], bid_sizes[:k10], mid)
    wa = _weighted_mass(ask_prices[:k10], ask_sizes[:k10], mid)
    wobi = 0.0 if (wb + wa) <= 0.0 else (wb - wa) / (wb + wa)

    return {
        "snapshot_timestamp": row["timestamp"],
        "symbol": str(row.get("symbol", "")),
        "lastUpdateId": row.get("lastUpdateId"),
        "mid": _safe_norm(mid),
        "spread": _safe_norm(spread),
        "obi_top1": _safe_norm(_obi(bid_sizes[:k1], ask_sizes[:k1])),
        "obi_top5": _safe_norm(_obi(bid_sizes[:k5], ask_sizes[:k5])),
        "obi_top10": _safe_norm(_obi(bid_sizes[:k10], ask_sizes[:k10])),
        "wobi_top10": _safe_norm(wobi),
        "bid_near_mass": _safe_norm(float(np.sum(bid_sizes[:near]))),
        "ask_near_mass": _safe_norm(float(np.sum(ask_sizes[:near]))),
        "bid_total": _safe_norm(bid_total),
        "ask_total": _safe_norm(ask_total),
        "bid_com": _safe_norm(bid_com),
        "ask_com": _safe_norm(ask_com),
        "bid_var": _safe_norm(bid_var),
        "ask_var": _safe_norm(ask_var),
        "bid_entropy": _safe_norm(_entropy(bid_sizes[:k10])),
        "ask_entropy": _safe_norm(_entropy(ask_sizes[:k10])),
        "js_div": _safe_norm(_js_divergence(bid_sizes[:k10], ask_sizes[:k10])),
    }


def _load_depth_features(path: Path, near_levels: int, depth_levels: int) -> pd.DataFrame:
    raw = pd.read_json(path, lines=True)
    if "timestamp" not in raw.columns:
        raise SystemExit("depth file missing timestamp")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    rows = [_support_features(row, near_levels, depth_levels) for _, row in raw.iterrows()]
    features = pd.DataFrame(rows)
    features["snapshot_index"] = np.arange(len(features))
    return features


def _extract_pair_rows(summary: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    if "selected" in summary:
        return _extract_pair_rows_from_selected(summary, rows)
    return _extract_pair_rows_from_artifacts(summary, rows)


def _extract_pair_rows_from_artifacts(summary: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for artifact in summary.get("artifacts", []):
        label = str(artifact.get("label", ""))
        pair = artifact.get("cross_asset_row_pair") or artifact.get("row_pair") or {}
        klass = str(
            pair.get("cross_asset_warning_class")
            or pair.get("warning_class")
            or artifact.get("prefix_pattern_class")
            or ""
        )
        warning_t = pair.get("warning_t")
        response_t = pair.get("response_t")
        subset = rows[rows["label"].astype(str) == label].copy()

        record: dict[str, Any] = {
            "label": label,
            "artifact": str(artifact.get("artifact", "")),
            "class": klass,
            "warning_t": warning_t,
            "response_t": response_t,
        }

        for prefix, sort_t in (("warning", warning_t), ("response", response_t)):
            if sort_t is None:
                continue
            hit = subset[pd.to_numeric(subset["sort_t"], errors="coerce") == int(sort_t)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            record[f"{prefix}_timestamp"] = row.get("timestamp")
            record[f"{prefix}_family"] = str(row.get("family", ""))
            record[f"{prefix}_lead_signal"] = str(row.get("lead_signal", ""))
            record[f"{prefix}_executed_expected_surplus"] = _safe_float(row.get("executed_expected_surplus"), 0.0)
        records.append(record)

    out = pd.DataFrame(records)
    if out.empty:
        raise SystemExit("no warning/response pair rows found")
    for col in ("warning_timestamp", "response_timestamp"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def _derive_live_pair(selected: dict[str, Any]) -> tuple[int | None, int | None, str]:
    continuation = selected.get("continuation_forensics") or {}
    episode_type = str(selected.get("episode_type", ""))
    t_open = selected.get("t_open")
    first_non_interior_t = continuation.get("first_non_interior_t")
    first_adverse_family = str(continuation.get("first_adverse_family") or "")
    terminal_family = str(continuation.get("terminal_family") or "")

    response_t: int | None = None
    warning_t: int | None = None
    if first_non_interior_t is not None:
        response_t = int(first_non_interior_t)
        if t_open is not None:
            warning_t = max(int(t_open), response_t - 1)
    elif t_open is not None and selected.get("t_close") is not None and int(selected.get("t_close")) > int(t_open):
        warning_t = int(t_open)
        response_t = int(t_open) + 1

    if terminal_family == "flatten_transition" and episode_type == "immediate_unwind":
        klass = "immediate_flatten"
    elif first_adverse_family in {"adverse_continuation", "boundary_break"}:
        klass = "confirmed_collapse_local"
    else:
        klass = "warning_without_response"
    return warning_t, response_t, klass


def _extract_pair_rows_from_selected(summary: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for selected in summary.get("selected", []):
        episode_id = selected.get("episode_id")
        warning_t, response_t, klass = _derive_live_pair(selected)
        if episode_id is None or warning_t is None:
            continue
        subset = rows[pd.to_numeric(rows.get("episode_id"), errors="coerce") == int(episode_id)].copy()
        record: dict[str, Any] = {
            "label": f"episode_{int(episode_id)}",
            "artifact": str(summary.get("input", "")),
            "class": klass,
            "warning_t": warning_t,
            "response_t": response_t,
        }
        for prefix, sort_t in (("warning", warning_t), ("response", response_t)):
            if sort_t is None:
                continue
            hit = subset[pd.to_numeric(subset["sort_t"], errors="coerce") == int(sort_t)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            record[f"{prefix}_timestamp"] = row.get("timestamp")
            record[f"{prefix}_family"] = str(row.get("family", ""))
            record[f"{prefix}_lead_signal"] = str(row.get("lead_signal", ""))
            record[f"{prefix}_executed_expected_surplus"] = _safe_float(row.get("executed_expected_surplus"), 0.0)
        records.append(record)

    out = pd.DataFrame(records)
    if out.empty:
        raise SystemExit("no warning/response pair rows found")
    for col in ("warning_timestamp", "response_timestamp"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def _merge_snapshot_times(pair_rows: pd.DataFrame, depth: pd.DataFrame, tolerance: str) -> pd.DataFrame:
    merged = pair_rows.copy()
    base = depth[["snapshot_timestamp", "snapshot_index"]].sort_values("snapshot_timestamp", kind="stable")
    for prefix in ("warning", "response"):
        ts_col = f"{prefix}_timestamp"
        merged[f"{prefix}_snapshot_index"] = np.nan
        valid = merged[[ts_col]].dropna().reset_index().rename(columns={"index": "_row_index", ts_col: "event_timestamp"})
        if valid.empty:
            continue
        joined = pd.merge_asof(
            valid.sort_values("event_timestamp", kind="stable"),
            base,
            left_on="event_timestamp",
            right_on="snapshot_timestamp",
            direction="backward",
            tolerance=pd.Timedelta(tolerance),
        )
        merged.loc[joined["_row_index"].to_numpy(dtype=int), f"{prefix}_snapshot_index"] = joined["snapshot_index"].to_numpy()
    return merged


def _support_restored(w: pd.Series, r: pd.Series) -> bool:
    obi_restore = _safe_float(r.get("obi_top5")) - _safe_float(w.get("obi_top5"))
    near_restore = (
        (_safe_float(r.get("bid_near_mass")) - _safe_float(w.get("bid_near_mass")))
        - (_safe_float(r.get("ask_near_mass")) - _safe_float(w.get("ask_near_mass")))
    )
    spread_change = _safe_float(r.get("spread")) - _safe_float(w.get("spread"))
    return obi_restore > 0.05 and near_restore > 0.0 and spread_change <= 0.0


def _support_dead(w: pd.Series, r: pd.Series) -> bool:
    near_restore = (
        (_safe_float(r.get("bid_near_mass")) - _safe_float(w.get("bid_near_mass")))
        - (_safe_float(r.get("ask_near_mass")) - _safe_float(w.get("ask_near_mass")))
    )
    return _safe_float(r.get("obi_top5")) <= 0.0 and near_restore <= 0.0


def _support_score(row: pd.Series) -> float:
    return (
        0.45 * _safe_float(row.get("obi_top5"))
        + 0.35 * _safe_float(row.get("wobi_top10"))
        + 0.20 * np.tanh((_safe_float(row.get("bid_near_mass")) - _safe_float(row.get("ask_near_mass"))) / 5.0)
    )


def _compute_path_features(response_idx: int, warning_mid: float, depth: pd.DataFrame, horizon: int) -> dict[str, Any]:
    start = int(response_idx)
    stop = min(len(depth), start + max(1, int(horizon)) + 1)
    path = depth.iloc[start:stop].copy()
    if path.empty:
        return {
            "support_persistence_mean": 0.0,
            "support_persistence_min": 0.0,
            "support_positive_frac": 0.0,
            "support_monotone_nondec": False,
            "mid_follow_through_1": 0.0,
            "mid_follow_through_h": 0.0,
            "max_favorable_excursion": 0.0,
            "max_adverse_excursion": 0.0,
            "support_decay_after_restore": 0.0,
            "support_regime": "uncertain_support",
            "false_restoration_flag": False,
            "real_restoration_flag": False,
            "dead_support_flag": False,
        }

    scores = path.apply(_support_score, axis=1).to_numpy(dtype=float)
    mids = path["mid"].to_numpy(dtype=float)
    path_start_mid = mids[0]
    favorable = mids - warning_mid

    support_restore_1 = scores[0]
    support_persistence_mean = float(scores.mean())
    support_persistence_min = float(scores.min())
    support_positive_frac = float(np.mean(scores > 0.0))
    support_monotone_nondec = bool(np.all(np.diff(scores) >= -1e-12)) if len(scores) > 1 else True
    mid_follow_through_1 = float(path_start_mid - warning_mid)
    mid_follow_through_h = float(mids[-1] - warning_mid)
    max_favorable_excursion = float(np.max(favorable))
    max_adverse_excursion = float(np.min(favorable))
    support_decay_after_restore = float(scores[-1] - scores[0])

    dead_support_flag = support_restore_1 <= 0.0 and support_persistence_mean <= 0.0
    false_restoration_flag = support_restore_1 > 0.0 and support_positive_frac >= 0.5 and mid_follow_through_h <= 0.0
    real_restoration_flag = (
        support_restore_1 > 0.0
        and support_positive_frac >= 0.75
        and mid_follow_through_h > 0.0
        and max_adverse_excursion > -abs(max_favorable_excursion)
    )

    support_regime = "uncertain_support"
    if dead_support_flag:
        support_regime = "dead_support"
    elif real_restoration_flag:
        support_regime = "real_support"
    elif false_restoration_flag:
        support_regime = "false_support"

    return {
        "support_persistence_mean": support_persistence_mean,
        "support_persistence_min": support_persistence_min,
        "support_positive_frac": support_positive_frac,
        "support_monotone_nondec": support_monotone_nondec,
        "mid_follow_through_1": mid_follow_through_1,
        "mid_follow_through_h": mid_follow_through_h,
        "max_favorable_excursion": max_favorable_excursion,
        "max_adverse_excursion": max_adverse_excursion,
        "support_decay_after_restore": support_decay_after_restore,
        "support_regime": support_regime,
        "false_restoration_flag": false_restoration_flag,
        "real_restoration_flag": real_restoration_flag,
        "dead_support_flag": dead_support_flag,
    }


def _compute_pair_features(record: pd.Series, depth: pd.DataFrame, horizon: int) -> dict[str, Any]:
    warning_idx = record.get("warning_snapshot_index")
    if warning_idx != warning_idx:
        raise ValueError("warning snapshot index missing")
    response_idx = record.get("response_snapshot_index")

    w = depth.iloc[int(warning_idx)]
    r = depth.iloc[int(response_idx)] if response_idx == response_idx else None

    out: dict[str, Any] = {
        "label": record.get("label"),
        "artifact": record.get("artifact"),
        "class": record.get("class"),
        "warning_snapshot_index": int(warning_idx),
        "response_snapshot_index": None if response_idx != response_idx else int(response_idx),
        "warning_mid": _safe_float(w["mid"]),
        "warning_spread": _safe_float(w["spread"]),
        "warning_obi_top5": _safe_float(w["obi_top5"]),
        "warning_wobi_top10": _safe_float(w["wobi_top10"]),
        "warning_bid_near_minus_ask": _safe_float(w["bid_near_mass"]) - _safe_float(w["ask_near_mass"]),
        "warning_js_div": _safe_float(w["js_div"]),
    }

    if r is None:
        out.update(
            {
                "support_restored": False,
                "support_dead": False,
                "thinning": False,
                "support_persistence_mean": 0.0,
                "support_persistence_min": 0.0,
                "support_positive_frac": 0.0,
                "support_monotone_nondec": False,
                "mid_follow_through_1": 0.0,
                "mid_follow_through_h": 0.0,
                "max_favorable_excursion": 0.0,
                "max_adverse_excursion": 0.0,
                "support_decay_after_restore": 0.0,
                "support_regime": "uncertain_support",
                "false_restoration_flag": False,
                "real_restoration_flag": False,
                "dead_support_flag": False,
            }
        )
        return out

    obi_restore = _safe_float(r["obi_top5"]) - _safe_float(w["obi_top5"])
    near_restore = (
        (_safe_float(r["bid_near_mass"]) - _safe_float(w["bid_near_mass"]))
        - (_safe_float(r["ask_near_mass"]) - _safe_float(w["ask_near_mass"]))
    )
    spread_change = _safe_float(r["spread"]) - _safe_float(w["spread"])
    jsd_change = _safe_float(r["js_div"]) - _safe_float(w["js_div"])
    thinning = _safe_float(r["bid_near_mass"]) < (_safe_float(w["bid_near_mass"]) * 0.8)

    path_features = _compute_path_features(int(response_idx), _safe_float(w["mid"]), depth, horizon)

    out.update(
        {
            "response_mid": _safe_float(r["mid"]),
            "response_spread": _safe_float(r["spread"]),
            "response_obi_top5": _safe_float(r["obi_top5"]),
            "response_wobi_top10": _safe_float(r["wobi_top10"]),
            "response_bid_near_minus_ask": _safe_float(r["bid_near_mass"]) - _safe_float(r["ask_near_mass"]),
            "response_js_div": _safe_float(r["js_div"]),
            "obi_restore": obi_restore,
            "near_mass_restore": near_restore,
            "spread_change": spread_change,
            "jsd_change": jsd_change,
            "support_restored": _support_restored(w, r),
            "support_dead": _support_dead(w, r),
            "thinning": thinning,
            "warning_to_response_snapshot_delta": int(response_idx) - int(warning_idx),
        }
    )
    out.update(path_features)
    return out


def _summarize_pairs(features: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pair_count": int(len(features)),
        "class_counts": {str(k): int(v) for k, v in features["class"].fillna("unknown").value_counts().items()},
    }
    grouped: dict[str, Any] = {}
    for klass, group in features.groupby(features["class"].fillna("unknown"), sort=True):
        grouped[str(klass)] = {
            "count": int(len(group)),
            "support_restored_rate": float(group["support_restored"].mean()) if "support_restored" in group else 0.0,
            "support_dead_rate": float(group["support_dead"].mean()) if "support_dead" in group else 0.0,
            "thinning_rate": float(group["thinning"].mean()) if "thinning" in group else 0.0,
            "false_restoration_rate": float(group["false_restoration_flag"].mean()) if "false_restoration_flag" in group else 0.0,
            "real_restoration_rate": float(group["real_restoration_flag"].mean()) if "real_restoration_flag" in group else 0.0,
            "dead_support_path_rate": float(group["dead_support_flag"].mean()) if "dead_support_flag" in group else 0.0,
            "support_regime_counts": {
                str(k): int(v)
                for k, v in group["support_regime"].fillna("unknown").value_counts().items()
            } if "support_regime" in group else {},
            "obi_restore_mean": float(group["obi_restore"].fillna(0.0).mean()) if "obi_restore" in group else 0.0,
            "near_mass_restore_mean": float(group["near_mass_restore"].fillna(0.0).mean()) if "near_mass_restore" in group else 0.0,
            "spread_change_mean": float(group["spread_change"].fillna(0.0).mean()) if "spread_change" in group else 0.0,
            "jsd_change_mean": float(group["jsd_change"].fillna(0.0).mean()) if "jsd_change" in group else 0.0,
            "support_persistence_mean": float(group["support_persistence_mean"].fillna(0.0).mean()) if "support_persistence_mean" in group else 0.0,
            "support_positive_frac_mean": float(group["support_positive_frac"].fillna(0.0).mean()) if "support_positive_frac" in group else 0.0,
            "mid_follow_through_h_mean": float(group["mid_follow_through_h"].fillna(0.0).mean()) if "mid_follow_through_h" in group else 0.0,
            "support_decay_after_restore_mean": float(group["support_decay_after_restore"].fillna(0.0).mean()) if "support_decay_after_restore" in group else 0.0,
        }
    summary["classes"] = grouped
    return summary


def main() -> None:
    args = parse_args()
    depth = _load_depth_features(Path(args.depth_ndjson), args.near_levels, args.depth_levels)

    summary: dict[str, Any] = {
        "snapshot_count": int(len(depth)),
        "timestamp_min": depth["snapshot_timestamp"].min().isoformat() if not depth.empty else None,
        "timestamp_max": depth["snapshot_timestamp"].max().isoformat() if not depth.empty else None,
        "feature_means": {
            "obi_top5": float(depth["obi_top5"].mean()) if not depth.empty else 0.0,
            "wobi_top10": float(depth["wobi_top10"].mean()) if not depth.empty else 0.0,
            "spread": float(depth["spread"].mean()) if not depth.empty else 0.0,
            "js_div": float(depth["js_div"].mean()) if not depth.empty else 0.0,
        },
    }

    pair_features = None
    if args.summary_json and args.rows_csv:
        summary_json = json.loads(Path(args.summary_json).read_text())
        rows = pd.read_csv(args.rows_csv)
        if "timestamp" not in rows.columns:
            raise SystemExit("rows CSV missing timestamp")
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
        pair_rows = _extract_pair_rows(summary_json, rows)
        pair_rows = _merge_snapshot_times(pair_rows, depth, args.timestamp_tolerance)
        matched = pair_rows[pair_rows["warning_snapshot_index"].notna()].copy()
        if matched.empty:
            summary["pair_join"] = {"matched_pairs": 0}
        else:
            records = [_compute_pair_features(row, depth, args.path_horizon) for _, row in matched.iterrows()]
            pair_features = pd.DataFrame(records)
            summary["pair_join"] = {
                "matched_pairs": int(len(pair_features)),
                "unmatched_pairs": int(len(pair_rows) - len(pair_features)),
            }
            summary["pair_summary"] = _summarize_pairs(pair_features)

    if args.output_csv and pair_features is not None:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        pair_features.to_csv(args.output_csv, index=False)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

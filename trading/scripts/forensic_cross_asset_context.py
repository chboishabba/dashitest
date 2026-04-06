#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ASSETS = {
    "BTC": Path("data/raw/stooq/btc.us.csv"),
    "SPY": Path("data/raw/stooq/spy.us.csv"),
    "AAPL": Path("data/raw/stooq/aapl.us.csv"),
    "MSFT": Path("data/raw/stooq/msft.us.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        action="append",
        required=True,
        help="Pair-summary JSON from scripts/debug_nashi_trade_pairs.py; repeatable",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional display label aligned by position with --summary-json",
    )
    parser.add_argument(
        "--asset",
        action="append",
        help="Optional asset override in NAME=path form; repeatable",
    )
    parser.add_argument(
        "--panel-csv",
        help="Optional aligned cross-asset panel from scripts/build_intraday_cross_asset_panel.py.",
    )
    parser.add_argument(
        "--panel-time-col",
        default=None,
        help="Optional timestamp column for --panel-csv. Defaults to 'timestamp' or 'date' if present.",
    )
    parser.add_argument(
        "--join-mode",
        choices=("date", "timestamp"),
        default="date",
        help="Join pair rows to cross-asset context by normalized date or exact timestamp.",
    )
    parser.add_argument(
        "--timestamp-join",
        choices=("exact", "backward"),
        default="exact",
        help="When --join-mode=timestamp, use exact match or backward asof join.",
    )
    parser.add_argument(
        "--timestamp-tolerance",
        default="10min",
        help="Pandas timedelta tolerance for backward timestamp joins.",
    )
    parser.add_argument("--rolling-window", type=int, default=20, help="Rolling window for cross-asset correlation.")
    parser.add_argument("--output-csv", help="Optional CSV path for annotated in-episode rows.")
    parser.add_argument("--summary-out", help="Optional JSON path for the aggregated summary.")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _coerce_ts_to_date(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    millis = pd.Series(np.nan, index=series.index, dtype="float64")
    abs_numeric = numeric.abs()
    ms_mask = abs_numeric >= 10**11
    s_mask = (abs_numeric >= 10**9) & ~ms_mask
    millis.loc[ms_mask] = numeric.loc[ms_mask]
    millis.loc[s_mask] = numeric.loc[s_mask] * 1000.0
    return pd.to_datetime(millis, unit="ms", utc=True, errors="coerce").dt.normalize()


def _coerce_ts_to_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    millis = pd.Series(np.nan, index=series.index, dtype="float64")
    abs_numeric = numeric.abs()
    ms_mask = abs_numeric >= 10**11
    s_mask = (abs_numeric >= 10**9) & ~ms_mask
    millis.loc[ms_mask] = numeric.loc[ms_mask]
    millis.loc[s_mask] = numeric.loc[s_mask] * 1000.0
    numeric_parsed = pd.to_datetime(millis, unit="ms", utc=True, errors="coerce")
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return numeric_parsed.where(numeric_parsed.notna(), parsed)


def _normalize_assets(raw_items: list[str] | None) -> dict[str, Path]:
    assets = dict(DEFAULT_ASSETS)
    for raw in raw_items or []:
        name, sep, value = raw.partition("=")
        if not sep or not name or not value:
            raise SystemExit(f"invalid --asset {raw!r}; expected NAME=path")
        assets[name.strip().upper()] = Path(value.strip())
    missing = [name for name, path in assets.items() if not path.exists()]
    if missing:
        raise SystemExit(f"missing asset CSVs for: {', '.join(missing)}")
    return assets


def _load_price_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    cols = {str(name).lower(): name for name in frame.columns}

    if "data" in cols and "zamkniecie" in cols:
        out = frame[[cols["data"], cols["zamkniecie"]]].rename(
            columns={cols["data"]: "date", cols["zamkniecie"]: "close"}
        )
    elif "date" in cols and "close" in cols:
        out = frame[[cols["date"], cols["close"]]].rename(
            columns={cols["date"]: "date", cols["close"]: "close"}
        )
    else:
        raise ValueError(f"unsupported CSV shape for {path}")

    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date", kind="stable").reset_index(drop=True)
    out["return"] = out["close"].pct_change().fillna(0.0)
    return out[["date", "close", "return"]]


def _safe_corr_matrix(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] < 2:
        return np.eye(max(values.shape[1], 1))
    if values.shape[0] < 2:
        return np.eye(values.shape[1])
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _sign_entropy(values: np.ndarray) -> float:
    signs = np.sign(values)
    unique, counts = np.unique(signs, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)))
    return float(entropy / np.log(3.0))


def _compute_cross_asset_context(asset_paths: dict[str, Path], *, rolling_window: int) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    asset_names = list(asset_paths.keys())
    for name, path in asset_paths.items():
        frame = _load_price_history(path)
        renamed = frame.rename(columns={"return": f"{name}__return", "close": f"{name}__close"})
        merged = renamed if merged is None else merged.merge(renamed, on="date", how="inner")
    if merged is None or merged.empty:
        raise SystemExit("no overlapping dates across asset histories")

    return_cols = [f"{name}__return" for name in asset_names if f"{name}__return" in merged.columns]
    values = merged[return_cols].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for idx in range(len(merged)):
        start = max(0, idx - rolling_window + 1)
        window = values[start : idx + 1, :]
        corr = _safe_corr_matrix(window)
        eigvals = np.sort(np.real(np.linalg.eigvalsh(corr)))[::-1]
        lambda1 = float(eigvals[0]) if len(eigvals) else 1.0
        eigsum = float(np.sum(eigvals)) if len(eigvals) else 1.0
        lambda1_share = lambda1 / max(eigsum, 1e-12)
        eigengap = float(eigvals[0] - eigvals[1]) if len(eigvals) > 1 else lambda1
        row_ret = values[idx, :]
        return_consensus = float(abs(np.mean(np.sign(row_ret))))
        return_dispersion = float(np.std(row_ret))
        sign_entropy = _sign_entropy(row_ret)
        common_factor_regime = lambda1_share >= 0.62 and return_consensus >= 0.67
        fragmented_regime = lambda1_share <= 0.48 and return_dispersion >= 0.0025 and sign_entropy >= 0.45
        recovery_friendly_regime = bool(fragmented_regime and not common_factor_regime)
        rows.append(
            {
                "date": merged.iloc[idx]["date"],
                "asset_count": len(return_cols),
                "lambda1": lambda1,
                "lambda1_share": lambda1_share,
                "eigengap": eigengap,
                "return_consensus": return_consensus,
                "return_dispersion": return_dispersion,
                "sign_entropy": sign_entropy,
                "common_factor_regime": common_factor_regime,
                "fragmented_regime": fragmented_regime,
                "recovery_friendly_regime": recovery_friendly_regime,
            }
        )
    return pd.DataFrame(rows)


def _infer_panel_time_col(frame: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise SystemExit(f"--panel-time-col {explicit!r} not present in panel")
        return explicit
    for candidate in ("timestamp", "date"):
        if candidate in frame.columns:
            return candidate
    raise SystemExit("could not infer panel time column; pass --panel-time-col")


def _compute_cross_asset_context_from_panel(
    panel_path: Path,
    *,
    panel_time_col: str | None,
    join_mode: str,
    rolling_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(panel_path)
    time_col = _infer_panel_time_col(panel, panel_time_col)
    parsed_time = _coerce_ts_to_timestamp(panel[time_col])
    panel = panel.assign(timestamp=parsed_time).dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    return_cols = [col for col in panel.columns if col.endswith("__return")]
    if len(return_cols) < 2:
        raise SystemExit(f"{panel_path}: need at least two *__return columns for cross-asset context")
    observed_cols = {col[:-10]: col for col in panel.columns if col.endswith("__observed")}
    asset_names = [col[:-8] for col in return_cols]

    values = panel[return_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for idx in range(len(panel)):
        active_indices = [
            i
            for i, name in enumerate(asset_names)
            if observed_cols.get(name) is None
            or int(_safe_float(panel.iloc[idx][observed_cols[name]], 0.0)) == 1
        ]
        active_count = len(active_indices)
        if active_count < 2:
            record = {
                "timestamp": panel.iloc[idx]["timestamp"],
                "asset_count": active_count,
                "lambda1": np.nan,
                "lambda1_share": np.nan,
                "eigengap": np.nan,
                "return_consensus": np.nan,
                "return_dispersion": np.nan,
                "sign_entropy": np.nan,
                "common_factor_regime": False,
                "fragmented_regime": False,
                "recovery_friendly_regime": False,
            }
            if join_mode == "date":
                record["date"] = pd.Timestamp(record["timestamp"]).normalize()
            rows.append(record)
            continue
        start = max(0, idx - rolling_window + 1)
        window = values[start : idx + 1, :][:, active_indices]
        corr = _safe_corr_matrix(window)
        eigvals = np.sort(np.real(np.linalg.eigvalsh(corr)))[::-1]
        lambda1 = float(eigvals[0]) if len(eigvals) else 1.0
        eigsum = float(np.sum(eigvals)) if len(eigvals) else 1.0
        lambda1_share = lambda1 / max(eigsum, 1e-12)
        eigengap = float(eigvals[0] - eigvals[1]) if len(eigvals) > 1 else lambda1
        row_ret = values[idx, active_indices]
        return_consensus = float(abs(np.mean(np.sign(row_ret))))
        return_dispersion = float(np.std(row_ret))
        sign_entropy = _sign_entropy(row_ret)
        common_factor_regime = lambda1_share >= 0.62 and return_consensus >= 0.67
        fragmented_regime = lambda1_share <= 0.48 and return_dispersion >= 0.0025 and sign_entropy >= 0.45
        recovery_friendly_regime = bool(fragmented_regime and not common_factor_regime)
        record = {
            "timestamp": panel.iloc[idx]["timestamp"],
            "asset_count": active_count,
            "lambda1": lambda1,
            "lambda1_share": lambda1_share,
            "eigengap": eigengap,
            "return_consensus": return_consensus,
            "return_dispersion": return_dispersion,
            "sign_entropy": sign_entropy,
            "common_factor_regime": common_factor_regime,
            "fragmented_regime": fragmented_regime,
            "recovery_friendly_regime": recovery_friendly_regime,
        }
        if join_mode == "date":
            record["date"] = pd.Timestamp(record["timestamp"]).normalize()
        rows.append(record)
    metadata = {
        "context_source": "panel_csv",
        "panel_csv": str(panel_path),
        "panel_time_col": time_col,
        "panel_return_columns": return_cols,
        "panel_row_count": int(len(panel)),
    }
    return pd.DataFrame(rows), metadata


def _classify_row_pair(rows: pd.DataFrame) -> dict[str, Any]:
    episode = rows[rows.get("in_episode", 0).astype(bool)].copy().reset_index(drop=True)
    if len(episode) < 2:
        return {
            "cross_asset_warning_class": "insufficient_prefix",
            "cross_asset_warning_reason": "need at least entry row plus warning row",
        }

    warning = episode.iloc[1].to_dict()
    response = episode.iloc[2].to_dict() if len(episode) > 2 else {}
    response_present = len(episode) > 2

    warning_family = str(warning.get("family", ""))
    response_family = str(response.get("family", ""))
    warning_expected = _safe_float(warning.get("executed_expected_surplus"))
    response_expected = _safe_float(response.get("executed_expected_surplus"))
    warning_dead = bool(warning.get("support_collapsed", warning_expected <= 1e-9))
    response_dead = bool(response.get("support_collapsed", response_expected <= 1e-9))
    response_restored = response_present and response_expected > 1e-9 and not response_dead
    warning_flatten = warning_family == "flatten_transition"
    response_flatten = response_present and response_family == "flatten_transition"
    response_common_factor = response.get("common_factor_regime", False) is True
    response_fragmented = response.get("fragmented_regime", False) is True
    response_recovery_friendly = response.get("recovery_friendly_regime", False) is True
    response_has_context = not pd.isna(response.get("date")) and not pd.isna(response.get("lambda1_share"))

    if warning_flatten or response_flatten:
        klass = "immediate_flatten"
        reason = "warning or response row is already terminal flatten"
    elif response_present and not response_has_context:
        klass = "no_cross_asset_context"
        reason = "response row has no aligned cross-asset market context"
    elif warning_dead and response_present and response_dead:
        klass = "confirmed_collapse_global" if response_common_factor else "confirmed_collapse_local"
        reason = (
            "dead support confirmed under common-factor cross-asset alignment"
            if response_common_factor
            else "dead support confirmed without common-factor market alignment"
        )
    elif warning_dead and response_restored:
        global_support = response_fragmented or response_recovery_friendly
        klass = "recovering_after_warning_global_support" if global_support else "recovering_after_warning_local"
        reason = (
            "support restored on the next row in a fragmented/recovery-friendly regime"
            if global_support
            else "support restored on the next row without fragmented cross-asset help"
        )
    elif not response_present:
        klass = "warning_without_response"
        reason = "warning row exists but no response row is available"
    else:
        klass = "uncertain_pair"
        reason = "warning/response pair does not resolve cleanly under cross-asset context"

    return {
        "warning_t": int(_safe_float(warning.get("sort_t"), 0.0)),
        "warning_date": str(warning.get("date", "")),
        "warning_family": warning_family,
        "warning_support_dead": warning_dead,
        "warning_lambda1_share": _safe_float(warning.get("lambda1_share")),
        "response_t": int(_safe_float(response.get("sort_t"), 0.0)) if response_present else None,
        "response_date": str(response.get("date", "")) if response_present else None,
        "response_family": response_family if response_present else None,
        "response_support_dead": response_dead if response_present else None,
        "response_lambda1_share": _safe_float(response.get("lambda1_share")) if response_present else None,
        "response_common_factor_regime": response_common_factor if response_present else None,
        "response_fragmented_regime": response_fragmented if response_present else None,
        "response_recovery_friendly_regime": response_recovery_friendly if response_present else None,
        "cross_asset_warning_class": klass,
        "cross_asset_warning_reason": reason,
    }


def _annotate_rows(
    rows: pd.DataFrame,
    context: pd.DataFrame,
    *,
    join_mode: str,
    timestamp_join: str,
    timestamp_tolerance: str,
) -> pd.DataFrame:
    out = rows.copy()
    out["timestamp"] = _coerce_ts_to_timestamp(out["ts"])
    out["date"] = out["timestamp"].dt.normalize()
    out["_row_order"] = np.arange(len(out))
    if join_mode == "timestamp":
        ctx = context.sort_values("timestamp", kind="stable").reset_index(drop=True)
        dated = out[out["timestamp"].notna()].sort_values("timestamp", kind="stable").reset_index(drop=True)
        undated = out[out["timestamp"].isna()].copy()
        if not dated.empty:
            if timestamp_join == "backward":
                merged = pd.merge_asof(
                    dated,
                    ctx,
                    on="timestamp",
                    direction="backward",
                    tolerance=pd.Timedelta(timestamp_tolerance),
                )
            else:
                merged = dated.merge(ctx, on="timestamp", how="left")
        else:
            merged = dated
    else:
        ctx = context.sort_values("date", kind="stable").reset_index(drop=True)
        dated = out[out["date"].notna()].sort_values("date", kind="stable").reset_index(drop=True)
        undated = out[out["date"].isna()].copy()
        merged = pd.merge_asof(dated, ctx, on="date", direction="backward") if not dated.empty else dated
    combined = pd.concat([merged, undated], ignore_index=True, sort=False)
    combined = combined.sort_values("_row_order", kind="stable").drop(columns="_row_order").reset_index(drop=True)
    return combined


def main() -> None:
    args = parse_args()
    paths = [Path(raw) for raw in args.summary_json]
    labels = list(args.label or [])
    if labels and len(labels) != len(paths):
        raise SystemExit("--label count must match --summary-json count when provided")
    if not labels:
        labels = [path.stem for path in paths]

    context_metadata: dict[str, Any]
    if args.panel_csv:
        context, context_metadata = _compute_cross_asset_context_from_panel(
            Path(args.panel_csv),
            panel_time_col=args.panel_time_col,
            join_mode=str(args.join_mode),
            rolling_window=int(args.rolling_window),
        )
        asset_paths: dict[str, Path] = {}
    else:
        asset_paths = _normalize_assets(args.asset)
        context = _compute_cross_asset_context(asset_paths, rolling_window=int(args.rolling_window))
        context_metadata = {
            "context_source": "daily_assets",
            "assets": {name: str(path) for name, path in asset_paths.items()},
        }

    records: list[dict[str, Any]] = []
    row_tables: list[pd.DataFrame] = []
    for label, path in zip(labels, paths, strict=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        selected = list(data.get("selected") or [])
        for episode in selected:
            rows = pd.DataFrame(episode.get("rows") or [])
            if rows.empty:
                continue
            annotated = _annotate_rows(
                rows,
                context,
                join_mode=str(args.join_mode),
                timestamp_join=str(args.timestamp_join),
                timestamp_tolerance=str(args.timestamp_tolerance),
            )
            row_pair = _classify_row_pair(annotated)
            records.append(
                {
                    "label": label,
                    "artifact": str(path),
                    "episode_id": int(_safe_float(episode.get("episode_id"), -1)),
                    "episode_type": str(episode.get("episode_type", "")),
                    "duration_rows": int(_safe_float(episode.get("duration_rows"), 0.0)),
                    "realized_surplus_sum": _safe_float(episode.get("realized_surplus_sum")),
                    "executed_expected_surplus_sum": _safe_float(episode.get("executed_expected_surplus_sum")),
                    "entry_reason": str(episode.get("entry_reason", "")),
                    "exit_reason": str(episode.get("exit_reason", "")),
                    "cross_asset_row_pair": row_pair,
                }
            )
            subset = annotated[annotated.get("in_episode", 0).astype(bool)].copy()
            subset.insert(0, "label", label)
            if "episode_id" not in subset.columns:
                subset.insert(1, "episode_id", int(_safe_float(episode.get("episode_id"), -1)))
            row_tables.append(subset)

    combined_rows = pd.concat(row_tables, ignore_index=True) if row_tables else pd.DataFrame()
    class_counts: dict[str, int] = {}
    for record in records:
        klass = str(record["cross_asset_row_pair"]["cross_asset_warning_class"])
        class_counts[klass] = class_counts.get(klass, 0) + 1

    summary = {
        "assets": {name: str(path) for name, path in asset_paths.items()},
        "context_metadata": context_metadata,
        "join_mode": str(args.join_mode),
        "timestamp_join": str(args.timestamp_join),
        "timestamp_tolerance": str(args.timestamp_tolerance),
        "artifact_count": len(paths),
        "episode_count": len(records),
        "class_counts": class_counts,
        "artifacts": records,
    }

    if args.output_csv and not combined_rows.empty:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined_rows.to_csv(out, index=False)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

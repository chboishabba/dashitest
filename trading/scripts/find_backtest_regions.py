from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.compute_gate_metrics import compute_metrics as compute_gate_metrics
    from scripts.sweep_tau_conf import compute_metrics as compute_accept_metrics
    from scripts.sweep_tau_conf import compute_pnl_metrics, compute_return_splits
except ModuleNotFoundError:  # pragma: no cover
    from compute_gate_metrics import compute_metrics as compute_gate_metrics
    from sweep_tau_conf import compute_metrics as compute_accept_metrics
    from sweep_tau_conf import compute_pnl_metrics, compute_return_splits


def list_default_logs(log_dir: Path) -> list[Path]:
    candidates = []
    for path in sorted(log_dir.glob("trading_log*.csv")):
        name = path.name
        if "trades_" in name:
            continue
        if "_all" in name:
            continue
        candidates.append(path)
    return candidates


def prepare_log(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "ts" in frame.columns:
        frame["ts_dt"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    else:
        frame["ts_dt"] = pd.NaT
    return frame


def window_metrics(window: pd.DataFrame) -> dict[str, float]:
    window = window.copy()
    for col, default in {
        "ell": np.nan,
        "action": 0,
        "hold": 1,
        "acceptable": False,
        "price": np.nan,
        "fill": 0.0,
        "fee": 0.0,
        "slippage": 0.0,
    }.items():
        if col not in window.columns:
            window[col] = default
    if "pnl" in window.columns:
        pnl = pd.to_numeric(window["pnl"], errors="coerce").fillna(method="ffill").fillna(0.0)
        window["pnl"] = pnl - float(pnl.iloc[0])
    metrics = {}
    metrics.update(compute_gate_metrics(window))
    if {"acceptable", "action", "hold"}.issubset(window.columns):
        metrics.update(compute_accept_metrics(window))
    metrics.update(compute_pnl_metrics(window))
    metrics.update(compute_return_splits(window))
    return metrics


def analyze_log(path: Path, *, window: int, stride: int, min_trades: int) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    if len(df) < window:
        return []
    df = prepare_log(df)
    rows: list[dict[str, object]] = []
    for start in range(0, len(df) - window + 1, stride):
        stop = start + window
        sub = df.iloc[start:stop].copy()
        metrics = window_metrics(sub)
        trades = int(metrics.get("trades", 0) or 0)
        if trades < min_trades:
            continue
        ts_valid = sub["ts_dt"].dropna() if "ts_dt" in sub.columns else pd.Series(dtype="datetime64[ns, UTC]")
        start_ts = ts_valid.iloc[0].isoformat() if not ts_valid.empty else ""
        end_ts = ts_valid.iloc[-1].isoformat() if not ts_valid.empty else ""
        rows.append(
            {
                "log_path": str(path),
                "source": str(sub.get("source", pd.Series([""])).iloc[0]) if "source" in sub.columns else path.stem,
                "start_idx": int(start),
                "end_idx": int(stop - 1),
                "start_ts": start_ts,
                "end_ts": end_ts,
                **metrics,
            }
        )
    return rows


def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    positive_cols = [
        "acceptable_pct",
        "precision",
        "recall",
        "pnl_net",
        "edge_per_turnover",
        "ret_engaged",
    ]
    negative_cols = ["max_dd", "flip_rate", "hold_pct"]

    score_cols: list[pd.Series] = []
    for col in positive_cols:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        score_cols.append(vals.rank(pct=True, method="average"))
    for col in negative_cols:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if col == "max_dd":
            vals = vals.abs()
        score_cols.append(1.0 - vals.rank(pct=True, method="average"))
    if score_cols:
        score_df = pd.concat(score_cols, axis=1)
        frame["region_score"] = score_df.mean(axis=1, skipna=True)
    else:
        frame["region_score"] = np.nan
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank good historical regions for backtesting using standard repo metrics.")
    ap.add_argument("--log", type=Path, action="append", default=None, help="Specific per-step log CSV(s) to analyze.")
    ap.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory to search for trading_log*.csv.")
    ap.add_argument("--window", type=int, default=512, help="Rolling window size in rows.")
    ap.add_argument("--stride", type=int, default=128, help="Step between windows in rows.")
    ap.add_argument("--min-trades", type=int, default=3, help="Minimum number of non-zero fill rows per window.")
    ap.add_argument("--top-k", type=int, default=20, help="Number of top windows to print.")
    ap.add_argument("--out", type=Path, default=Path("logs/backtest_regions.csv"), help="CSV output path.")
    args = ap.parse_args()

    logs = args.log or list_default_logs(args.log_dir)
    all_rows: list[dict[str, object]] = []
    for path in logs:
        if not path.exists():
            continue
        all_rows.extend(analyze_log(path, window=args.window, stride=args.stride, min_trades=args.min_trades))

    if not all_rows:
        raise SystemExit("No candidate windows found. Lower --min-trades or adjust --window.")

    df = pd.DataFrame(all_rows)
    df = add_composite_score(df)
    df = df.sort_values(
        ["region_score", "pnl_net", "precision", "ret_engaged"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    display_cols = [
        "source",
        "start_ts",
        "end_ts",
        "region_score",
        "pnl_net",
        "max_dd",
        "precision",
        "recall",
        "acceptable_pct",
        "ret_engaged",
        "ret_flat",
        "edge_per_turnover",
        "trades",
        "hold_pct",
        "log_path",
    ]
    cols = [c for c in display_cols if c in df.columns]
    print(df.loc[: args.top_k - 1, cols].to_string(index=False))
    print(f"\nWrote ranked regions to {args.out}")


if __name__ == "__main__":
    main()

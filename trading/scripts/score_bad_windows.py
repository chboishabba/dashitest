"""
Compute severity-ranked bad windows and synthetic “bad” labels from a trading log.

Outputs:
- per-bar synthetic bad flags (abs return > k*sigma or drawdown slope > thr)
- top-N windows ranked by severity (sum p_bad over window)

Usage:
  PYTHONPATH=. python trading/scripts/score_bad_windows.py --log logs/trading_log.csv --out logs/bad_windows.csv
"""

import argparse
import pathlib

import numpy as np
import pandas as pd


def compute_synthetic_bad(df, k_sigma=3.0, dd_slope_thr=0.001):
    # returns
    df = df.copy()
    df["ret"] = df["price"].pct_change().fillna(0.0)
    sigma = df["ret"].rolling(50).std().fillna(method="bfill").fillna(0.0)
    df["dd"] = df["pnl"].cummax() - df["pnl"]
    dd_slope = df["dd"].diff().fillna(0.0)
    df["synthetic_bad"] = (
        (df["ret"].abs() > k_sigma * sigma)
        | (dd_slope > dd_slope_thr)
    ).astype(int)
    return df


def rank_windows(df, p_bad_thr=0.7, window=500, top_n=20):
    # severity = sum p_bad over window; window indexed by rows
    p_bad = df["p_bad"].fillna(0.0).to_numpy()
    ts = df["ts"] if "ts" in df.columns else None
    sev = pd.Series(p_bad).rolling(window, min_periods=1).sum()
    rows = []
    for idx, s in sev.tail(len(sev) - window + 1).reset_index(drop=True).items():
        end_idx = idx + window - 1
        start_idx = idx
        rows.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "sev_sum_p_bad": float(s),
                "p_bad_mean": float(np.mean(p_bad[start_idx:end_idx + 1])),
                "p_bad_max": float(np.max(p_bad[start_idx:end_idx + 1])),
                "bad_rate": float(np.mean(df["bad_flag"].iloc[start_idx:end_idx + 1])),
                "synthetic_bad_rate": float(np.mean(df["synthetic_bad"].iloc[start_idx:end_idx + 1])),
                "ts_start": ts.iloc[start_idx] if ts is not None else np.nan,
                "ts_end": ts.iloc[end_idx] if ts is not None else np.nan,
            }
        )
    top = sorted(rows, key=lambda r: r["sev_sum_p_bad"], reverse=True)[:top_n]
    return pd.DataFrame(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=pathlib.Path, default=pathlib.Path("logs/trading_log.csv"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("logs/bad_windows.csv"))
    ap.add_argument("--window", type=int, default=500, help="Window length (bars) to aggregate severity.")
    ap.add_argument("--top", type=int, default=20, help="Top-N windows to keep.")
    ap.add_argument("--p-bad-thr", type=float, default=0.7, help="Threshold for bad_flag; informational.")
    ap.add_argument("--k-sigma", type=float, default=3.0, help="Threshold multiplier for synthetic bad return.")
    ap.add_argument("--dd-slope-thr", type=float, default=0.001, help="Drawdown slope threshold for synthetic bad.")
    args = ap.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"log not found: {args.log}")

    df = pd.read_csv(args.log)
    df = compute_synthetic_bad(df, k_sigma=args.k_sigma, dd_slope_thr=args.dd_slope_thr)
    top = rank_windows(df, p_bad_thr=args.p_bad_thr, window=args.window, top_n=args.top)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.out, index=False)
    print(top.to_string(index=False))
    print(f"Wrote top-{args.top} windows to {args.out}")


if __name__ == "__main__":
    main()

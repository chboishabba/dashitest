"""
Aggregate trading logs into daily bad-day scores for news correlation.

Usage:
  PYTHONPATH=. python scripts/rollup_bad_days.py --log logs/trading_log.csv --out logs/bad_days.csv --top 10
"""

import argparse
import pathlib

import numpy as np
import pandas as pd


def longest_run(values: pd.Series) -> int:
    """Return the longest consecutive run length of truthy values."""
    run = best = 0
    for v in values:
        if bool(v):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def rollup_bad_days(df: pd.DataFrame, alpha=1.0, beta=0.5, gamma=0.1) -> pd.DataFrame:
    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        # fabricate a time index if missing; days become bar buckets
        df = df.copy()
        df["ts"] = pd.to_datetime(df["t"], unit="s", errors="coerce")
    df = df.dropna(subset=["ts"])
    df["date"] = df["ts"].dt.date

    def day_return(s):
        start = s.iloc[0]
        end = s.iloc[-1]
        return (end - start) / (start if start != 0 else np.nan)

    grouped = df.groupby("date")
    summary = grouped.agg(
        start_ts=("ts", "min"),
        end_ts=("ts", "max"),
        bars=("t", "count"),
        price_start=("price", "first"),
        price_end=("price", "last"),
        pnl_start=("pnl", "first"),
        pnl_end=("pnl", "last"),
        ret_day=("pnl", day_return),
        mean_p_bad=("p_bad", "mean"),
        max_p_bad=("p_bad", "max"),
        bad_rate=("bad_flag", "mean"),
    )
    summary["bad_run"] = grouped["bad_flag"].apply(longest_run)
    summary["bad_score"] = (
        alpha * summary["mean_p_bad"].fillna(0)
        + beta * summary["max_p_bad"].fillna(0)
        + gamma * summary["bad_run"].fillna(0)
    )
    summary = summary.sort_values("bad_score", ascending=False)
    return summary.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=pathlib.Path, default=pathlib.Path("logs/trading_log.csv"))
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Optional CSV output for the daily rollup.")
    ap.add_argument("--top", type=int, default=0, help="Print the top-N worst days by bad_score (0=all).")
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight for mean_p_bad in bad_score.")
    ap.add_argument("--beta", type=float, default=0.5, help="Weight for max_p_bad in bad_score.")
    ap.add_argument("--gamma", type=float, default=0.1, help="Weight for bad_run (longest streak) in bad_score.")
    args = ap.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"log not found: {args.log}")

    df = pd.read_csv(args.log)
    rollup = rollup_bad_days(df, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    if args.top and args.top > 0:
        print(rollup.head(args.top).to_string(index=False))
    else:
        print(rollup.to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rollup.to_csv(args.out, index=False)
        print(f"Wrote daily rollup to {args.out}")


if __name__ == "__main__":
    main()

"""
score_quotient_stability.py
---------------------------
Report quotient drift inside stable permission segments vs transition points.

Usage:
  PYTHONPATH=. python trading/scripts/score_quotient_stability.py --log logs/trading_log.csv --min-run 20
"""

import argparse

import numpy as np
import pandas as pd


Q_COLS = ["q_e64", "q_c64", "q_s64", "q_de", "q_dc", "q_ds"]


def load_log(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Failed to read log {path}: {e}")


def summarize_drift(df: pd.DataFrame, min_run: int) -> dict[str, float]:
    missing = [col for col in Q_COLS + ["permission"] if col not in df.columns]
    if missing:
        raise SystemExit(f"Log is missing columns: {', '.join(missing)}")

    q = df[Q_COLS].apply(pd.to_numeric, errors="coerce")
    dq = q.diff().abs()
    perm = pd.to_numeric(df["permission"], errors="coerce")
    segment_id = (perm != perm.shift(1)).cumsum()
    segment_sizes = segment_id.value_counts()
    stable_segments = segment_sizes[segment_sizes >= min_run].index
    stable_mask = segment_id.isin(stable_segments)
    transition_mask = perm != perm.shift(1)

    summary = {}
    for col in Q_COLS:
        stable_mean = float(dq.loc[stable_mask, col].mean(skipna=True))
        transition_mean = float(dq.loc[transition_mask, col].mean(skipna=True))
        ratio = transition_mean / stable_mean if stable_mean > 0 else np.nan
        summary[f"{col}_stable_mean"] = stable_mean
        summary[f"{col}_transition_mean"] = transition_mean
        summary[f"{col}_transition_ratio"] = ratio
    summary["stable_rows"] = int(stable_mask.sum())
    summary["transition_rows"] = int(transition_mask.sum())
    summary["min_run"] = int(min_run)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log CSV with q_* fields")
    ap.add_argument("--min-run", type=int, default=20, help="Minimum stable segment length")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    args = ap.parse_args()

    df = load_log(args.log)
    summary = summarize_drift(df, min_run=args.min_run)
    summary_df = pd.DataFrame([summary])
    print(summary_df.to_string(index=False))
    if args.out:
        summary_df.to_csv(args.out, index=False)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

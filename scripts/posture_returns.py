"""
Compute bar- and window-level returns by posture (ACT/HOLD/BAN) and show cumulative sums.

Usage:
  PYTHONPATH=. python scripts/posture_returns.py --log logs/trading_log_msft.us.csv

Inputs: CSV with columns like equity/pnl, ban, hold, action.
Posture inference (priority): ban→BAN (-1), else hold→HOLD (0), else ACT (+1).
Return default: equity.diff(); fallback to pnl; else price.diff().
"""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def infer_posture(df: pd.DataFrame) -> pd.Series:
    """Return posture series g ∈ {−1,0,+1} with BAN priority over HOLD."""
    g = pd.Series(1, index=df.index, dtype=int)  # default ACT
    if "ban" in df:
        g[df["ban"] > 0] = -1
    if "hold" in df:
        g[(g != -1) & (df["hold"] > 0)] = 0
    # Fallback: if no hold column, treat zero actions as HOLD
    elif "action" in df:
        g[(g != -1) & (df["action"] == 0)] = 0
    return g


def infer_returns(df: pd.DataFrame, ret_col: str | None) -> Tuple[pd.Series, str]:
    """Pick a return series and describe the choice."""
    if ret_col:
        return df[ret_col], ret_col
    if "equity" in df:
        return df["equity"].diff().fillna(0), "equity.diff()"
    if "pnl" in df:
        return df["pnl"], "pnl"
    if "price" in df:
        return df["price"].diff().fillna(0), "price.diff()"
    raise ValueError("No usable return column found; specify --ret-col")


def windows_by_posture(df: pd.DataFrame):
    starts = df.index[(df["g"].shift(1) != df["g"]) | (df.index == 0)]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(df)
        yield s, e, df.iloc[s:e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to trading log CSV")
    ap.add_argument(
        "--ret-col",
        type=str,
        default=None,
        help="Optional explicit return column to use (overrides auto inference)",
    )
    args = ap.parse_args()

    path = Path(args.log)
    df = pd.read_csv(path)

    df["g"] = infer_posture(df)
    df["ret"], chosen = infer_returns(df, args.ret_col)

    print(f"[posture_returns] log={path} rows={len(df)} return_source={chosen}")

    # Bar-level stats by posture
    bar_stats = (
        df.groupby("g")["ret"]
        .agg(mean="mean", std="std", sum="sum", count="count")
        .rename(index={-1: "BAN", 0: "HOLD", 1: "ACT"})
    )
    print("\nBar-level returns by posture:")
    print(bar_stats.to_string(float_format=lambda x: f"{x: .6f}"))

    # Window-level stats and cumulative sums
    rows = []
    for s, e, w in windows_by_posture(df):
        rows.append(
            {
                "posture": w["g"].iloc[0],
                "start": s,
                "end": e - 1,
                "n": len(w),
                "ret_sum": w["ret"].sum(),
                "ret_mean": w["ret"].mean(),
            }
        )
    win = pd.DataFrame(rows)
    win["cum_ret_all"] = win["ret_sum"].cumsum()
    win["cum_ret_by_posture"] = win.groupby("posture")["ret_sum"].cumsum()

    win["posture"] = win["posture"].map({-1: "BAN", 0: "HOLD", 1: "ACT"})
    print("\nWindow-level returns (cumulative adds each window):")
    print(
        win[["posture", "start", "end", "n", "ret_sum", "cum_ret_all", "cum_ret_by_posture"]]
        .to_string(index=False, float_format=lambda x: f"{x: .6f}")
    )

    posture_win_stats = (
        win.groupby("posture")[["ret_sum", "ret_mean", "n"]]
        .agg(["mean", "std", "count"])
    )
    print("\nWindow summary by posture:")
    print(posture_win_stats.to_string(float_format=lambda x: f"{x: .6f}"))


if __name__ == "__main__":
    main()

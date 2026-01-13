"""
plot_proposal_diagnostics.py
----------------------------
Quick diagnostics for proposal logs.

Plots:
  1) ell deciles vs mean delta_pnl_signed (stderr)
  2) ell deciles vs direction hit-rate
  3) veto vs non-veto delta_pnl_signed hist overlay

Usage:
  PYTHONPATH=. python trading/scripts/plot_proposal_diagnostics.py \
    --log logs/proposals_btc.us.csv \
    --save-prefix logs/plots/proposals_btc.us
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from plot_utils import timestamped_prefix
except ModuleNotFoundError:
    from scripts.plot_utils import timestamped_prefix


def _decile_stats(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    df = df[np.isfinite(df["ell"]) & np.isfinite(df["delta_pnl_signed"])]
    if df.empty:
        raise SystemExit("No finite ell/delta_pnl_signed rows.")
    df["decile"] = pd.qcut(df["ell"], bins, duplicates="drop")
    grouped = df.groupby("decile")["delta_pnl_signed"]
    stats = grouped.agg(["mean", "std", "count"]).reset_index()
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    stats["center"] = stats["decile"].apply(lambda x: float(x.mid))
    return stats


def _hit_rate_by_decile(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    df = df[np.isfinite(df["ell"]) & np.isfinite(df["delta_pnl"]) & (df["dir_pred"] != 0)]
    if df.empty:
        raise SystemExit("No finite rows for hit-rate.")
    df["decile"] = pd.qcut(df["ell"], bins, duplicates="drop")
    hits = (np.sign(df["delta_pnl"]) == df["dir_pred"]).astype(int)
    grouped = df.groupby("decile")
    out = grouped.apply(lambda g: hits.loc[g.index].mean()).reset_index(name="hit_rate")
    out["center"] = out["decile"].apply(lambda x: float(x.mid))
    return out


def _plot_decile_mean(stats: pd.DataFrame, save_path: Path | None) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(
        stats["center"],
        stats["mean"],
        yerr=stats["stderr"],
        fmt="o-",
        capsize=3,
        color="tab:blue",
    )
    plt.xlabel("ell decile center")
    plt.ylabel("mean delta_pnl_signed")
    plt.title("ell deciles vs mean signed future return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def _plot_hit_rate(stats: pd.DataFrame, save_path: Path | None) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(stats["center"], stats["hit_rate"], "o-", color="tab:green")
    plt.xlabel("ell decile center")
    plt.ylabel("direction hit-rate")
    plt.title("ell deciles vs direction hit-rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def _plot_veto_hist(df: pd.DataFrame, save_path: Path | None) -> None:
    df = df[np.isfinite(df["delta_pnl_signed"])]
    if df.empty:
        raise SystemExit("No finite delta_pnl_signed rows.")
    veto = df[df["veto"] > 0]["delta_pnl_signed"].to_numpy()
    allow = df[df["veto"] == 0]["delta_pnl_signed"].to_numpy()
    if veto.size == 0 or allow.size == 0:
        raise SystemExit("Need both veto and non-veto samples for histogram.")
    bins = max(20, int(math.sqrt(df.shape[0])))
    plt.figure(figsize=(7, 4.5))
    plt.hist(allow, bins=bins, alpha=0.6, label="allowed", density=True, color="tab:blue")
    plt.hist(veto, bins=bins, alpha=0.6, label="vetoed", density=True, color="tab:red")
    plt.xlabel("delta_pnl_signed")
    plt.ylabel("density")
    plt.title("Veto impact on signed future return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostics for proposal logs.")
    ap.add_argument("--log", type=Path, required=True, help="Proposal log CSV.")
    ap.add_argument("--bins", type=int, default=10, help="Number of ell bins.")
    ap.add_argument("--save-prefix", type=Path, default=None, help="Optional output prefix.")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    required = {"ell", "dir_pred", "delta_pnl", "delta_pnl_signed", "veto"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    stats = _decile_stats(df[df["dir_pred"] != 0], args.bins)
    hit = _hit_rate_by_decile(df, args.bins)

    prefix = None
    if args.save_prefix is not None:
        prefix = Path(timestamped_prefix(str(args.save_prefix)))
        prefix.parent.mkdir(parents=True, exist_ok=True)

    mean_path = hit_path = veto_path = None
    if prefix is not None:
        mean_path = prefix.with_name(f"{prefix.name}_decile_mean.png")
        hit_path = prefix.with_name(f"{prefix.name}_hit_rate.png")
        veto_path = prefix.with_name(f"{prefix.name}_veto_hist.png")

    _plot_decile_mean(stats, mean_path)
    _plot_hit_rate(hit, hit_path)
    _plot_veto_hist(df, veto_path)


if __name__ == "__main__":
    main()

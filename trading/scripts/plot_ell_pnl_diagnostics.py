"""
plot_ell_pnl_diagnostics.py
---------------------------
PnL diagnostics conditioned on ell and posture.

Produces:
  1) ell decile -> mean delta_pnl with error bars
  2) ell decile -> Sharpe-like proxy
  3) ACT vs HOLD delta_pnl histogram overlay

Usage:
  PYTHONPATH=. python trading/scripts/plot_ell_pnl_diagnostics.py --log logs/trading_log.csv --save-prefix logs/plots/ell_pnl
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


def _infer_delta_pnl(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "delta_pnl" in df:
        return pd.to_numeric(df["delta_pnl"], errors="coerce").fillna(0.0), "delta_pnl"
    if "pnl" in df:
        pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
        return pnl.diff().fillna(0.0), "pnl.diff()"
    if "equity" in df:
        equity = pd.to_numeric(df["equity"], errors="coerce").fillna(method="ffill").fillna(0.0)
        return equity.diff().fillna(0.0), "equity.diff()"
    raise SystemExit("No usable PnL series found. Need delta_pnl, pnl, or equity.")


def _infer_action(df: pd.DataFrame) -> pd.Series:
    if "action" in df:
        return pd.to_numeric(df["action"], errors="coerce").fillna(0.0)
    if "hold" in df:
        return (pd.to_numeric(df["hold"], errors="coerce").fillna(0.0) == 0).astype(int)
    raise SystemExit("No action/hold columns found; need action or hold for ACT/HOLD split.")


def _decile_stats(ell: np.ndarray, delta_pnl: np.ndarray, bins: int) -> pd.DataFrame:
    df = pd.DataFrame({"ell": ell, "delta_pnl": delta_pnl})
    df = df[np.isfinite(df["ell"]) & np.isfinite(df["delta_pnl"])]
    if df.empty:
        raise SystemExit("No finite ell/delta_pnl pairs found.")

    try:
        df["decile"] = pd.qcut(df["ell"], bins, duplicates="drop")
    except ValueError as exc:
        raise SystemExit(f"Failed to bin ell into deciles: {exc}") from exc

    grouped = df.groupby("decile")["delta_pnl"]
    stats = grouped.agg(["mean", "std", "count"])
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    stats["sharpe_like"] = stats["mean"] / stats["std"].replace(0.0, np.nan) * np.sqrt(stats["count"])
    stats = stats.reset_index()
    stats["center"] = stats["decile"].apply(lambda x: float(x.mid))
    return stats


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
    plt.ylabel("mean delta_pnl")
    plt.title("ell deciles vs mean delta_pnl (stderr)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def _plot_decile_sharpe(stats: pd.DataFrame, save_path: Path | None) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(stats["center"], stats["sharpe_like"], "o-", color="tab:green")
    plt.xlabel("ell decile center")
    plt.ylabel("sharpe-like (mean/std * sqrt(n))")
    plt.title("ell deciles vs sharpe-like proxy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def _plot_act_hold(delta_pnl: np.ndarray, act_mask: np.ndarray, save_path: Path | None) -> None:
    act = delta_pnl[act_mask]
    hold = delta_pnl[~act_mask]
    if act.size == 0 or hold.size == 0:
        raise SystemExit("Need both ACT and HOLD samples for histogram.")
    bins = max(20, int(math.sqrt(delta_pnl.size)))
    plt.figure(figsize=(7, 4.5))
    plt.hist(hold, bins=bins, alpha=0.6, label="HOLD", density=True, color="tab:orange")
    plt.hist(act, bins=bins, alpha=0.6, label="ACT", density=True, color="tab:blue")
    plt.xlabel("delta_pnl")
    plt.ylabel("density")
    plt.title("ACT vs HOLD delta_pnl distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="PnL diagnostics conditioned on ell and posture.")
    ap.add_argument("--log", type=Path, required=True, help="Trading log CSV with ell + pnl.")
    ap.add_argument("--bins", type=int, default=10, help="Number of ell bins (default deciles).")
    ap.add_argument("--tau-on", type=float, default=0.5, help="ell threshold for ACT region (optional stats).")
    ap.add_argument("--tau-off", type=float, default=0.49, help="ell threshold for HOLD region (optional stats).")
    ap.add_argument("--save-prefix", type=Path, default=None, help="Optional output prefix (without extension).")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    if "ell" not in df:
        raise SystemExit("log must contain ell column")

    ell = pd.to_numeric(df["ell"], errors="coerce").to_numpy(dtype=float)
    delta_pnl, pnl_source = _infer_delta_pnl(df)
    delta_pnl = delta_pnl.to_numpy(dtype=float)
    action = _infer_action(df).to_numpy(dtype=float)
    act_mask = np.isfinite(action) & (action != 0)

    stats = _decile_stats(ell, delta_pnl, args.bins)

    prefix = None
    if args.save_prefix is not None:
        prefix = Path(timestamped_prefix(str(args.save_prefix)))
        prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ell_pnl] log={args.log} rows={len(df)} pnl_source={pnl_source}")
    print(stats[["decile", "mean", "std", "count", "sharpe_like"]].to_string(index=False))

    if prefix is not None:
        decile_path = prefix.with_name(f"{prefix.name}_decile_mean.png")
        sharpe_path = prefix.with_name(f"{prefix.name}_decile_sharpe.png")
        hist_path = prefix.with_name(f"{prefix.name}_act_hold_hist.png")
    else:
        decile_path = sharpe_path = hist_path = None

    _plot_decile_mean(stats, decile_path)
    _plot_decile_sharpe(stats, sharpe_path)
    _plot_act_hold(delta_pnl[np.isfinite(delta_pnl)], act_mask[np.isfinite(delta_pnl)], hist_path)

    # Optional region stats for quick checks
    finite = np.isfinite(ell) & np.isfinite(delta_pnl)
    near = finite & (ell >= args.tau_off) & (ell <= args.tau_on)
    deep_act = finite & (ell > args.tau_on)
    deep_hold = finite & (ell < args.tau_off)
    for name, mask in [("near", near), ("deep_act", deep_act), ("deep_hold", deep_hold)]:
        if mask.any():
            mean = float(np.nanmean(delta_pnl[mask]))
            std = float(np.nanstd(delta_pnl[mask]))
            print(f"[ell_pnl] {name}: mean={mean:.6f} std={std:.6f} n={int(mask.sum())}")


if __name__ == "__main__":
    main()

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
    df = df[np.isfinite(df["ell"]) & np.isfinite(df["delta_pnl_signed"])].copy()
    if df.empty:
        raise SystemExit("No finite ell/delta_pnl_signed rows.")
    df["decile"] = pd.qcut(df["ell"], bins, duplicates="drop")
    grouped = df.groupby("decile", observed=False)["delta_pnl_signed"]
    stats = grouped.agg(["mean", "std", "count"]).reset_index()
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    stats["center"] = stats["decile"].apply(lambda x: float(x.mid))
    return stats


def _hit_rate_by_decile(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    df = df[np.isfinite(df["ell"]) & np.isfinite(df["delta_pnl"]) & (df["dir_pred"] != 0)].copy()
    if df.empty:
        raise SystemExit("No finite rows for hit-rate.")
    df["decile"] = pd.qcut(df["ell"], bins, duplicates="drop")
    hits = (np.sign(df["delta_pnl"]) == df["dir_pred"]).astype(int)
    grouped = df.groupby("decile", observed=False)
    out = grouped.apply(lambda g: hits.loc[g.index].mean(), include_groups=False).reset_index(name="hit_rate")
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


def _plot_left_tail(df: pd.DataFrame, alpha: float, save_path: Path | None) -> None:
    df = df[np.isfinite(df["delta_pnl_signed"])]
    if df.empty:
        raise SystemExit("No finite delta_pnl_signed rows.")
    allowed = df[df["veto"] == 0]["delta_pnl_signed"].to_numpy()
    vetoed = df[df["veto"] > 0]["delta_pnl_signed"].to_numpy()
    if allowed.size == 0 or vetoed.size == 0:
        raise SystemExit("Need both veto and non-veto samples for tail plot.")
    q_allow = np.quantile(allowed, alpha)
    q_veto = np.quantile(vetoed, alpha)
    tail_allow = allowed[allowed <= q_allow]
    tail_veto = vetoed[vetoed <= q_veto]
    bins = max(15, int(math.sqrt(tail_allow.size + tail_veto.size)))
    plt.figure(figsize=(7, 4.5))
    plt.hist(tail_allow, bins=bins, alpha=0.6, label=f"allowed tail@{alpha:.2f}", density=True, color="tab:blue")
    plt.hist(tail_veto, bins=bins, alpha=0.6, label=f"vetoed tail@{alpha:.2f}", density=True, color="tab:red")
    plt.xlabel("delta_pnl_signed (tail)")
    plt.ylabel("density")
    plt.title("Left-tail comparison (allowed vs vetoed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def _print_veto_summary(df: pd.DataFrame, alpha: float) -> None:
    df = df[np.isfinite(df["delta_pnl_signed"])].copy()
    if "veto_reason" not in df.columns:
        return
    total = df.shape[0]
    print("\nVeto summary (tail stats):")
    rows = []
    for reason, grp in df.groupby("veto_reason", observed=False):
        vals = grp["delta_pnl_signed"].to_numpy()
        if vals.size == 0:
            continue
        q = float(np.quantile(vals, alpha))
        tail = vals[vals <= q]
        tail_mean = float(np.mean(tail)) if tail.size else float("nan")
        rows.append(
            {
                "reason": reason,
                "count": int(vals.size),
                "rate": float(vals.size / total),
                "q": q,
                "tail_mean": tail_mean,
            }
        )
    summary = pd.DataFrame(rows).sort_values("count", ascending=False)
    if not summary.empty:
        print(summary.to_string(index=False, float_format=lambda x: f"{x: .6f}"))


def _plot_instrument_by_decile(df: pd.DataFrame, col: str, bins: int, title: str, save_path: Path | None) -> None:
    df = df[np.isfinite(df[col]) & df["instrument_pred"].notna()].copy()
    if df.empty:
        return
    df["decile"] = pd.qcut(df[col], bins, duplicates="drop")
    counts = df.groupby(["decile", "instrument_pred"], observed=False).size().unstack(fill_value=0)
    centers = [float(x.mid) for x in counts.index]
    plt.figure(figsize=(7.5, 4.8))
    bottom = np.zeros(len(centers), dtype=float)
    for name in counts.columns:
        vals = counts[name].to_numpy(dtype=float)
        plt.bar(centers, vals, bottom=bottom, width=0.8 * (centers[1] - centers[0]) if len(centers) > 1 else 0.05, label=name)
        bottom += vals
    plt.xlabel(f"{col} decile center")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def _plot_size_by_hazard(df: pd.DataFrame, bins: int, save_path: Path | None) -> None:
    if "size_pred" not in df.columns or "hazard" not in df.columns:
        return
    df = df[np.isfinite(df["hazard"])].copy()
    if df.empty:
        return
    size_map = {"0": 0.0, "0.5": 0.5, "1": 1.0, "2": 2.0}
    df["size_val"] = df["size_pred"].map(size_map).fillna(0.0)
    df["decile"] = pd.qcut(df["hazard"], bins, duplicates="drop")
    stats = df.groupby("decile", observed=False)["size_val"].mean().reset_index()
    centers = [float(x.mid) for x in stats["decile"]]
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(centers, stats["size_val"], "o-", color="tab:purple")
    plt.xlabel("hazard decile center")
    plt.ylabel("mean size_pred")
    plt.title("Size vs hazard deciles")
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
    ap.add_argument("--tail-alpha", type=float, default=0.10, help="Tail quantile for diagnostics.")
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

    mean_path = hit_path = veto_path = tail_path = None
    iv_path = hazard_path = size_hazard_path = None
    if prefix is not None:
        mean_path = prefix.with_name(f"{prefix.name}_decile_mean.png")
        hit_path = prefix.with_name(f"{prefix.name}_hit_rate.png")
        veto_path = prefix.with_name(f"{prefix.name}_veto_hist.png")
        tail_path = prefix.with_name(f"{prefix.name}_left_tail.png")
        iv_path = prefix.with_name(f"{prefix.name}_instrument_by_iv.png")
        hazard_path = prefix.with_name(f"{prefix.name}_instrument_by_hazard.png")
        size_hazard_path = prefix.with_name(f"{prefix.name}_size_by_hazard.png")

    _plot_decile_mean(stats, mean_path)
    _plot_hit_rate(hit, hit_path)
    _plot_veto_hist(df, veto_path)
    _plot_left_tail(df, args.tail_alpha, tail_path)
    _print_veto_summary(df, args.tail_alpha)

    if "instrument_pred" in df.columns and "opt_mark_iv_p50" in df.columns:
        _plot_instrument_by_decile(
            df,
            "opt_mark_iv_p50",
            args.bins,
            "Instrument selection by IV decile",
            iv_path,
        )
    if "instrument_pred" in df.columns and "hazard" in df.columns:
        _plot_instrument_by_decile(
            df,
            "hazard",
            args.bins,
            "Instrument selection by hazard decile",
            hazard_path,
        )
    _plot_size_by_hazard(df, args.bins, size_hazard_path)


if __name__ == "__main__":
    main()

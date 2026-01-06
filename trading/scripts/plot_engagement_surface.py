"""
plot_engagement_surface.py
--------------------------
Visualize engagement surfaces produced by sweep scripts (`*_surface.csv`).
Each CSV is expected to have columns: tau_off, bin_center, engagement (0..1).
Plots a heatmap of P(ACT | acceptable, actionability) over (actionability_bin × tau_off).

Usage:
  PYTHONPATH=. python trading/scripts/plot_engagement_surface.py --left logs/engagement_surface.csv --right logs/motif_surface.csv --save surface.png
  (left/right optional; single heatmap if only left is provided)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import timestamped_path


def load_surface(path: str):
    df = pd.read_csv(path)
    required = {"tau_off", "bin_center", "engagement"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {required}")
    table = df.pivot_table(index="tau_off", columns="bin_center", values="engagement", aggfunc="mean")
    table = table.sort_index().sort_index(axis=1)
    return table


def plot_surface(ax, table: pd.DataFrame, title: str):
    # build grid for imshow
    taus = table.index.to_numpy(dtype=float)
    bins = table.columns.to_numpy(dtype=float)
    Z = table.to_numpy()
    # extend edges for nicer extent
    if len(bins) > 1:
        dx = np.diff(bins).mean()
    else:
        dx = 0.05
    if len(taus) > 1:
        dy = np.diff(taus).mean()
    else:
        dy = 0.05
    extent = [
        bins.min() - dx * 0.5,
        bins.max() + dx * 0.5,
        taus.min() - dy * 0.5,
        taus.max() + dy * 0.5,
    ]
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xlabel("actionability (bin center)")
    ax.set_ylabel("tau_off")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(ACT | acceptable)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=str, required=True, help="CSV with tau_off, bin_center, engagement")
    ap.add_argument("--right", type=str, default=None, help="Optional second CSV for side-by-side comparison")
    ap.add_argument("--left-label", type=str, default="left", help="Title for left plot")
    ap.add_argument("--right-label", type=str, default="right", help="Title for right plot")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    left_table = load_surface(args.left)
    right_table = load_surface(args.right) if args.right else None

    if right_table is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        plot_surface(axes[0], left_table, args.left_label)
        plot_surface(axes[1], right_table, args.right_label)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        plot_surface(ax, left_table, args.left_label)

    if args.save:
        save_path = timestamped_path(args.save)
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    main()

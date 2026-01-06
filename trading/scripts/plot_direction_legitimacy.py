"""
plot_direction_legitimacy.py
----------------------------
Direction–legitimacy coupling (orthogonality test).

Option A: 2x2 counts of (acceptable/unacceptable) × (buy/sell).
Option B: Signed heatmap: x=actionability, y=signed direction (-1..+1), color=P(acceptable).

Usage:
  PYTHONPATH=. python trading/scripts/plot_direction_legitimacy.py --log logs/trading_log.csv --save dir_legit.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_utils import timestamped_path


def plot_counts(df, ax):
    acc = df["acceptable"].astype(bool)
    acts = df["action"]
    buy = acts == 1
    sell = acts == -1
    counts = {
        "acc & buy": (acc & buy).sum(),
        "acc & sell": (acc & sell).sum(),
        "unacc & buy": ((~acc) & buy).sum(),
        "unacc & sell": ((~acc) & sell).sum(),
    }
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    ax.bar(labels, vals, color=["steelblue", "teal", "salmon", "tomato"])
    ax.set_title("Counts: acceptable vs direction")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)


def plot_heatmap(df, ax, bins=20):
    actionability = pd.to_numeric(df["actionability"], errors="coerce")
    direction = pd.to_numeric(df["action"], errors="coerce")
    acceptable = df["acceptable"].astype(bool)
    mask = np.isfinite(actionability) & np.isfinite(direction)
    actionability = actionability[mask]
    direction = direction[mask]
    acceptable = acceptable[mask]
    if actionability.empty:
        raise SystemExit("No valid data for heatmap.")
    x_edges = np.linspace(0.0, 1.0, bins + 1)
    y_edges = np.linspace(-1.5, 1.5, 4)  # bins for -1,0,+1
    sum_acc, _, _ = np.histogram2d(direction, actionability, bins=[y_edges, x_edges], weights=acceptable.astype(int))
    counts, _, _ = np.histogram2d(direction, actionability, bins=[y_edges, x_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, sum_acc / counts, np.nan)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xlabel("actionability")
    ax.set_ylabel("direction (-1,0,+1)")
    ax.set_title("P(acceptable) over direction × actionability")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log CSV with acceptable/action/actionability")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    ap.add_argument("--bins", type=int, default=20, help="Actionability bins for heatmap")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    for col in ("acceptable", "action", "actionability"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in log.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    plot_counts(df, axes[0])
    plot_heatmap(df, axes[1], bins=args.bins)

    if args.save:
        save_path = timestamped_path(args.save)
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    main()

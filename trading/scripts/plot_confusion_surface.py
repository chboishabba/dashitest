"""
plot_confusion_surface.py
-------------------------
Plot false-positive and false-negative heatmaps over (actionability bin × tau_off).
Expects CSV from sweep_confusion_surface.py with columns: tau_off, bin_center, fp_rate, fn_rate.

Usage:
  PYTHONPATH=. python trading/scripts/plot_confusion_surface.py --csv logs/confusion_surface.csv --save confusion.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tau_off", "bin_center", "fp_rate", "fn_rate"}
    if not required.issubset(df.columns):
        raise SystemExit(f"{path} missing required columns {required}")
    return df


def pivot(df: pd.DataFrame, value: str):
    table = df.pivot_table(index="tau_off", columns="bin_center", values=value, aggfunc="mean")
    return table.sort_index().sort_index(axis=1)


def plot_heat(ax, table: pd.DataFrame, title: str, vmin=0.0, vmax=1.0):
    x = table.columns.to_numpy(dtype=float)
    y = table.index.to_numpy(dtype=float)
    Z = table.to_numpy()
    dx = np.diff(x).mean() if len(x) > 1 else 0.05
    dy = np.diff(y).mean() if len(y) > 1 else 0.05
    extent = [x.min() - dx * 0.5, x.max() + dx * 0.5, y.min() - dy * 0.5, y.max() + dy * 0.5]
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="magma")
    ax.set_xlabel("actionability (bin center)")
    ax.set_ylabel("tau_off")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV from sweep_confusion_surface.py")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    df = load_table(args.csv)
    fp_table = pivot(df, "fp_rate")
    fn_table = pivot(df, "fn_rate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    plot_heat(axes[0], fp_table, "False-positive rate (ACT & ¬acceptable)", vmin=0.0, vmax=1.0)
    plot_heat(axes[1], fn_table, "False-negative rate (HOLD & acceptable)", vmin=0.0, vmax=1.0)

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()

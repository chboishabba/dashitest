"""
plot_regime_surface.py
----------------------
Plot acceptable% over RegimeSpec sweeps (min_run_length × vol cap percentile).
Expects CSV from sweep_regime_acceptability.py with columns:
  min_run_length, max_vol_pct, acceptable_pct

Usage:
  PYTHONPATH=. python scripts/plot_regime_surface.py --csv logs/accept_surface.csv --save regime_surface.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"min_run_length", "max_vol_pct", "acceptable_pct"}
    if not required.issubset(df.columns):
        raise SystemExit(f"{path} missing required columns {required}")
    return df


def pivot_surface(df: pd.DataFrame):
    table = df.pivot_table(
        index="min_run_length",
        columns="max_vol_pct",
        values="acceptable_pct",
        aggfunc="mean",
    )
    table = table.sort_index().sort_index(axis=1)
    return table


def plot_heatmap(table: pd.DataFrame, title: str, save: str = None):
    x = table.columns.to_numpy()
    y = table.index.to_numpy()
    Z = table.to_numpy()
    if len(x) > 1:
        dx = np.diff(x[np.isfinite(x)]).mean() if np.isfinite(x).any() else 1.0
    else:
        dx = 1.0
    if len(y) > 1:
        dy = np.diff(y).mean()
    else:
        dy = 1.0
    extent = [
        np.nanmin(x) - dx * 0.5,
        np.nanmax(x) + dx * 0.5,
        y.min() - dy * 0.5,
        y.max() + dy * 0.5,
    ]
    plt.figure(figsize=(6, 5))
    im = plt.imshow(Z, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")
    plt.xlabel("max_vol_pct (None as nan)")
    plt.ylabel("min_run_length")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("acceptable%")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"Saved {save}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV from sweep_regime_acceptability.py")
    ap.add_argument("--save", type=str, default=None, help="Optional image output path")
    args = ap.parse_args()

    df = load_table(args.csv)
    table = pivot_surface(df)
    plot_heatmap(table, title="Acceptable% over RegimeSpec sweep", save=args.save)


if __name__ == "__main__":
    main()

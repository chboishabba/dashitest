"""
plot_regime_surface.py
----------------------
Plot acceptable% over RegimeSpec sweeps.
If max_flip_rate is present, prefer plotting (min_run_length × max_flip_rate).
Else plot (min_run_length × max_vol_pct).
Expects CSV from sweep_regime_acceptability.py with columns:
  min_run_length, acceptable_pct, and either max_flip_rate or max_vol_pct

Usage:
  PYTHONPATH=. python trading/scripts/plot_regime_surface.py --csv logs/accept_surface.csv --save regime_surface.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import timestamped_path


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_base = {"min_run_length", "acceptable_pct"}
    if not required_base.issubset(df.columns):
        raise SystemExit(f"{path} missing required columns {required_base}")
    if "max_flip_rate" in df.columns:
        df["max_flip_rate"] = df["max_flip_rate"]
    if "max_vol_pct" in df.columns:
        df["max_vol_pct"] = df["max_vol_pct"]
    return df


def pivot_surface(df: pd.DataFrame):
    if "max_flip_rate" in df.columns and df["max_flip_rate"].notna().any():
        table = df.pivot_table(
            index="min_run_length",
            columns="max_flip_rate",
            values="acceptable_pct",
            aggfunc="mean",
        )
    elif "max_vol_pct" in df.columns:
        table = df.pivot_table(
            index="min_run_length",
            columns="max_vol_pct",
            values="acceptable_pct",
            aggfunc="mean",
        )
    else:
        raise SystemExit("No sweep dimension found (max_flip_rate or max_vol_pct).")
    return table.sort_index().sort_index(axis=1)


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
        save_path = timestamped_path(save)
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
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

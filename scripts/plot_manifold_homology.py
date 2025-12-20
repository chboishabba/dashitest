"""
plot_manifold_homology.py
-------------------------
Trader vs CA acceptable manifold overlap (homology check).
Build binary masks over (time × actionability bins) for trader and motif CA,
then plot overlap/mismatch:
  - overlap: acceptable in both
  - trader-only: acceptable trader, not CA
  - CA-only: acceptable CA, not trader

Inputs:
  - trader log with t, actionability, acceptable
  - motif CA log (same structure) or a surface CSV with acceptable_density over bins

Usage:
  PYTHONPATH=. python scripts/plot_manifold_homology.py --trader logs/trading_log.csv --ca logs/motif_surface.csv --save homology.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mask_from_log(path: str, time_bins=100, act_bins=20):
    df = pd.read_csv(path)
    if not {"acceptable", "actionability", "t"}.issubset(df.columns):
        raise SystemExit(f"{path} must contain acceptable, actionability, t")
    t = pd.to_numeric(df["t"], errors="coerce")
    a = pd.to_numeric(df["actionability"], errors="coerce")
    acc = df["acceptable"].astype(bool)
    mask = np.isfinite(t) & np.isfinite(a)
    t = t[mask]
    a = a[mask]
    acc = acc[mask]
    time_edges = np.linspace(t.min(), t.max(), time_bins + 1)
    act_edges = np.linspace(0.0, 1.0, act_bins + 1)
    counts, _, _ = np.histogram2d(t, a, bins=[time_edges, act_edges])
    acc_counts, _, _ = np.histogram2d(t, a, bins=[time_edges, act_edges], weights=acc.astype(int))
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(counts > 0, acc_counts / counts, np.nan)
    mask_grid = np.where(density >= 0.5, 1, 0)  # majority acceptable in bin
    return mask_grid, time_edges, act_edges


def mask_from_surface(path: str):
    df = pd.read_csv(path)
    required = {"tau_off", "bin_center", "engagement"}
    if not required.issubset(df.columns):
        raise SystemExit(f"{path} must have tau_off, bin_center, engagement")
    # Use tau_off as y and bin_center as x; treat engagement > 0.5 as acceptable proxy
    table = df.pivot_table(index="tau_off", columns="bin_center", values="engagement", aggfunc="mean")
    table = table.sort_index().sort_index(axis=1)
    mask = (table.to_numpy() > 0.5).astype(int)
    tau_edges = np.linspace(table.index.min(), table.index.max(), len(table.index) + 1)
    bin_edges = np.linspace(table.columns.min(), table.columns.max(), len(table.columns) + 1)
    return mask, tau_edges, bin_edges


def plot_overlap(mask_trader, mask_ca, x_edges, y_edges, title, save=None):
    overlap = (mask_trader == 1) & (mask_ca == 1)
    trader_only = (mask_trader == 1) & (mask_ca == 0)
    ca_only = (mask_trader == 0) & (mask_ca == 1)

    grid = np.full_like(mask_trader, fill_value=-1, dtype=int)
    grid[overlap] = 2
    grid[trader_only] = 1
    grid[ca_only] = 0

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    cmap = plt.get_cmap("Set1", 3)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(grid, origin="lower", aspect="auto", extent=extent, vmin=-0.5, vmax=2.5, cmap=cmap)
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["CA-only", "Trader-only", "Overlap"])
    plt.xlabel("actionability bin center")
    plt.ylabel("time/tau axis")
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"Saved {save}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trader", type=str, required=True, help="Trader log CSV (t, actionability, acceptable)")
    ap.add_argument("--ca", type=str, required=True, help="CA surface CSV (bin_center, tau_off, engagement) or CA log with acceptable/actionability/t")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    ap.add_argument("--time_bins", type=int, default=100, help="Time bins for trader log")
    ap.add_argument("--act_bins", type=int, default=20, help="Actionability bins for trader log")
    args = ap.parse_args()

    mask_trader, t_edges, a_edges = mask_from_log(args.trader, time_bins=args.time_bins, act_bins=args.act_bins)
    # Try CA surface; fall back to log parsing
    try:
        mask_ca, y_edges, x_edges = mask_from_surface(args.ca)
    except Exception:
        mask_ca, y_edges, x_edges = mask_from_log(args.ca, time_bins=args.time_bins, act_bins=args.act_bins)

    # Align shapes by interpolation via resizing (nearest)
    from scipy.ndimage import zoom

    scale_y = mask_trader.shape[0] / mask_ca.shape[0]
    scale_x = mask_trader.shape[1] / mask_ca.shape[1]
    mask_ca_resized = zoom(mask_ca, zoom=(scale_y, scale_x), order=0)
    plot_overlap(mask_trader, mask_ca_resized, a_edges, t_edges, title="Trader vs CA acceptable manifold overlap", save=args.save)


if __name__ == "__main__":
    main()

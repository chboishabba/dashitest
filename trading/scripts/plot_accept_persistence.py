"""
plot_accept_persistence.py
--------------------------
Temporal persistence surface: heatmap of consecutive acceptable run-lengths.
x: actionability (binned)
y: time
color: run_length(acceptable)

Usage:
  PYTHONPATH=. python trading/scripts/plot_accept_persistence.py --log logs/trading_log.csv --save accept_persistence.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def acceptable_runs(acc: np.ndarray) -> np.ndarray:
    runs = np.zeros_like(acc, dtype=int)
    run = 0
    for i, a in enumerate(acc):
        if a:
            run += 1
        else:
            run = 0
        runs[i] = run
    return runs


def build_heatmap(t, actionability, runs, time_bins=100, act_bins=20):
    mask = np.isfinite(t) & np.isfinite(actionability) & np.isfinite(runs)
    t = t[mask]
    actionability = actionability[mask]
    runs = runs[mask]
    if t.size == 0:
        raise SystemExit("No valid data for heatmap.")
    time_edges = np.linspace(t.min(), t.max(), time_bins + 1)
    act_edges = np.linspace(0.0, 1.0, act_bins + 1)
    sum_runs, _, _ = np.histogram2d(t, actionability, bins=[time_edges, act_edges], weights=runs)
    counts, _, _ = np.histogram2d(t, actionability, bins=[time_edges, act_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, sum_runs / counts, np.nan)
    return grid, time_edges, act_edges


def plot_heatmap(grid, time_edges, act_edges, title, save=None):
    extent = [act_edges[0], act_edges[-1], time_edges[0], time_edges[-1]]
    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="plasma",
    )
    plt.xlabel("actionability")
    plt.ylabel("time (t)")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("acceptable run-length (bars)")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"Saved {save}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log CSV with acceptable/actionability/t")
    ap.add_argument("--time_bins", type=int, default=100, help="Time bins")
    ap.add_argument("--act_bins", type=int, default=20, help="Actionability bins")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    for col in ("acceptable", "actionability", "t"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in log.")

    acc = df["acceptable"].astype(bool).to_numpy()
    runs = acceptable_runs(acc)
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    actionability = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)

    grid, te, ae = build_heatmap(t, actionability, runs, time_bins=args.time_bins, act_bins=args.act_bins)
    plot_heatmap(grid, te, ae, title="Acceptable run-length (time × actionability)", save=args.save)


if __name__ == "__main__":
    main()

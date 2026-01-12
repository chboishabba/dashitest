"""
plot_acceptability.py
---------------------
Visualize the acceptable manifold from a trading log.
Heatmap: time bins × actionability bins -> acceptable density.

Usage:
  PYTHONPATH=. python trading/scripts/plot_acceptability.py --log logs/trading_log.csv --save acceptable.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_utils import timestamped_path


def load_log(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Failed to read log {path}: {e}")


def build_heatmap(df: pd.DataFrame, time_bins: int, act_bins: int, clip: float = None):
    if "acceptable" not in df or "actionability" not in df or "t" not in df:
        raise SystemExit("Log must contain columns: acceptable, actionability, t")
    acc = df["acceptable"].astype(bool)
    act = pd.to_numeric(df["actionability"], errors="coerce")
    t = pd.to_numeric(df["t"], errors="coerce")
    mask = acc.notna() & act.notna() & t.notna()
    acc = acc[mask]
    act = act[mask]
    t = t[mask]
    if acc.empty:
        raise SystemExit("No valid rows after filtering.")

    # bin edges
    time_edges = np.linspace(t.min(), t.max(), time_bins + 1)
    act_edges = np.linspace(0.0, 1.0, act_bins + 1)

    # 2D histogram weighted by acceptable
    hist_acc, _, _ = np.histogram2d(t, act, bins=[time_edges, act_edges], weights=acc.astype(int))
    hist_count, _, _ = np.histogram2d(t, act, bins=[time_edges, act_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(hist_count > 0, hist_acc / hist_count, np.nan)
    if clip is not None:
        density = np.clip(density, 0.0, clip)
    return density, time_edges, act_edges


def plot_heatmap(density, time_edges, act_edges, title: str, save: str = None):
    # extent for imshow
    extent = [act_edges[0], act_edges[-1], time_edges[0], time_edges[-1]]
    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        density.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    plt.xlabel("actionability")
    plt.ylabel("time (t)")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("P(acceptable)")
    plt.tight_layout()
    if save:
        save_path = timestamped_path(save)
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log CSV with acceptable/actionability/t")
    ap.add_argument("--time_bins", type=int, default=100, help="Number of time bins")
    ap.add_argument("--act_bins", type=int, default=20, help="Number of actionability bins")
    ap.add_argument("--clip", type=float, default=None, help="Optional clip for density to improve contrast")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    df = load_log(args.log)
    density, t_edges, a_edges = build_heatmap(df, args.time_bins, args.act_bins, clip=args.clip)
    plot_heatmap(density, t_edges, a_edges, title="Acceptable density (time × actionability)", save=args.save)


if __name__ == "__main__":
    main()

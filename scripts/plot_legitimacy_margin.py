"""
plot_legitimacy_margin.py
-------------------------
Boundary-thickness map: heatmap of distance to RegimeSpec failure.
Computes per-bar margin = min distance to violating any RegimeSpec constraint
and plots it over (time × actionability).

Inputs:
  - trading log with columns: t, actionability, state, price (for vol)
  - RegimeSpec params: min_run_length, max_flip_rate, max_vol, window

Usage:
  PYTHONPATH=. python scripts/plot_legitimacy_margin.py --log logs/trading_log.csv --save margin.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regime import RegimeSpec


def sign_run_lengths(states: np.ndarray) -> np.ndarray:
    runs = np.zeros(len(states), dtype=int)
    run = 0
    for i, s in enumerate(states):
        if s == 0:
            run = 0
        else:
            if i > 0 and s == states[i - 1]:
                run += 1
            else:
                run = 1
        runs[i] = run
    return runs


def flip_rate_series(states: np.ndarray, window: int) -> np.ndarray:
    flips = np.abs(np.diff(states, prepend=states[0])) > 0
    fr = np.zeros_like(states, dtype=float)
    w = max(1, window)
    for i in range(len(states)):
        lo = max(0, i - w + 1)
        span = max(1, i - lo)
        fr[i] = flips[lo:i].sum() / span if span > 0 else 0.0
    return fr


def vol_series(prices: np.ndarray, window: int) -> np.ndarray:
    rets = np.diff(prices, prepend=prices[0])
    return pd.Series(rets).rolling(window).std().to_numpy()


def compute_margin(states, prices, spec: RegimeSpec):
    runs = sign_run_lengths(states)
    flips = flip_rate_series(states, spec.window)
    vols = vol_series(prices, spec.window)

    big = 1e9
    margins = np.full_like(states, fill_value=big, dtype=float)

    if spec.min_run_length is not None:
        margins = np.minimum(margins, runs - spec.min_run_length)
    if spec.max_flip_rate is not None:
        margins = np.minimum(margins, spec.max_flip_rate - flips)
    if spec.max_vol is not None:
        margins = np.minimum(margins, spec.max_vol - vols)
    return margins


def build_heatmap(t, actionability, margin, time_bins=100, act_bins=20, clip=None):
    mask = np.isfinite(t) & np.isfinite(actionability) & np.isfinite(margin)
    t = t[mask]
    actionability = actionability[mask]
    margin = margin[mask]
    if t.size == 0:
        raise SystemExit("No valid data for heatmap.")
    if clip is not None:
        margin = np.clip(margin, -clip, clip)
    time_edges = np.linspace(t.min(), t.max(), time_bins + 1)
    act_edges = np.linspace(0.0, 1.0, act_bins + 1)
    # average margin per bin
    sum_margin, _, _ = np.histogram2d(t, actionability, bins=[time_edges, act_edges], weights=margin)
    counts, _, _ = np.histogram2d(t, actionability, bins=[time_edges, act_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, sum_margin / counts, np.nan)
    return grid, time_edges, act_edges


def plot_heatmap(grid, time_edges, act_edges, title, save=None):
    extent = [act_edges[0], act_edges[-1], time_edges[0], time_edges[-1]]
    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
    )
    plt.xlabel("actionability")
    plt.ylabel("time (t)")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("legitimacy margin (>0 deep inside; <0 violating)")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"Saved {save}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log CSV with t,state,price,actionability")
    ap.add_argument("--min_run_length", type=int, default=3, help="RegimeSpec min_run_length")
    ap.add_argument("--max_flip_rate", type=float, default=None, help="RegimeSpec max_flip_rate")
    ap.add_argument("--max_vol", type=float, default=None, help="RegimeSpec max_vol")
    ap.add_argument("--window", type=int, default=50, help="RegimeSpec window for flip/vol")
    ap.add_argument("--time_bins", type=int, default=100, help="Time bins")
    ap.add_argument("--act_bins", type=int, default=20, help="Actionability bins")
    ap.add_argument("--clip", type=float, default=5.0, help="Clip margin for visualization (+/-)")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    for col in ("t", "state", "price", "actionability"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in log.")

    states = pd.to_numeric(df["state"], errors="coerce").fillna(0).to_numpy(dtype=int)
    prices = pd.to_numeric(df["price"], errors="coerce").fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    actionability = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)

    spec = RegimeSpec(
        min_run_length=args.min_run_length,
        max_flip_rate=args.max_flip_rate,
        max_vol=args.max_vol,
        window=args.window,
    )
    margin = compute_margin(states, prices, spec)
    grid, te, ae = build_heatmap(t, actionability, margin, time_bins=args.time_bins, act_bins=args.act_bins, clip=args.clip)
    plot_heatmap(grid, te, ae, title="Legitimacy margin (distance to RegimeSpec failure)", save=args.save)


if __name__ == "__main__":
    main()

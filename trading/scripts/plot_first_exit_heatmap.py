"""
plot_first_exit_heatmap.py
--------------------------
First-exit time heatmap: expected time until legitimacy failure if you are here now.
Axes:
  x: actionability
  y: legitimacy margin (distance to nearest RegimeSpec violation)
Color:
  expected bars until unacceptable

Usage:
  PYTHONPATH=. python trading/scripts/plot_first_exit_heatmap.py --log logs/trading_log.csv --save first_exit.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_utils import timestamped_path
from trading.regime import RegimeSpec


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


def legitimacy_margin(states, prices, spec: RegimeSpec):
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


def first_exit_horizon(acceptable: np.ndarray) -> np.ndarray:
    """
    For each time t, compute steps until first unacceptable (including t if already unacceptable -> 0).
    If never unacceptable ahead, return NaN.
    """
    n = len(acceptable)
    next_unacc = np.full(n, fill_value=np.nan)
    last_unacc = np.nan
    for i in range(n - 1, -1, -1):
        if not acceptable[i]:
            last_unacc = i
            next_unacc[i] = 0
        else:
            next_unacc[i] = last_unacc - i if not np.isnan(last_unacc) else np.nan
    return next_unacc


def build_heatmap(actionability, margin, horizon, bins=20):
    mask = np.isfinite(actionability) & np.isfinite(margin) & np.isfinite(horizon)
    actionability = actionability[mask]
    margin = margin[mask]
    horizon = horizon[mask]
    if actionability.size == 0:
        raise SystemExit("No valid data for heatmap.")
    act_edges = np.linspace(0.0, 1.0, bins + 1)
    marg_lo, marg_hi = np.percentile(margin, [1, 99])
    marg_edges = np.linspace(marg_lo, marg_hi, bins + 1)
    sum_h, _, _ = np.histogram2d(margin, actionability, bins=[marg_edges, act_edges], weights=horizon)
    counts, _, _ = np.histogram2d(margin, actionability, bins=[marg_edges, act_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, sum_h / counts, np.nan)
    return grid, act_edges, marg_edges


def plot_heatmap(grid, act_edges, marg_edges, title, save=None):
    extent = [act_edges[0], act_edges[-1], marg_edges[0], marg_edges[-1]]
    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    plt.xlabel("actionability")
    plt.ylabel("legitimacy margin")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("expected bars until unacceptable")
    plt.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="margin=0 boundary")
    plt.legend()
    plt.tight_layout()
    if save:
        save_path = timestamped_path(save)
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log with acceptable/actionability/state/price")
    ap.add_argument("--min_run_length", type=int, default=3, help="RegimeSpec min_run_length")
    ap.add_argument("--max_flip_rate", type=float, default=None, help="RegimeSpec max_flip_rate")
    ap.add_argument("--max_vol", type=float, default=None, help="RegimeSpec max_vol")
    ap.add_argument("--window", type=int, default=50, help="RegimeSpec window")
    ap.add_argument("--bins", type=int, default=20, help="Bins for actionability/margin")
    ap.add_argument("--save", type=str, default=None, help="Optional image output path")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    for col in ("acceptable", "actionability", "state", "price"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in log.")

    acceptable = df["acceptable"].astype(bool).to_numpy()
    actionability = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)
    states = pd.to_numeric(df["state"], errors="coerce").fillna(0).to_numpy(dtype=int)
    prices = pd.to_numeric(df["price"], errors="coerce").fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)

    spec = RegimeSpec(
        min_run_length=args.min_run_length,
        max_flip_rate=args.max_flip_rate,
        max_vol=args.max_vol,
        window=args.window,
    )
    margin = legitimacy_margin(states, prices, spec)
    horizon = first_exit_horizon(acceptable)
    grid, act_edges, marg_edges = build_heatmap(actionability, margin, horizon, bins=args.bins)
    plot_heatmap(grid, act_edges, marg_edges, title="Expected bars until unacceptable", save=args.save)


if __name__ == "__main__":
    main()

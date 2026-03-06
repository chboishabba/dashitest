"""
plot_policy_curvature.py
------------------------
Policy curvature vs engagement.
Computes second difference of actionability (curvature) and bins by legitimacy margin.
Plots:
  - Heatmap: mean |curvature| over (actionability × margin) bins
  - Scatter: |curvature| vs actionability colored by ACT/HOLD

Usage:
  PYTHONPATH=. python scripts/plot_policy_curvature.py --log logs/trading_log.csv --save policy_curvature.png
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


def compute_curvature(actionability: np.ndarray):
    # central second difference: a_{t+1} - 2 a_t + a_{t-1}
    if len(actionability) < 3:
        return np.array([])
    a = actionability
    curv = a[2:] - 2 * a[1:-1] + a[:-2]
    return curv


def build_heatmap(act_mid, margin_mid, curv, bins=20):
    mask = np.isfinite(act_mid) & np.isfinite(margin_mid) & np.isfinite(curv)
    act_mid = act_mid[mask]
    margin_mid = margin_mid[mask]
    curv = np.abs(curv[mask])
    if act_mid.size == 0:
        raise SystemExit("No valid data for heatmap.")
    act_edges = np.linspace(0.0, 1.0, bins + 1)
    marg_lo, marg_hi = np.percentile(margin_mid, [1, 99])
    marg_edges = np.linspace(marg_lo, marg_hi, bins + 1)
    sum_c, _, _ = np.histogram2d(margin_mid, act_mid, bins=[marg_edges, act_edges], weights=curv)
    counts, _, _ = np.histogram2d(margin_mid, act_mid, bins=[marg_edges, act_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, sum_c / counts, np.nan)
    return grid, act_edges, marg_edges


def plot_results(act_mid, curv, state_mid, grid, act_edges, marg_edges, save=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # scatter
    colors = np.where(state_mid, "tab:orange", "tab:blue")
    axes[0].scatter(act_mid, np.abs(curv), c=colors, s=5, alpha=0.4)
    axes[0].set_xlabel("actionability")
    axes[0].set_ylabel("|curvature| (second diff)")
    axes[0].set_title("Curvature vs actionability (ACT=orange, HOLD=blue)")
    axes[0].grid(True, alpha=0.3)

    extent = [act_edges[0], act_edges[-1], marg_edges[0], marg_edges[-1]]
    im = axes[1].imshow(grid, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[1].set_xlabel("actionability")
    axes[1].set_ylabel("legitimacy margin")
    axes[1].set_title("Mean |curvature| over (actionability × margin)")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("mean |curvature|")

    if save:
        plt.savefig(save, dpi=200)
        print(f"Saved {save}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log with actionability/state/price")
    ap.add_argument("--min_run_length", type=int, default=3, help="RegimeSpec min_run_length")
    ap.add_argument("--max_flip_rate", type=float, default=None, help="RegimeSpec max_flip_rate")
    ap.add_argument("--max_vol", type=float, default=None, help="RegimeSpec max_vol")
    ap.add_argument("--window", type=int, default=50, help="RegimeSpec window")
    ap.add_argument("--bins", type=int, default=20, help="Bins for heatmap")
    ap.add_argument("--save", type=str, default=None, help="Optional image output path")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    for col in ("actionability", "state", "price", "action"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in log.")

    actionability = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)
    states = pd.to_numeric(df["state"], errors="coerce").fillna(0).to_numpy(dtype=int)
    prices = pd.to_numeric(df["price"], errors="coerce").fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
    control_state = (df["action"] != 0).to_numpy(dtype=bool)

    spec = RegimeSpec(
        min_run_length=args.min_run_length,
        max_flip_rate=args.max_flip_rate,
        max_vol=args.max_vol,
        window=args.window,
    )
    margin = legitimacy_margin(states, prices, spec)
    curv = compute_curvature(actionability)
    # align midpoints to t indices starting at 1..n-2
    act_mid = actionability[1:-1]
    marg_mid = margin[1:-1]
    state_mid = control_state[1:-1]

    grid, act_edges, marg_edges = build_heatmap(act_mid, marg_mid, curv, bins=args.bins)
    plot_results(act_mid, curv, state_mid, grid, act_edges, marg_edges, save=args.save)


if __name__ == "__main__":
    main()

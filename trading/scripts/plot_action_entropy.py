"""
plot_action_entropy.py
----------------------
Entropy of action choice over the manifold.
Computes H(ACT/HOLD) in bins of (actionability × margin) or (actionability × tau_off surrogate via margin).
Plots heatmap of entropy (0 = confident, 1 = max uncertainty).

Usage:
  PYTHONPATH=. python trading/scripts/plot_action_entropy.py --log logs/trading_log.csv --save action_entropy.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from plot_utils import timestamped_path
except ModuleNotFoundError:
    from scripts.plot_utils import timestamped_path

try:
    from trading.regime import RegimeSpec
except ModuleNotFoundError:
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


def entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def build_heatmap(actionability, margin, act_prob, bins=20):
    mask = np.isfinite(actionability) & np.isfinite(margin) & np.isfinite(act_prob)
    a = actionability[mask]
    m = margin[mask]
    p = act_prob[mask]
    if a.size == 0:
        raise SystemExit("No valid data for heatmap.")
    act_edges = np.linspace(0.0, 1.0, bins + 1)
    marg_lo, marg_hi = np.percentile(m, [1, 99])
    marg_edges = np.linspace(marg_lo, marg_hi, bins + 1)
    entropy_vals = entropy(p)
    sum_e, _, _ = np.histogram2d(m, a, bins=[marg_edges, act_edges], weights=entropy_vals)
    counts, _, _ = np.histogram2d(m, a, bins=[marg_edges, act_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, sum_e / counts, np.nan)
    return grid, act_edges, marg_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log with actionability/state/price/action")
    ap.add_argument("--actionability-col", type=str, default="actionability")
    ap.add_argument("--state-col", type=str, default="state")
    ap.add_argument("--price-col", type=str, default="price")
    ap.add_argument("--action-col", type=str, default="action")
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
    col_map = {
        "actionability": args.actionability_col,
        "state": args.state_col,
        "price": args.price_col,
        "action": args.action_col,
    }
    for name, col in col_map.items():
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' for {name}.")

    actionability = pd.to_numeric(df[col_map["actionability"]], errors="coerce").to_numpy(dtype=float)
    states = pd.to_numeric(df[col_map["state"]], errors="coerce").fillna(0).to_numpy(dtype=int)
    prices = pd.to_numeric(df[col_map["price"]], errors="coerce").fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
    act_flag = (df[col_map["action"]] != 0).to_numpy(dtype=int)

    # local ACT probability via rolling mean (simple smoother)
    act_prob = pd.Series(act_flag).rolling(window=5, min_periods=1, center=True).mean().to_numpy()

    spec = RegimeSpec(
        min_run_length=args.min_run_length,
        max_flip_rate=args.max_flip_rate,
        max_vol=args.max_vol,
        window=args.window,
    )
    margin = legitimacy_margin(states, prices, spec)
    grid, act_edges, marg_edges = build_heatmap(actionability, margin, act_prob, bins=args.bins)

    extent = [act_edges[0], act_edges[-1], marg_edges[0], marg_edges[-1]]
    plt.figure(figsize=(7, 5))
    im = plt.imshow(grid, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="inferno")
    plt.xlabel("actionability")
    plt.ylabel("legitimacy margin")
    plt.title("Entropy of ACT/HOLD over (actionability × margin)")
    cbar = plt.colorbar(im)
    cbar.set_label("entropy (bits)")
    plt.axhline(0.0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if args.save:
        save_path = timestamped_path(args.save)
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    main()

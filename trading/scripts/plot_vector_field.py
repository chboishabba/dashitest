"""
plot_vector_field.py
--------------------
Vector field over the acceptable manifold.
Bins actionability and legitimacy margin; computes mean deltas:
  dx = actionability(t+1) - actionability(t)
  dy = margin(t+1) - margin(t)
Plots arrows (quiver) colored by acceptable status.

Usage:
  PYTHONPATH=. python trading/scripts/plot_vector_field.py --log logs/trading_log.csv --save vector_field.png
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


def compute_vectors(df: pd.DataFrame, spec: RegimeSpec, bins=20):
    actionability = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)
    states = pd.to_numeric(df["state"], errors="coerce").fillna(0).to_numpy(dtype=int)
    prices = pd.to_numeric(df["price"], errors="coerce").fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
    acceptable = df["acceptable"].astype(bool).to_numpy() if "acceptable" in df else None

    margin = legitimacy_margin(states, prices, spec)

    # t and t+1
    act_t = actionability[:-1]
    act_t1 = actionability[1:]
    marg_t = margin[:-1]
    marg_t1 = margin[1:]
    acc_t = acceptable[:-1] if acceptable is not None else np.ones(len(act_t), dtype=bool)

    mask = np.isfinite(act_t) & np.isfinite(act_t1) & np.isfinite(marg_t) & np.isfinite(marg_t1)
    act_t = act_t[mask]
    act_t1 = act_t1[mask]
    marg_t = marg_t[mask]
    marg_t1 = marg_t1[mask]
    acc_t = acc_t[mask]

    dx = act_t1 - act_t
    dy = marg_t1 - marg_t

    # bins
    act_edges = np.linspace(0.0, 1.0, bins + 1)
    # margin bins: use percentiles to avoid extreme spread
    marg_valid = marg_t[np.isfinite(marg_t)]
    if marg_valid.size == 0:
        raise SystemExit("No finite margins.")
    lo, hi = np.percentile(marg_valid, [1, 99])
    marg_edges = np.linspace(lo, hi, bins + 1)

    # aggregate mean dx, dy per bin; track acceptable fraction for coloring
    bin_dx = np.zeros((bins, bins))
    bin_dy = np.zeros((bins, bins))
    bin_count = np.zeros((bins, bins))
    bin_acc = np.zeros((bins, bins))

    act_idx = np.digitize(act_t, act_edges) - 1
    marg_idx = np.digitize(marg_t, marg_edges) - 1
    for i in range(len(act_t)):
        ix = act_idx[i]
        iy = marg_idx[i]
        if ix < 0 or ix >= bins or iy < 0 or iy >= bins:
            continue
        bin_dx[iy, ix] += dx[i]
        bin_dy[iy, ix] += dy[i]
        bin_count[iy, ix] += 1
        bin_acc[iy, ix] += 1 if acc_t[i] else 0

    with np.errstate(invalid="ignore", divide="ignore"):
        bin_dx = np.where(bin_count > 0, bin_dx / bin_count, np.nan)
        bin_dy = np.where(bin_count > 0, bin_dy / bin_count, np.nan)
        bin_acc = np.where(bin_count > 0, bin_acc / bin_count, np.nan)

    act_centers = 0.5 * (act_edges[:-1] + act_edges[1:])
    marg_centers = 0.5 * (marg_edges[:-1] + marg_edges[1:])
    return act_centers, marg_centers, bin_dx, bin_dy, bin_acc


def plot_quiver(act_centers, marg_centers, dx, dy, acc_frac, title, save=None):
    X, Y = np.meshgrid(act_centers, marg_centers)
    mask = np.isfinite(dx) & np.isfinite(dy)
    Xp = X[mask]
    Yp = Y[mask]
    U = dx[mask]
    V = dy[mask]
    C = acc_frac[mask] if acc_frac is not None else np.zeros_like(U)
    plt.figure(figsize=(7, 5))
    q = plt.quiver(Xp, Yp, U, V, C, cmap="viridis", angles="xy", scale_units="xy", scale=1)
    plt.xlabel("actionability")
    plt.ylabel("legitimacy margin")
    plt.title(title)
    cbar = plt.colorbar(q)
    cbar.set_label("acceptable fraction in bin")
    plt.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="margin=0 (boundary)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        save_path = timestamped_path(save)
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log with actionability/state/price/(acceptable)")
    ap.add_argument("--min_run_length", type=int, default=3, help="RegimeSpec min_run_length")
    ap.add_argument("--max_flip_rate", type=float, default=None, help="RegimeSpec max_flip_rate")
    ap.add_argument("--max_vol", type=float, default=None, help="RegimeSpec max_vol")
    ap.add_argument("--window", type=int, default=50, help="RegimeSpec window")
    ap.add_argument("--bins", type=int, default=20, help="Bins for actionability and margin")
    ap.add_argument("--save", type=str, default=None, help="Optional output image")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    for col in ("actionability", "state", "price"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in log.")

    spec = RegimeSpec(
        min_run_length=args.min_run_length,
        max_flip_rate=args.max_flip_rate,
        max_vol=args.max_vol,
        window=args.window,
    )
    act_centers, marg_centers, dx, dy, acc_frac = compute_vectors(df, spec, bins=args.bins)
    plot_quiver(act_centers, marg_centers, dx, dy, acc_frac, title="Vector field over legitimacy manifold", save=args.save)


if __name__ == "__main__":
    main()

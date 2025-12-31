"""
plot_decision_geometry.py
-------------------------
Visualize controller geometry from trading logs (no behavior changes).

Usage:
  PYTHONPATH=. python scripts/plot_decision_geometry.py --csv logs/trading_log_SPY_1d.csv --save-prefix logs/spy_geom
  PYTHONPATH=. python scripts/plot_decision_geometry.py --csv logs/trading_log_btc_yf.csv --overlay --save-prefix logs/btc_geom
"""

import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _choose_plane_rate_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise SystemExit(f"Missing plane-rate column: {preferred}")
        return preferred
    if "plane_rate" in df.columns:
        return "plane_rate"
    if "delta_plane" in df.columns:
        return "delta_plane"
    plane_rate_cols = [c for c in df.columns if c.startswith("plane_rate")]
    if plane_rate_cols:
        return plane_rate_cols[0]
    raise SystemExit("No plane-rate proxy found (plane_rate, delta_plane, or plane_rate*).")


def _quantile_edges(series: pd.Series, bins: int, q_low: float, q_high: float) -> np.ndarray:
    if bins < 3:
        raise SystemExit("bins must be >= 3 to allow under/overflow bins.")
    interior_bins = bins - 2
    values = series[np.isfinite(series)].to_numpy()
    if values.size == 0:
        raise SystemExit("No finite values for binning.")
    qs = np.quantile(values, np.linspace(q_low, q_high, interior_bins + 1))
    edges = np.concatenate(([-np.inf], qs, [np.inf]))
    edges = np.unique(edges)
    if edges.size < 4:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise SystemExit("No finite min/max values for binning.")
        if vmin == vmax:
            pad = max(abs(vmin) * 0.01, 1e-6)
            vmin -= pad
            vmax += pad
        linear = np.linspace(vmin, vmax, interior_bins + 1)
        edges = np.concatenate(([-np.inf], linear, [np.inf]))
        edges = np.unique(edges)
    return edges


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    edges = edges.astype(float).copy()
    if len(edges) < 3:
        return np.array([])
    if not np.isfinite(edges[0]):
        edges[0] = edges[1] - (edges[2] - edges[1])
    if not np.isfinite(edges[-1]):
        edges[-1] = edges[-2] + (edges[-2] - edges[-3])
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers


def _pivot_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int,
    q_low: float,
    q_high: float,
    min_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_edges = _quantile_edges(df[x_col], bins, q_low, q_high)
    y_edges = _quantile_edges(df[y_col], bins, q_low, q_high)
    x_centers = _bin_centers(x_edges)
    y_centers = _bin_centers(y_edges)
    df = df.copy()
    if x_centers.size + 1 != x_edges.size:
        raise SystemExit("x bin centers do not match edges after de-duplication.")
    if y_centers.size + 1 != y_edges.size:
        raise SystemExit("y bin centers do not match edges after de-duplication.")
    df["x_bin"] = pd.cut(df[x_col], bins=x_edges, labels=x_centers, include_lowest=True)
    df["y_bin"] = pd.cut(df[y_col], bins=y_edges, labels=y_centers, include_lowest=True)

    grouped = df.groupby(["y_bin", "x_bin"], observed=True)
    promote = grouped["shadow_would_promote"].sum()
    reject = grouped["shadow_reject"].sum()
    counts = grouped.size()
    action_rate = grouped["action_active"].mean()
    mean_pnl = grouped["pnl_value"].mean()

    promote_rate = promote / (promote + reject)
    promote_rate = promote_rate.where((promote + reject) > 0)

    promote_table = promote_rate.unstack()
    action_table = action_rate.unstack()
    pnl_table = mean_pnl.unstack()
    count_table = counts.unstack().reindex_like(action_table)

    promote_table = promote_table.where(count_table >= min_count)
    action_table = action_table.where(count_table >= min_count)
    pnl_table = pnl_table.where(count_table >= min_count)

    return promote_table, action_table, pnl_table


def _plot_heat(ax, table: pd.DataFrame, title: str, cmap: str, vmin=None, vmax=None):
    x = table.columns.to_numpy(dtype=float)
    y = table.index.to_numpy(dtype=float)
    Z = table.to_numpy()
    if x.size == 0 or y.size == 0:
        ax.set_title(f"{title} (no data)")
        return
    dx = np.diff(x).mean() if len(x) > 1 else 0.05
    dy = np.diff(y).mean() if len(y) > 1 else 0.05
    extent = [x.min() - dx * 0.5, x.max() + dx * 0.5, y.min() - dy * 0.5, y.max() + dy * 0.5]
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel("plane_abs")
    ax.set_ylabel("stress")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_heatmaps(df: pd.DataFrame, args) -> plt.Figure:
    promote_table, action_table, pnl_table = _pivot_heatmap(
        df,
        "plane_abs",
        args.stress_col,
        args.bins,
        args.quantile_low,
        args.quantile_high,
        args.min_count,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    _plot_heat(axes[0], promote_table, "Promotion rate", "magma", vmin=0.0, vmax=1.0)
    _plot_heat(axes[1], action_table, "Action rate", "viridis", vmin=0.0, vmax=1.0)
    pnl_vals = pnl_table.to_numpy()
    finite = pnl_vals[np.isfinite(pnl_vals)]
    if finite.size:
        max_abs = np.nanpercentile(np.abs(finite), 95)
    else:
        max_abs = 1.0
    _plot_heat(axes[2], pnl_table, "Mean ΔPnL", "coolwarm", vmin=-max_abs, vmax=max_abs)
    return fig


def _compute_flip_count(plane_rate: pd.Series, window: int) -> pd.Series:
    sign = np.sign(plane_rate.fillna(0.0))
    prev = sign.shift(1)
    flip = (sign != prev) & (sign != 0) & (prev != 0)
    return flip.rolling(window=window, min_periods=1).sum()


def plot_overlay(df: pd.DataFrame, plane_rate: str, args) -> plt.Figure:
    start = args.overlay_start
    end = args.overlay_end
    if start is not None or end is not None:
        df = df.iloc[start:end]
    t = df["t"] if "t" in df.columns else np.arange(len(df))
    plane = df[plane_rate].astype(float)
    plane_abs = df["plane_abs"].astype(float)
    stress = df[args.stress_col].astype(float)
    price = df["price"].astype(float)

    promote_mask = df["shadow_would_promote"] == 1
    action_mask = df["action_active"] == 1
    flip_mask = None
    if "plane_would_veto" in df.columns:
        flip_mask = df["plane_would_veto"] == 1
    elif "plane_sign_flips_W" in df.columns:
        flip_mask = df["plane_sign_flips_W"] > 1
    elif args.flip_window and args.flip_window > 1:
        flips = _compute_flip_count(plane, args.flip_window)
        flip_mask = flips > 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, price, color="black", linewidth=1.0)
    axes[0].set_ylabel("price")

    axes[1].plot(t, plane, color="tab:blue", linewidth=1.0, label="plane_rate")
    axes[1].fill_between(t, 0.0, plane_abs, color="tab:orange", alpha=0.3, label="plane_abs")
    axes[1].set_ylabel("plane")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, stress, color="tab:red", linewidth=1.0)
    axes[2].set_ylabel("stress")
    axes[2].set_xlabel("t")

    for ax in axes:
        for x in t[promote_mask]:
            ax.axvline(x, color="purple", alpha=0.15, linewidth=1.0)
        for x in t[action_mask]:
            ax.axvline(x, color="green", alpha=0.12, linewidth=1.0)
        if flip_mask is not None:
            for x in t[flip_mask]:
                ax.axvline(x, color="gray", alpha=0.12, linewidth=1.0, linestyle="--")

    return fig


def plot_simplex(df: pd.DataFrame, args) -> plt.Figure:
    cols = ["p_bad", "plane_abs", args.stress_col]
    for col in cols:
        if col not in df.columns:
            raise SystemExit(f"Missing simplex column: {col}")
    data = df[cols].astype(float).to_numpy()
    data = np.clip(data, 0.0, None)
    sums = data.sum(axis=1, keepdims=True)
    valid = sums[:, 0] > 0
    data = data[valid] / sums[valid]
    if data.shape[0] == 0:
        raise SystemExit("No valid rows for simplex plot.")
    if args.simplex_sample and data.shape[0] > args.simplex_sample:
        rng = np.random.default_rng(seed=7)
        idx = rng.choice(data.shape[0], size=args.simplex_sample, replace=False)
        data = data[idx]
        action = df.loc[valid, "action_t"].to_numpy()[idx]
        promote = df.loc[valid, "shadow_would_promote"].to_numpy()[idx]
    else:
        action = df.loc[valid, "action_t"].to_numpy()
        promote = df.loc[valid, "shadow_would_promote"].to_numpy()

    a = data[:, 0]
    b = data[:, 1]
    c = data[:, 2]
    x = b + 0.5 * c
    y = (np.sqrt(3) / 2.0) * c

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    colors = np.where(action > 0, "tab:blue", np.where(action < 0, "tab:orange", "tab:gray"))
    markers = np.where(promote == 1, "o", "x")
    for m in ("o", "x"):
        mask = markers == m
        ax.scatter(x[mask], y[mask], s=10, c=colors[mask], alpha=0.5, marker=m)

    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2.0, 0]
    ax.plot(tri_x, tri_y, color="black", linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Decision simplex (p_bad, plane_abs, stress)")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Trading log CSV")
    ap.add_argument("--save-prefix", type=str, default=None, help="Prefix for saved plots")
    ap.add_argument("--plane-rate-col", type=str, default=None, help="Signed plane-rate column override")
    ap.add_argument("--stress-col", type=str, default="stress", help="Stress column name")
    ap.add_argument("--pnl-col", type=str, default="realized_pnl_step", help="PnL column for mean map")
    ap.add_argument("--bins", type=int, default=20, help="Total bins including under/overflow")
    ap.add_argument("--quantile-low", type=float, default=0.05, help="Lower quantile for binning")
    ap.add_argument("--quantile-high", type=float, default=0.95, help="Upper quantile for binning")
    ap.add_argument("--min-count", type=int, default=30, help="Minimum bin count to display")
    ap.add_argument("--overlay", action="store_true", help="Render time-series overlay")
    ap.add_argument("--overlay-start", type=int, default=None, help="Overlay slice start index")
    ap.add_argument("--overlay-end", type=int, default=None, help="Overlay slice end index")
    ap.add_argument("--flip-window", type=int, default=None, help="Window for sign-flip veto overlay")
    ap.add_argument("--simplex", action="store_true", help="Render ternary simplex")
    ap.add_argument("--simplex-sample", type=int, default=5000, help="Max points for simplex plot")
    ap.add_argument("--no-show", action="store_true", help="Skip interactive display")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    plane_rate_col = _choose_plane_rate_col(df, args.plane_rate_col)
    df["plane_rate_used"] = df[plane_rate_col].astype(float)
    df["plane_abs"] = df["plane_rate_used"].abs()
    df["action_active"] = (df["action_t"].astype(float) != 0).astype(int)
    if args.pnl_col not in df.columns:
        raise SystemExit(f"Missing pnl column: {args.pnl_col}")
    df["pnl_value"] = df[args.pnl_col].astype(float)

    fig = plot_heatmaps(df, args)
    if args.save_prefix:
        out = f"{args.save_prefix}_heatmaps.png"
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    if not args.no_show:
        plt.show()

    if args.overlay:
        fig = plot_overlay(df, "plane_rate_used", args)
        if args.save_prefix:
            out = f"{args.save_prefix}_overlay.png"
            fig.savefig(out, dpi=200)
            print(f"Saved {out}")
        if not args.no_show:
            plt.show()

    if args.simplex:
        fig = plot_simplex(df, args)
        if args.save_prefix:
            out = f"{args.save_prefix}_simplex.png"
            fig.savefig(out, dpi=200)
            print(f"Saved {out}")
        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()

"""
plot_fn_anatomy.py
------------------
Missed-opportunity anatomy (false negatives decomposition).
For FN bars (HOLD & acceptable), attribute reasons:
  - low_actionability: actionability < tau_on
  - near_boundary: legitimacy margin <= margin_eps
  - cooldown: within cooldown_bars since last ACT
  - weak_persistence: acceptable run-length < persistence_k

Plots a stacked bar over actionability bins showing counts per reason.
Requires log with columns: acceptable, action, actionability, t, state, price.

Usage:
  PYTHONPATH=. python trading/scripts/plot_fn_anatomy.py --log logs/trading_log.csv --tau_on 0.5 --tau_off 0.3 --save fn_anatomy.png
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


def fn_anatomy(df: pd.DataFrame, spec: RegimeSpec, tau_on: float, cooldown_bars: int, margin_eps: float, persistence_k: int, bins: int):
    acc = df["acceptable"].astype(bool).to_numpy()
    act = (df["action"] != 0).to_numpy()
    actionability = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)
    states = pd.to_numeric(df["state"], errors="coerce").fillna(0).to_numpy(dtype=int)
    prices = pd.to_numeric(df["price"], errors="coerce").fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)

    margin = legitimacy_margin(states, prices, spec)
    acc_runs = sign_run_lengths(acc.astype(int))

    # distance since last ACT
    last_act = -1e9
    dist_since_act = np.zeros(len(act), dtype=int)
    for i, a in enumerate(act):
        if a:
            last_act = i
        dist_since_act[i] = i - last_act if last_act >= 0 else 1e9

    fn_mask = (~act) & acc & np.isfinite(actionability)
    fn_indices = np.where(fn_mask)[0]
    if fn_indices.size == 0:
        return None, None, None

    reasons = {
        "low_actionability": np.zeros(len(fn_indices), dtype=bool),
        "near_boundary": np.zeros(len(fn_indices), dtype=bool),
        "cooldown": np.zeros(len(fn_indices), dtype=bool),
        "weak_persistence": np.zeros(len(fn_indices), dtype=bool),
    }

    for j, idx in enumerate(fn_indices):
        reasons["low_actionability"][j] = actionability[idx] < tau_on
        reasons["near_boundary"][j] = margin[idx] <= margin_eps
        reasons["cooldown"][j] = dist_since_act[idx] <= cooldown_bars
        reasons["weak_persistence"][j] = acc_runs[idx] < persistence_k

    edges = np.linspace(0.0, 1.0, max(5, bins) + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    cats = pd.cut(actionability[fn_indices], edges, include_lowest=True, labels=labels)

    counts = pd.DataFrame({"bin": cats})
    for name, flags in reasons.items():
        counts[name] = flags

    grouped = counts.groupby("bin").agg({k: "sum" for k in reasons.keys()})
    bin_centers = grouped.index.astype(float).to_numpy()
    # ensure all reasons present
    return bin_centers, grouped.to_numpy().T, list(reasons.keys())


def plot_stacked(bin_centers, reason_counts, reason_names, title, save=None):
    plt.figure(figsize=(8, 5))
    bottoms = np.zeros_like(bin_centers, dtype=float)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, name in enumerate(reason_names):
        vals = reason_counts[i]
        plt.bar(bin_centers, vals, bottom=bottoms, width=0.8 * (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 0.04, color=colors[i % len(colors)], label=name)
        bottoms = bottoms + vals
    plt.xlabel("actionability (bin center)")
    plt.ylabel("FN count")
    plt.title(title)
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
    ap.add_argument("--log", type=str, required=True, help="Trading log CSV with acceptable/action/actionability/state/price")
    ap.add_argument("--acceptable-col", type=str, default="acceptable")
    ap.add_argument("--action-col", type=str, default="action")
    ap.add_argument("--actionability-col", type=str, default="actionability")
    ap.add_argument("--state-col", type=str, default="state")
    ap.add_argument("--price-col", type=str, default="price")
    ap.add_argument("--tau_on", type=float, default=0.5, help="Entry threshold (for low_actionability reason)")
    ap.add_argument("--tau_off", type=float, default=0.3, help="Exit threshold (unused directly, for reference)")
    ap.add_argument("--cooldown", type=int, default=5, help="Cooldown bars after ACT counts as cooldown reason")
    ap.add_argument("--margin_eps", type=float, default=0.5, help="Margin threshold for near_boundary")
    ap.add_argument("--persistence_k", type=int, default=3, help="Acceptable run-length threshold for weak_persistence")
    ap.add_argument("--min_run_length", type=int, default=3, help="RegimeSpec min_run_length (for margin)")
    ap.add_argument("--max_flip_rate", type=float, default=None, help="RegimeSpec max_flip_rate (for margin)")
    ap.add_argument("--max_vol", type=float, default=None, help="RegimeSpec max_vol (for margin)")
    ap.add_argument("--window", type=int, default=50, help="RegimeSpec window (for margin)")
    ap.add_argument("--bins", type=int, default=20, help="Actionability bins")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    col_map = {
        "acceptable": args.acceptable_col,
        "action": args.action_col,
        "actionability": args.actionability_col,
        "state": args.state_col,
        "price": args.price_col,
    }
    for name, col in col_map.items():
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' for {name}.")

    spec = RegimeSpec(
        min_run_length=args.min_run_length,
        max_flip_rate=args.max_flip_rate,
        max_vol=args.max_vol,
        window=args.window,
    )
    res = fn_anatomy(
        df.rename(columns={v: k for k, v in col_map.items()}),
        spec=spec,
        tau_on=args.tau_on,
        cooldown_bars=args.cooldown,
        margin_eps=args.margin_eps,
        persistence_k=args.persistence_k,
        bins=args.bins,
    )
    if res[0] is None:
        print("No false negatives found; nothing to plot.")
        return
    bin_centers, reason_counts, reason_names = res
    plot_stacked(bin_centers, reason_counts, reason_names, title="FN decomposition over actionability", save=args.save)


if __name__ == "__main__":
    main()

"""
training_dashboard.py
---------------------
Lightweight live/refreshable visualization for triadic trading logs.

Expected CSV columns (append rows during run):
    t, price, pnl, z_norm, z_vel, hold, entropy, regime, action
Optional PR curve CSV (from hysteresis sweep):
    tau_on, tau_off, acceptable_pct, precision, recall, trades, hold_pct
Optional engagement curve (from log itself):
    actionability, acceptable, action

Run: `python training_dashboard.py --log logs/trading_log.csv --refresh 1.0`
Sweep PR (optional sparkline): `python scripts/sweep_tau_conf.py --out logs/pr_curve.csv`
If the log file does not exist, a synthetic demo is shown.

Views:
 1) Equity curve vs time
 2) Latent velocity + HOLD% (rolling)
 3) Price with action markers (buy/sell/hold)
 4) (optional) Precision/recall vs tau_off sparkline
 5) (optional) Engagement vs actionability (P(ACT | actionability, acceptable))
"""

import argparse
import time
import pathlib
import pandas as pd
import numpy as np
import matplotlib
# Prefer an interactive backend; fall back if unavailable.
if matplotlib.get_backend().lower().startswith("agg"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt


def load_log(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def synthetic_log(n=200):
    t = np.arange(n)
    price = 100 + np.cumsum(np.random.normal(0, 0.2, size=n))
    pnl = np.cumsum(np.random.normal(0, 0.01, size=n))
    z_norm = np.abs(np.random.normal(1.0, 0.2, size=n))
    z_vel = np.abs(np.random.normal(0.05, 0.02, size=n))
    hold = (np.random.rand(n) > 0.6).astype(int)
    entropy = np.random.uniform(0, 1, size=n)
    regime = np.random.randint(0, 3, size=n)
    action = np.random.choice([-1, 0, 1], size=n)
    return pd.DataFrame(
        {
            "t": t,
            "price": price,
            "pnl": pnl,
            "z_norm": z_norm,
            "z_vel": z_vel,
            "hold": hold,
            "entropy": entropy,
            "regime": regime,
            "action": action,
        }
    )


def rolling_hold(log, window=50):
    if "hold" not in log:
        return None
    return log["hold"].rolling(window, min_periods=1).mean()


def prepare_pr_curve(pr_curve: pd.DataFrame):
    if pr_curve is None or pr_curve.empty:
        return None
    required = {"tau_off", "precision", "recall"}
    if not required.issubset(pr_curve.columns):
        return None
    return pr_curve.sort_values("tau_off")


def engagement_vs_actionability(log: pd.DataFrame, bins: int = 20):
    """
    Compute P(ACT | actionability, acceptable) and P(ACT | actionability, unacceptable) using binned actionability.
    Returns (bin_centers, act_rate_acc, act_rate_unacc) or (None, None, None) if data is missing.
    """
    if log is None or log.empty or "actionability" not in log or "action" not in log:
        return None, None, None
    actionability = pd.to_numeric(log["actionability"], errors="coerce")
    act = (log["action"] != 0).astype(int)
    if "acceptable" not in log:
        return None, None, None
    acc_mask = log["acceptable"].astype(bool)
    unacc_mask = ~acc_mask

    def bin_rates(mask):
        a = actionability[mask]
        y = act[mask]
        a = a[np.isfinite(a)]
        y = y.loc[a.index]
        if a.empty:
            return None
        edges = np.linspace(0.0, 1.0, max(5, bins) + 1)
        labels = 0.5 * (edges[:-1] + edges[1:])
        cat = pd.cut(a, edges, include_lowest=True, labels=labels)
        grouped = y.groupby(cat).mean()
        return grouped.index.astype(float).to_numpy(), grouped.to_numpy()

    res_acc = bin_rates(acc_mask)
    res_unacc = bin_rates(unacc_mask)
    if res_acc is None and res_unacc is None:
        return None, None, None
    # align centers: prefer acceptable centers; else unacceptable
    centers = res_acc[0] if res_acc is not None else res_unacc[0]
    rates_acc = res_acc[1] if res_acc is not None else None
    rates_unacc = res_unacc[1] if res_unacc is not None else None
    return centers, rates_acc, rates_unacc


def draw(log: pd.DataFrame, pr_curve: pd.DataFrame = None):
    if log is None or log.empty:
        return

    pr_curve = prepare_pr_curve(pr_curve)
    centers, rates_acc, rates_unacc = engagement_vs_actionability(log)
    has_engagement = centers is not None and (rates_acc is not None or rates_unacc is not None)
    rows = 3
    if pr_curve is not None:
        rows += 1
    if has_engagement:
        rows += 1
    t = log["t"]
    hold_roll = rolling_hold(log)

    plt.clf()
    plt.subplot(rows, 1, 1)
    plt.plot(t, log["pnl"], label="PnL")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(rows, 1, 2)
    if "z_vel" in log:
        plt.plot(t, log["z_vel"], label="latent vel")
    if hold_roll is not None:
        plt.plot(t, hold_roll, label="HOLD% (roll)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(rows, 1, 3)
    plt.plot(t, log["price"], label="price", color="black", alpha=0.6)
    if "action" in log:
        buys = log["action"] == 1
        sells = log["action"] == -1
        holds = log["action"] == 0
        plt.scatter(t[buys], log["price"][buys], c="green", s=10, label="buy")
        plt.scatter(t[sells], log["price"][sells], c="red", s=10, label="sell")
        plt.scatter(t[holds], log["price"][holds], c="gray", s=5, label="hold", alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if pr_curve is not None:
        plt.subplot(rows, 1, 4)
        tau = pr_curve["tau_off"]
        plt.plot(tau, pr_curve["precision"], marker="o", label="precision")
        plt.plot(tau, pr_curve["recall"], marker="o", label="recall")
        plt.ylim(0, 1.05)
        plt.xlabel("tau_off (hysteresis exit)")
        plt.ylabel("probability")
        plt.legend()
        plt.grid(True, alpha=0.3)

    if has_engagement:
        plt.subplot(rows, 1, rows)
        if rates_acc is not None:
            plt.plot(centers, rates_acc, marker="o", label="P(ACT | actionability, acceptable)")
        if rates_unacc is not None:
            plt.plot(centers, rates_unacc, marker="x", linestyle="--", label="P(ACT | actionability, unacceptable)")
        plt.ylim(0, 1.05)
        plt.xlabel("actionability (binned)")
        plt.ylabel("engagement rate")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.pause(0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="logs/trading_log.csv", help="CSV log file path")
    ap.add_argument("--refresh", type=float, default=1.0, help="Refresh seconds; set 0 for one-shot")
    ap.add_argument(
        "--pr",
        type=str,
        default=None,
        help="Optional CSV with tau_off precision/recall sweep (from scripts/sweep_tau_conf.py --out ...).",
    )
    args = ap.parse_args()

    log_path = pathlib.Path(args.log)
    pr_path = pathlib.Path(args.pr) if args.pr else None
    plt.ion()

    if args.refresh <= 0:
        log = load_log(log_path)
        if log is None:
            log = synthetic_log()
        pr_curve = load_log(pr_path) if pr_path else None
        draw(log, pr_curve)
        plt.show(block=True)
        return

    while True:
        log = load_log(log_path)
        if log is None:
            log = synthetic_log()
        pr_curve = load_log(pr_path) if pr_path else None
        draw(log, pr_curve)
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()

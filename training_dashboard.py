"""
training_dashboard.py
---------------------
Lightweight live/refreshable visualization for triadic trading logs.

Expected CSV columns (append rows during run):
    t, price, pnl, z_norm, z_vel, hold, entropy, regime, action

Run: `python training_dashboard.py --log logs/trading_log.csv --refresh 1.0`
If the log file does not exist, a synthetic demo is shown.

Views:
 1) Equity curve vs time
 2) Latent velocity + HOLD% (rolling)
 3) Price with action markers (buy/sell/hold)
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


def draw(log: pd.DataFrame):
    if log is None or log.empty:
        return

    t = log["t"]
    hold_roll = rolling_hold(log)

    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(t, log["pnl"], label="PnL")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    if "z_vel" in log:
        plt.plot(t, log["z_vel"], label="latent vel")
    if hold_roll is not None:
        plt.plot(t, hold_roll, label="HOLD% (roll)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
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

    plt.tight_layout()
    plt.pause(0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="logs/trading_log.csv", help="CSV log file path")
    ap.add_argument("--refresh", type=float, default=1.0, help="Refresh seconds; set 0 for one-shot")
    args = ap.parse_args()

    log_path = pathlib.Path(args.log)
    plt.ion()

    if args.refresh <= 0:
        log = load_log(log_path)
        if log is None:
            log = synthetic_log()
        draw(log)
        plt.show(block=True)
        return

    while True:
        log = load_log(log_path)
        if log is None:
            log = synthetic_log()
        draw(log)
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()

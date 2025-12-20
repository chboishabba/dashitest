"""
plot_microstructure_overlay.py
------------------------------
Acceptable × market microstructure overlays (PnL-free).
Time series with acceptable stripes over:
  - realized volatility (rolling std of returns)
  - absolute returns (bar range proxy)
  - volume (if present)

Usage:
  PYTHONPATH=. python scripts/plot_microstructure_overlay.py --log logs/trading_log.csv --save microstructure.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log with price,t,acceptable")
    ap.add_argument("--vol_window", type=int, default=50, help="Rolling window for realized vol")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    for col in ("price", "t", "acceptable"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in log.")

    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    price = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=float)
    acceptable = df["acceptable"].astype(bool).to_numpy()

    ret = np.zeros_like(price, dtype=float)
    ret[1:] = price[1:] / price[:-1] - 1.0
    vol = pd.Series(ret).rolling(args.vol_window).std().to_numpy()
    abs_ret = np.abs(ret)
    has_volume = "volume" in df.columns
    if has_volume:
        volume = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=float)

    rows = 3 if has_volume else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    if rows == 2:
        ax_vol, ax_abs = axes
    else:
        ax_vol, ax_abs, ax_volume = axes

    # realized vol
    ax_vol.plot(t, vol, label="realized vol (rolling)", color="tab:blue")
    ax_vol.fill_between(t, vol.min(), vol.max(), where=acceptable, color="gray", alpha=0.1, label="acceptable")
    ax_vol.set_ylabel("vol")
    ax_vol.legend()
    ax_vol.grid(True, alpha=0.3)

    # abs returns
    ax_abs.plot(t, abs_ret, label="abs return (bar range proxy)", color="tab:orange")
    ax_abs.fill_between(t, abs_ret.min(), abs_ret.max(), where=acceptable, color="gray", alpha=0.1)
    ax_abs.set_ylabel("|ret|")
    ax_abs.legend()
    ax_abs.grid(True, alpha=0.3)

    if has_volume:
        ax_volume.plot(t, volume, label="volume", color="tab:green")
        ax_volume.fill_between(t, volume.min(), volume.max(), where=acceptable, color="gray", alpha=0.1)
        ax_volume.set_ylabel("volume")
        ax_volume.legend()
        ax_volume.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (t)")
    plt.suptitle("Acceptable overlays on microstructure features")
    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()

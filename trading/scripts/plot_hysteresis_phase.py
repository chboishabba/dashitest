"""
plot_hysteresis_phase.py
------------------------
Hysteresis loop trace (phase portrait) for actionability.
x: actionability(t)
y: actionability(t+1)
color: control state (HOLD vs ACT)
Overlay tau_on and tau_off as lines.

Usage:
  PYTHONPATH=. python trading/scripts/plot_hysteresis_phase.py --log logs/trading_log.csv --tau_on 0.5 --tau_off 0.3 --save hysteresis_phase.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log with actionability/action")
    ap.add_argument("--tau_on", type=float, default=0.5, help="Hysteresis entry threshold")
    ap.add_argument("--tau_off", type=float, default=0.3, help="Hysteresis exit threshold")
    ap.add_argument("--max_points", type=int, default=5000, help="Sample up to this many points for plotting")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.log)
    except Exception as e:
        raise SystemExit(f"Failed to read log: {e}")
    if "actionability" not in df or "action" not in df:
        raise SystemExit("Log must contain actionability and action columns.")

    act = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)
    state = (df["action"] != 0).to_numpy(dtype=bool)  # ACT vs HOLD
    # Build (t, t+1)
    act_t = act[:-1]
    act_t1 = act[1:]
    state_t = state[:-1]
    mask = np.isfinite(act_t) & np.isfinite(act_t1)
    act_t = act_t[mask]
    act_t1 = act_t1[mask]
    state_t = state_t[mask]

    # Downsample if needed
    if len(act_t) > args.max_points:
        idx = np.random.choice(len(act_t), size=args.max_points, replace=False)
        act_t = act_t[idx]
        act_t1 = act_t1[idx]
        state_t = state_t[idx]

    colors = np.where(state_t, "tab:orange", "tab:blue")

    plt.figure(figsize=(6, 6))
    plt.scatter(act_t, act_t1, c=colors, s=5, alpha=0.5, label=None)
    # Overlay tau_on/off
    plt.axvline(args.tau_on, color="green", linestyle="--", label="tau_on")
    plt.axvline(args.tau_off, color="red", linestyle="--", label="tau_off")
    plt.axhline(args.tau_on, color="green", linestyle=":")
    plt.axhline(args.tau_off, color="red", linestyle=":")
    plt.xlabel("actionability(t)")
    plt.ylabel("actionability(t+1)")
    plt.title("Hysteresis phase portrait (ACT=orange, HOLD=blue)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()

"""
plot_quotient_features.py
-------------------------
Plot quotient invariant features (q_*) over time from a trading log.

Usage:
  PYTHONPATH=. python trading/scripts/plot_quotient_features.py --log logs/trading_log.csv --save quotient.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import timestamped_path


FEATURES = [
    ("q_e64", "E64 (energy)"),
    ("q_c64", "C64 (cancellation)"),
    ("q_s64", "S64 (spectral)"),
    ("q_de", "Delta E"),
    ("q_dc", "Delta C"),
    ("q_ds", "Delta S"),
]


def load_log(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Failed to read log {path}: {e}")


def plot_features(df: pd.DataFrame, window: int, save: str | None) -> None:
    missing = [name for name, _ in FEATURES if name not in df.columns]
    if missing:
        raise SystemExit(f"Log is missing columns: {', '.join(missing)}")

    t = df["t"] if "t" in df.columns else df.index
    rolling = df[[name for name, _ in FEATURES]].rolling(window, min_periods=1).mean()

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    for ax, (name, label) in zip(axes, FEATURES, strict=False):
        ax.plot(t, df[name], alpha=0.35, label="raw")
        ax.plot(t, rolling[name], linewidth=1.5, label=f"ma{window}")
        ax.set_title(label)
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("t")
    fig.suptitle("Quotient features over time", y=0.98)
    plt.tight_layout()
    if save:
        save_path = timestamped_path(save)
        plt.savefig(save_path, dpi=200)
        print(f"Saved {save_path}")
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log CSV with q_* columns")
    ap.add_argument("--window", type=int, default=64, help="Rolling mean window")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    df = load_log(args.log)
    plot_features(df, window=args.window, save=args.save)


if __name__ == "__main__":
    main()

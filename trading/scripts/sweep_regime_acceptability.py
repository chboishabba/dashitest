"""
sweep_regime_acceptability.py
-----------------------------
Sweep RegimeSpec parameters to map the acceptable manifold.
Defaults: vary min_run_length and (optionally) a volatility cap percentile.

Outputs a CSV with acceptable% for each combo; can be heatmapped separately.

Usage:
  PYTHONPATH=. python trading/scripts/sweep_regime_acceptability.py --csv data/raw/stooq/btc_intraday.csv --out logs/accept_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from trading.regime import RegimeSpec, check_regime
from trading.run_trader import load_prices, compute_triadic_state


def compute_vol(price: np.ndarray, window: int = 50):
    rets = np.diff(price, prepend=price[0])
    return pd.Series(rets).rolling(window).std().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Price CSV (Stooq BTC intraday expected)")
    ap.add_argument(
        "--min_run",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="Values of min_run_length to sweep",
    )
    ap.add_argument(
        "--vol_pct",
        type=float,
        nargs="+",
        default=[None, 70.0, 80.0, 90.0],
        help="Volatility cap percentiles to sweep (None disables cap).",
    )
    ap.add_argument(
        "--flip_cap",
        type=float,
        nargs="+",
        default=[None, 0.05, 0.1],
        help="Flip-rate caps to sweep (None disables cap).",
    )
    ap.add_argument("--window", type=int, default=50, help="Rolling window for volatility")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = ap.parse_args()

    price, _ = load_prices(args.csv)
    state = compute_triadic_state(price)
    vol = compute_vol(price, window=args.window)

    rows = []
    for m in args.min_run:
        for pct in args.vol_pct:
            max_vol = None
            max_vol_pct = None
            if pct is not None:
                max_vol = np.nanpercentile(vol, pct)
                max_vol_pct = pct
            for flip_cap in args.flip_cap:
                spec = RegimeSpec(min_run_length=m, max_vol=max_vol, max_flip_rate=flip_cap, window=args.window)
                acceptable = check_regime(state, vol, spec)
                acceptable_pct = float(np.mean(acceptable))
                rows.append(
                    {
                        "min_run_length": m,
                        "max_vol_pct": max_vol_pct,
                        "max_flip_rate": flip_cap,
                        "acceptable_pct": acceptable_pct,
                    }
                )
                print(
                    f"min_run={m} vol_cap_pct={max_vol_pct} flip_cap={flip_cap} acceptable={acceptable_pct:.3f}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote acceptability surface to {args.out}")


if __name__ == "__main__":
    main()

"""
build_synthetic_options_surface.py
----------------------------------
Generate a synthetic options surface Parquet (tenor × moneyness × call/put)
using the same deterministic rules as the option chain.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from trading.trading_io.prices import load_prices
    from trading.vk_qfeat import QFeatTape
    from options.chain import SyntheticOptionChain, SyntheticOptionsConfig
except ModuleNotFoundError:
    from trading_io.prices import load_prices
    from vk_qfeat import QFeatTape
    from options.chain import SyntheticOptionChain, SyntheticOptionsConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a synthetic options surface Parquet.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--prices-csv", type=Path, required=True, help="Prices CSV for bars.")
    ap.add_argument("--out", type=Path, required=True, help="Output Parquet path.")
    ap.add_argument("--sigma0", type=float, default=0.60, help="Synthetic IV base.")
    ap.add_argument("--sigma-min", type=float, default=0.20, help="Synthetic IV min.")
    ap.add_argument("--sigma-max", type=float, default=1.50, help="Synthetic IV max.")
    ap.add_argument("--k-v", type=float, default=0.50, help="Vol ratio sensitivity.")
    ap.add_argument("--k-b", type=float, default=0.10, help="Burst sensitivity.")
    args = ap.parse_args()

    price, _vol, ts = load_prices(args.prices_csv, return_time=True)
    ts_int = pd.to_datetime(ts).astype("int64") if ts is not None else np.arange(price.size, dtype=np.int64)
    tape = QFeatTape.from_existing(str(args.tape), rows=price.size)
    qfeat = tape.mm[0, :, :6].astype(np.float32, copy=False)

    config = SyntheticOptionsConfig(
        sigma0=float(args.sigma0),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        k_v=float(args.k_v),
        k_b=float(args.k_b),
    )
    chain = SyntheticOptionChain(config)

    rows = []
    for t in range(price.size):
        spot = float(price[t])
        candidates = chain.generate(t, spot, qfeat[t], int(ts_int[t]) if ts is not None else int(t))
        bar_ts_ns = int(ts_int[t]) if ts is not None else t
        bar_ts_ms = bar_ts_ns // 1_000_000
        for cand in candidates:
            mny = math.log(max(cand.strike, 1e-6) / max(cand.spot, 1e-6))
            rows.append(
                {
                    "ts": int(bar_ts_ms),
                    "opt_type": cand.opt_type,
                    "expiry_days": cand.expiry_days,
                    "tenor_bin": cand.tenor_bin,
                    "mny_bin": cand.mny_bin,
                    "strike": float(cand.strike),
                    "mid": float(cand.mid) if cand.mid is not None else math.nan,
                    "iv": float(cand.iv) if cand.iv is not None else math.nan,
                    "spot": float(cand.spot),
                    "mny": float(mny),
                    "source": cand.source,
                }
            )

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()

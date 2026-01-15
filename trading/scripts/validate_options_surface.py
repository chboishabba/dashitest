"""
validate_options_surface.py
----------------------------
Inspect a Deribit/Binance-style options surface Parquet for coverage, bins, and candidate readiness.

Usage:
  python scripts/validate_options_surface.py --surface data/options/deribit_surface.parquet
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _parse_list(raw: str, typ):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [typ(p) for p in parts]


def _check_ts_units(values: Iterable[int]) -> None:
    min_ts = min(values)
    max_ts = max(values)
    if max_ts >= 1e15:
        raise SystemExit("Surface ts values look like nanoseconds; expected milliseconds.")
    if max_ts < 1e12:
        raise SystemExit("Surface ts values look like seconds; expected milliseconds.")


def _assign_tenor_bin(expiry_days: float, bins: Sequence[str], targets: Sequence[int]) -> str:
    if not targets:
        return bins[0]
    diffs = [abs(expiry_days - target) for target in targets]
    idx = int(np.argmin(diffs))
    return bins[idx]


def _assign_mny_bin(mny: float, bins: Sequence[str], cutoffs: Sequence[float]) -> str:
    for idx, cutoff in enumerate(cutoffs):
        if mny < cutoff:
            return bins[idx]
    return bins[-1]


def _simulate_grid(
    snap: pd.DataFrame, tenor_bins: Sequence[str], mny_bins: Sequence[str]
) -> tuple[int, int]:
    wanted = {(tenor, mny, opt) for tenor in tenor_bins for mny in mny_bins for opt in ("call", "put")}
    available = set(
        zip(
            snap["tenor_bin"].astype(str),
            snap["mny_bin"].astype(str),
            snap["opt_type"].astype(str),
        )
    )
    found = wanted & available
    return len(found), len(wanted)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate an options surface for Phase-4 readiness.")
    ap.add_argument("--surface", type=Path, required=True, help="Parquet file containing options surface.")
    ap.add_argument("--tenor-bins", type=str, default="e_1_3,e_4_7,e_8_21,e_22_60,e_61_180", help="Comma-separated tenor bin labels.")
    ap.add_argument("--tenor-days", type=str, default="2,5,14,41,120", help="Comma-separated tenor target days.")
    ap.add_argument("--mny-bins", type=str, default="m_deep_itm,m_itm,m_atm,m_otm,m_deep_otm", help="Moneyness bin labels.")
    ap.add_argument("--mny-cutoffs", type=str, default="-0.15,-0.05,0.05,0.15", help="Cutoffs for moneyness bins (len = bins-1).")
    ap.add_argument("--sample-snapshots", type=int, default=3, help="How many snapshots to print details for.")
    args = ap.parse_args()

    df = pd.read_parquet(args.surface)
    if df.empty:
        raise SystemExit("Options surface file is empty.")

    required = {"ts", "opt_type", "strike", "spot", "expiry_days"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Surface missing required columns: {missing}")

    tenor_bins = [t.strip() for t in args.tenor_bins.split(",") if t.strip()]
    tenor_days = _parse_list(args.tenor_days, int)
    if len(tenor_bins) != len(tenor_days):
        raise SystemExit("tenor_bins and tenor_days must have equal length.")

    mny_bins = [t.strip() for t in args.mny_bins.split(",") if t.strip()]
    mny_cutoffs = _parse_list(args.mny_cutoffs, float)
    if len(mny_cutoffs) != len(mny_bins) - 1:
        raise SystemExit("mny_cutoffs length must equal len(mny_bins)-1.")

    df = df.copy()
    df["ts"] = df["ts"].astype(np.int64)
    _check_ts_units(df["ts"].astype(int).values)
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    df["mny"] = np.log(df["strike"] / df["spot"])

    df["tenor_bin"] = df["expiry_days"].apply(lambda x: _assign_tenor_bin(float(x), tenor_bins, tenor_days))
    df["mny_bin"] = df["mny"].apply(lambda x: _assign_mny_bin(float(x), mny_bins, mny_cutoffs))

    print(f"Surface rows: {len(df)}")
    print(f"Snapshots: {df['ts'].nunique()} | Dates: {df['date'].nunique()}")

    coverage: list[tuple[str, float, float, float, float]] = []
    snapshot_stats = []
    for date, daily in df.groupby("date"):
        snaps = []
        for ts, snap in daily.groupby("ts"):
            found, total = _simulate_grid(snap, tenor_bins, mny_bins)
            snaps.append(found / total if total > 0 else 0.0)
        snaps = np.array(snaps, dtype=float)
        if snaps.size == 0:
            continue
        coverage.append(
            (
                str(date),
                float(snaps.mean()),
                float(snaps.min()),
                float(snaps.max()),
                float(snaps.size),
            )
        )
        snapshot_stats.append((date, daily["ts"].nunique(), int(snaps.mean() * 100), int(snaps.min() * 100)))

    print("\nCoverage per date (avg/min/max % of grid found, snapshots):")
    for date, avg, mini, maxi, count in coverage:
        print(f"  {date}: {avg*100:.1f}% / {mini*100:.1f}% / {maxi*100:.1f}% over {int(count)} snapshots")

    print("\nSample snapshots (first few dates):")
    sample_dates = sorted(set(date for date, *_ in coverage))[: args.sample_snapshots]
    for date in sample_dates:
        daily = df[df["date"] == pd.to_datetime(date).date()]
        ts_vals = sorted(daily["ts"].unique())[: args.sample_snapshots]
        for ts in ts_vals:
            snap = daily[daily["ts"] == ts]
            found, total = _simulate_grid(snap, tenor_bins, mny_bins)
            print(
                f"  [{date} {pd.to_datetime(ts, unit='ms')}] "
                f"{found}/{total} combos "
                f"(unique tenor={snap['tenor_bin'].nunique()}, "
                f"mny={snap['mny_bin'].nunique()})"
            )

    print("\nPer-tenor coverage summary:")
    for tenor in tenor_bins:
        mask = df["tenor_bin"] == tenor
        print(
            f"  {tenor}: {mask.sum()} rows, "
            f"unique mny bins={df.loc[mask, 'mny_bin'].nunique()}, "
            f"calls={np.sum((mask) & (df['opt_type'] == 'call'))}, "
            f"puts={np.sum((mask) & (df['opt_type'] == 'put'))}"
        )

    print("\nPer-moneyness coverage summary:")
    for mny in mny_bins:
        mask = df["mny_bin"] == mny
        print(
            f"  {mny}: {mask.sum()} rows, "
            f"unique tenor bins={df.loc[mask, 'tenor_bin'].nunique()}"
        )


if __name__ == "__main__":
    main()

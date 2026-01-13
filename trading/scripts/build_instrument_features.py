"""
build_instrument_features.py
----------------------------
Normalize market metadata (funding/basis/open interest/options) into a
time-aligned feature table for an instrument head.

This does NOT trade or train; it only prepares features.

Usage:
  PYTHONPATH=. python trading/scripts/build_instrument_features.py \
    --premium-json data/market_meta/binance_premium_BTCUSDT.json \
    --oi-json data/market_meta/binance_open_interest_BTCUSDT_5m.json \
    --options-instruments data/market_meta/deribit_options_instruments_BTC.json \
    --options-summary data/market_meta/deribit_options_summary_BTC.json \
    --prices-csv data/raw/stooq/btc.us.csv \
    --out logs/market_meta_features_btc.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from trading.trading_io.prices import load_prices
except ModuleNotFoundError:
    from trading_io.prices import load_prices


def _ts_ms_to_ns(ts_ms: int | float) -> int:
    return int(ts_ms) * 1_000_000


def _load_premium(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    ts = _ts_ms_to_ns(data["time"])
    row = {
        "ts": ts,
        "premium_mark_price": float(data.get("markPrice", "nan")),
        "premium_index_price": float(data.get("indexPrice", "nan")),
        "premium_estimated_settle": float(data.get("estimatedSettlePrice", "nan")),
        "premium_funding_rate": float(data.get("lastFundingRate", "nan")),
        "premium_interest_rate": float(data.get("interestRate", "nan")),
        "premium_next_funding_ts": _ts_ms_to_ns(data.get("nextFundingTime", 0)),
    }
    return pd.DataFrame([row])


def _load_open_interest(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    for item in data:
        ts = _ts_ms_to_ns(item["timestamp"])
        rows.append(
            {
                "ts": ts,
                "oi_sum_open_interest": float(item.get("sumOpenInterest", "nan")),
                "oi_sum_open_interest_value": float(item.get("sumOpenInterestValue", "nan")),
                "oi_cmc_supply": float(item.get("CMCCirculatingSupply", "nan")),
            }
        )
    return pd.DataFrame(rows)


def _load_deribit_instruments(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    inst = payload.get("result", [])
    if not inst:
        return pd.DataFrame()
    ref_ts = max(int(i.get("creation_timestamp", 0)) for i in inst)
    rows = []
    for item in inst:
        rows.append(
            {
                "option_type": item.get("option_type"),
                "strike": float(item.get("strike", "nan")),
                "expiration_timestamp": int(item.get("expiration_timestamp", 0)),
                "is_active": bool(item.get("is_active", True)),
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["is_active"]]
    if df.empty:
        return pd.DataFrame()
    exp_days = (df["expiration_timestamp"] - ref_ts) / 86_400_000
    row = {
        "ts": _ts_ms_to_ns(ref_ts),
        "opt_count": int(df.shape[0]),
        "opt_call_count": int((df["option_type"] == "call").sum()),
        "opt_put_count": int((df["option_type"] == "put").sum()),
        "opt_strike_mean": float(df["strike"].mean()),
        "opt_expiry_days_mean": float(exp_days.mean()),
        "opt_expiry_days_p50": float(exp_days.median()),
    }
    return pd.DataFrame([row])


def _load_deribit_summary(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    items = payload.get("result", [])
    if not items:
        return pd.DataFrame()
    ref_ts = max(int(i.get("creation_timestamp", 0)) for i in items)
    df = pd.DataFrame(items)
    row = {
        "ts": _ts_ms_to_ns(ref_ts),
        "opt_mark_iv_mean": float(pd.to_numeric(df.get("mark_iv"), errors="coerce").mean()),
        "opt_mark_iv_p50": float(pd.to_numeric(df.get("mark_iv"), errors="coerce").median()),
        "opt_open_interest_sum": float(pd.to_numeric(df.get("open_interest"), errors="coerce").sum()),
        "opt_volume_sum": float(pd.to_numeric(df.get("volume"), errors="coerce").sum()),
        "opt_underlying_price_mean": float(pd.to_numeric(df.get("underlying_price"), errors="coerce").mean()),
        "opt_mid_price_mean": float(pd.to_numeric(df.get("mid_price"), errors="coerce").mean()),
    }
    return pd.DataFrame([row])


def _merge_features(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("No metadata frames loaded.")
    df = pd.concat(frames, ignore_index=True)
    df = df.groupby("ts").mean(numeric_only=True).reset_index()
    df = df.sort_values("ts")
    return df


def _align_to_prices(df: pd.DataFrame, prices_csv: Path) -> pd.DataFrame:
    price, _vol, ts = load_prices(prices_csv, return_time=True)
    ts_int = pd.to_datetime(ts).astype("int64") if ts is not None else np.arange(price.size, dtype=np.int64)
    base = pd.DataFrame({"ts": ts_int})
    df = df.sort_values("ts")
    out = pd.merge_asof(base, df, on="ts", direction="backward")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build instrument-head features from market metadata.")
    ap.add_argument("--premium-json", type=Path, required=True)
    ap.add_argument("--oi-json", type=Path, required=True)
    ap.add_argument("--options-instruments", type=Path, required=True)
    ap.add_argument("--options-summary", type=Path, required=True)
    ap.add_argument("--prices-csv", type=Path, default=None, help="Optional price CSV for alignment.")
    ap.add_argument("--out", type=Path, default=Path("logs/market_meta_features.csv"))
    args = ap.parse_args()

    frames = [
        _load_premium(args.premium_json),
        _load_open_interest(args.oi_json),
        _load_deribit_instruments(args.options_instruments),
        _load_deribit_summary(args.options_summary),
    ]
    df = _merge_features(frames)
    if args.prices_csv is not None:
        df = _align_to_prices(df, args.prices_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

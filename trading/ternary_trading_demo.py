"""
ternary_trading_demo.py
-----------------------
Self-contained ternary trading demo with no API keys:

- Loads daily close data from a local CSV (if available) with columns: date, close.
- If no data is found, generates a synthetic random walk.
- Encodes returns into {-1, 0, +1} using a volatility-scaled dead-zone.
- Simple MoE-style gate:
    * low vol  -> trend-follow (use ternary return)
    * high vol -> mean-revert (invert ternary return)
- Backtests with per-trade cost and reports basic metrics.

Run: `python trading/ternary_trading_demo.py`
"""

import os
import math
import time
import numpy as np
import pandas as pd


def load_prices(path="prices.csv", n=1000, seed=0):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "close" not in df.columns:
            raise ValueError("CSV must contain a 'close' column")
        close = df["close"].astype(float).to_numpy()
    else:
        rng = np.random.default_rng(seed)
        steps = rng.normal(loc=0.0, scale=1.0, size=n)
        close = 100 + np.cumsum(steps)
    return close


def ternary_encode(rets, sigma, k=0.5):
    dead = k * sigma
    sig = np.zeros_like(rets, dtype=np.int8)
    sig[rets > dead] = 1
    sig[rets < -dead] = -1
    return sig


def backtest(close, k_dead=0.5, vol_window=20, vol_gate=1.5, fee_bp=1.0):
    fee = fee_bp / 1e4
    rets = np.diff(close) / close[:-1]
    vol = pd.Series(rets).rolling(vol_window, min_periods=1).std().to_numpy()
    tern = ternary_encode(rets, vol, k=k_dead)

    # MoE-style gate: low vol -> trend-follow, high vol -> mean-revert
    signals = np.zeros_like(tern)
    for i in range(len(tern)):
        if vol[i] < vol_gate * np.median(vol[max(0, i - vol_window): i + 1]):
            signals[i] = tern[i]
        else:
            signals[i] = -tern[i]

    # Strategy PnL with transaction costs
    pos = 0
    pnl = []
    trades = 0
    for s, r in zip(signals, rets):
        if s != pos:
            trades += 1
            pos = s
            r -= fee * abs(pos)  # cost on change
        pnl.append(pos * r)
    pnl = np.array(pnl)

    # Baseline: binary always-active (sign of return)
    base_sig = np.sign(rets)
    base_pos = base_sig
    base_cost = np.zeros_like(rets)
    base_cost[1:] = fee * (base_pos[1:] != base_pos[:-1]).astype(float)
    base_pnl = base_pos * rets - base_cost

    def stats(p):
        cum = p.sum()
        volp = p.std() * math.sqrt(252)
        sharpe = cum / (volp + 1e-9)
        hit = (p > 0).mean()
        return cum, sharpe, hit

    cum, sharpe, hit = stats(pnl)
    bcum, bsharpe, bhit = stats(base_pnl)

    return {
        "rets": rets,
        "vol": vol,
        "tern": tern,
        "signals": signals,
        "pnl": pnl,
        "trades": trades,
        "cum": cum,
        "sharpe": sharpe,
        "hit": hit,
        "baseline": {
            "cum": bcum,
            "sharpe": bsharpe,
            "hit": bhit,
        },
    }


def main():
    close = load_prices()
    t0 = time.perf_counter()
    res = backtest(close)
    t1 = time.perf_counter()

    print("Ternary trading demo (no API keys, local/synthetic data)")
    print(f"Close series length: {len(close)}")
    print(f"Trades executed    : {res['trades']}")
    print(f"Ternary cum return : {res['cum']*100:.2f}%   Sharpe: {res['sharpe']:.2f}   Hit rate: {res['hit']*100:.1f}%")
    print(f"Baseline (binary)  : {res['baseline']['cum']*100:.2f}%   Sharpe: {res['baseline']['sharpe']:.2f}   Hit rate: {res['baseline']['hit']*100:.1f}%")
    print(f"Runtime            : {(t1 - t0)*1e3:.2f} ms")
    print("\nTo use real data, provide prices.csv with columns: date, close.")


if __name__ == "__main__":
    main()

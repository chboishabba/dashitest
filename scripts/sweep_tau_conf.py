"""
Sweep tau_conf (HOLD gate) and report stability metrics.
This keeps direction/execution fixed and varies only the confidence threshold.
Metrics: trades, HOLD%, max drawdown, pnl.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from runner import run_bars
from run_trader import load_prices, compute_triadic_state
from scripts.run_bars_btc import confidence_from_disagreement


def max_drawdown(pnl_series):
    s = pd.Series(pnl_series)
    running_max = s.cummax()
    dd = (s - running_max).min()
    return float(dd)


def run_once(tau_conf):
    csv = Path("data/raw/stooq/btc_intraday.csv")
    price, _ = load_prices(csv)
    ts = np.arange(len(price))
    state = compute_triadic_state(price)
    bars = pd.DataFrame({"ts": ts, "close": price, "state": state})
    rets = np.diff(price, prepend=price[0])
    vol = pd.Series(rets).rolling(50).std().to_numpy()
    conf_seq = confidence_from_disagreement(
        state, window=50, ret_vol=vol, vol_thresh=np.nanpercentile(vol, 80)
    )

    def conf_fn(t, s):
        idx = int(t)
        if idx < 0 or idx >= len(conf_seq):
            return 1.0
        return conf_seq[idx]

    df = run_bars(
        bars,
        symbol="BTCUSDT",
        mode="bar",
        log_path=None,  # no need to write per sweep
        confidence_fn=conf_fn,
        tau_conf=tau_conf,
    )
    trades = (df["fill"] != 0).sum()
    hold = df["hold"].mean() if "hold" in df else 0.0
    pnl = df["pnl"].iloc[-1] if not df.empty else 0.0
    dd = max_drawdown(df["pnl"]) if not df.empty else 0.0
    return {"tau_conf": tau_conf, "trades": trades, "hold": hold, "pnl": pnl, "max_dd": dd}


def main():
    taus = [0.0, 0.2, 0.4, 0.6, 0.8]
    rows = [run_once(t) for t in taus]
    df = pd.DataFrame(rows)
    print(df)


if __name__ == "__main__":
    main()

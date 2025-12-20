import numpy as np
import pandas as pd
from pathlib import Path
from runner import run_bars
from run_trader import load_prices, compute_triadic_state
import math


def confidence_from_persistence(states: np.ndarray, run_scale: int = 20, ret_vol=None, vol_thresh=None):
    """
    Confidence grows with run length of a non-zero state; decays on flips/zero.
    Optionally zero confidence when volatility is high.
    """
    states = np.asarray(states, dtype=int)
    conf = np.zeros(len(states))
    run_len = 0
    for i, s in enumerate(states):
        if s == 0:
            run_len = 0
            conf[i] = 0.0
        else:
            if i > 0 and s == states[i - 1]:
                run_len += 1
            else:
                run_len = 1
            conf[i] = 1.0 - math.exp(-run_len / run_scale)
        if vol_thresh is not None and ret_vol is not None and i < len(ret_vol):
            if ret_vol[i] > vol_thresh:
                conf[i] = 0.0
    return conf


def main():
    # confidence threshold sweep hook (set here)
    tau_conf_enter = 0.5  # tau_on
    tau_conf_exit = 0.3   # tau_off

    csv = Path("data/raw/stooq/btc_intraday.csv")
    if not csv.exists():
        raise SystemExit("btc_intraday.csv not found; run data_downloader.py first.")

    price, _ = load_prices(csv)
    ts = np.arange(len(price))
    # Use the same triadic state logic as run_trader (EWMA of vol-normalized returns)
    state = compute_triadic_state(price)
    bars = pd.DataFrame({"ts": ts, "close": price, "state": state})

    # Build a confidence function closure over the state history
    rets = np.diff(price, prepend=price[0])
    vol = pd.Series(rets).rolling(50).std().to_numpy()
    conf_seq = confidence_from_persistence(
        state, run_scale=30, ret_vol=vol, vol_thresh=np.nanpercentile(vol, 80)
    )

    def conf_fn(t, s):
        # t is timestamp index; we map ts back to index
        idx = int(t)
        if idx < 0 or idx >= len(conf_seq):
            return 1.0
        return conf_seq[idx]

    run_bars(
        bars,
        symbol="BTCUSDT",
        mode="bar",
        log_path="logs/trading_log.csv",
        confidence_fn=conf_fn,
        tau_conf_enter=tau_conf_enter,
        tau_conf_exit=tau_conf_exit,
    )
    print("Wrote", len(bars), "rows to logs/trading_log.csv")


if __name__ == "__main__":
    main()

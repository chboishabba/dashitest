import numpy as np
import pandas as pd
from pathlib import Path
from runner import run_bars
from run_trader import load_prices, compute_triadic_state


def confidence_from_disagreement(states: np.ndarray, window: int = 50, ret_vol=None, vol_thresh=None):
    """
    Compute a confidence score in [0,1] based on disagreement/volatility of the state sequence.
    Lower confidence when the state flips often in the recent window or when vol is high.
    """
    states = np.asarray(states, dtype=int)
    flips = np.abs(np.diff(states))
    # rolling flip rate
    conf = np.ones(len(states))
    for i in range(len(states)):
        lo = max(0, i - window)
        window_flips = flips[lo:i].sum()
        # rate in [0,2] roughly; scale to penalize
        rate = window_flips / max(1, i - lo)
        conf[i] = max(0.0, 1.0 - 10.0 * rate)  # more flips -> lower confidence
        if vol_thresh is not None and ret_vol is not None and i < len(ret_vol):
            if ret_vol[i] > vol_thresh:
                conf[i] = 0.0
    return conf


def main():
    # confidence threshold sweep hook (set here)
    tau_conf = 0.5

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
    conf_seq = confidence_from_disagreement(state, window=50, ret_vol=vol, vol_thresh=np.nanpercentile(vol, 80))

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
        tau_conf=tau_conf,
    )
    print("Wrote", len(bars), "rows to logs/trading_log.csv")


if __name__ == "__main__":
    main()

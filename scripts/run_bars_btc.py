import numpy as np
import pandas as pd
from pathlib import Path
from runner import run_bars
from run_trader import load_prices

# Simple EMA + dead-zone state generator
def ema(x, alpha=0.03):
    y = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def make_state(price, alpha=0.03, dead=0.0005):
    rets = np.diff(price, prepend=price[0])
    score = ema(rets, alpha=alpha)
    state = np.zeros(len(score), dtype=int)
    state[score > dead] = +1
    state[score < -dead] = -1
    return state


def main():
    csv = Path("data/raw/stooq/btc_intraday.csv")
    if not csv.exists():
        raise SystemExit("btc_intraday.csv not found; run data_downloader.py first.")

    price, _ = load_prices(csv)
    ts = np.arange(len(price))
    state = make_state(price, alpha=0.03, dead=0.0005)
    bars = pd.DataFrame({"ts": ts, "close": price, "state": state})

    run_bars(bars, symbol="BTCUSDT", mode="bar", log_path="logs/trading_log.csv")
    print("Wrote", len(bars), "rows to logs/trading_log.csv")


if __name__ == "__main__":
    main()

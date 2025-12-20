"""
Execution runner with pluggable execution backends.

Strategy/state logic remains in run_trader; this runner is a stub to show
how to route intents through either bar-level execution or LOB replay via
hftbacktest when data is available.

Intent schema (expected):
  ts, side {-1,0,+1}, target_exposure, order_style, urgency, ttl

Execution backends:
  - BarExecution: existing bar-level execution (placeholder)
  - LOBReplayExecution: hftbacktest-based (requires L2+trades data)

TODO:
  - Wire strategy to emit intents instead of direct fills.
  - Implement BarExecution using the current run_trader logic.
  - Implement LOBReplayExecution using prepared Binance BTC/ETH data.
"""

from execution.bar_exec import BarExecution
from execution.hft_exec import LOBReplayExecution
from execution.intent import Intent
from strategy.triadic_strategy import TriadicStrategy
import pandas as pd
import pathlib


def get_executor(symbol: str, mode: str = "auto", lob_symbols=None):
    lob_symbols = lob_symbols or {"BTCUSDT", "ETHUSDT"}
    if mode == "lob" or (mode == "auto" and symbol.upper() in lob_symbols):
        return LOBReplayExecution(symbol=symbol)
    return BarExecution()


def run_bars(bars: pd.DataFrame, symbol: str, mode: str = "auto", log_path: str = None):
    """
    bars: DataFrame with columns [ts, close, state] (state ∈ {-1,0,+1})
    symbol: symbol string
    mode: "auto" | "bar" | "lob"
    log_path: optional CSV path for dashboard compatibility (logs/trading_log.csv)
    """
    # strategy emits intents from triadic state
    strategy = TriadicStrategy(symbol=symbol)
    executor = get_executor(symbol, mode)
    logs = []
    equity = 1.0
    prev_price = None
    prev_exposure = 0.0
    log_file = pathlib.Path(log_path) if log_path else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.unlink(missing_ok=True)
    for _, row in bars.iterrows():
        ts = int(row["ts"])
        state = int(row["state"])
        price = float(row["close"])
        # mark-to-market on previous exposure
        if prev_price is not None:
            ret = (price / prev_price) - 1.0
            equity *= (1 + prev_exposure * ret)
        intent = strategy.step(ts=ts, state=state)
        result = executor.execute(intent, price)
        equity += result.get("pnl", 0.0)
        logs.append(
            {
                "t": ts,  # dashboard-friendly
                "ts": ts,
                "symbol": symbol,
                "state": state,
                "intent_direction": intent.direction,
                "intent_target": intent.target_exposure,
                "urgency": intent.urgency,
                "fill": result["filled"],
                "fill_price": result["fill_price"],
                "fee": result["fee"],
                "pnl": equity - 1.0,
                "exposure": result["exposure"],
                "slippage": result["slippage"],
                # dashboard-friendly fields
                "price": price,
                "action": intent.direction,
                "hold": int(intent.direction == 0),
                "z_vel": 0.0,  # placeholder for dashboard compatibility
            }
        )
        if log_file:
            pd.DataFrame([logs[-1]]).to_csv(
                log_file, mode="a", header=not log_file.exists(), index=False
            )
        prev_price = price
        prev_exposure = result["exposure"]
    return pd.DataFrame(logs)


if __name__ == "__main__":
    # Demo: build synthetic bars/state for BTCUSDT and run bar executor
    import numpy as np

    ts = np.arange(0, 1000)
    price = 100 + np.cumsum(np.random.normal(0, 0.1, size=len(ts)))
    # simple synthetic state: + during up moves, - during down, 0 occasionally
    state = np.sign(np.diff(price, prepend=price[0]))
    # inject some HOLD
    state[::10] = 0
    bars = pd.DataFrame({"ts": ts, "close": price, "state": state})

    df = run_bars(bars, symbol="BTCUSDT", mode="bar")
    print(df.tail())

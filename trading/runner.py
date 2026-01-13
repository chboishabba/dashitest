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

try:
    from trading.bar_exec import BarExecution
    from trading.hft_exec import LOBReplayExecution
    from trading.intent import Intent
    from trading.strategy.triadic_strategy import TriadicStrategy
    from trading.strategy.learner_adapter import LearnerAdapter
    from trading.regime import RegimeSpec, check_regime
except ModuleNotFoundError:
    from bar_exec import BarExecution
    from hft_exec import LOBReplayExecution
    from intent import Intent
    from strategy.triadic_strategy import TriadicStrategy
    from strategy.learner_adapter import LearnerAdapter
    from regime import RegimeSpec, check_regime
import math
import pandas as pd
import pathlib
import numpy as np


def get_executor(symbol: str, mode: str = "auto", lob_symbols=None):
    lob_symbols = lob_symbols or {"BTCUSDT", "ETHUSDT"}
    if mode == "lob" or (mode == "auto" and symbol.upper() in lob_symbols):
        return LOBReplayExecution(symbol=symbol)
    return BarExecution()


def run_bars(
    bars: pd.DataFrame,
    symbol: str,
    mode: str = "auto",
    log_path: str = None,
    confidence_fn=None,
    tau_conf_enter: float = 0.0,
    tau_conf_exit: float = 0.0,
    use_stub_adapter: bool = False,
    adapter_kwargs: dict = None,
):
    """
    bars: DataFrame with columns [ts, close, state] (state ∈ {-1,0,+1})
    symbol: symbol string
    mode: "auto" | "bar" | "lob"
    log_path: optional CSV path for dashboard compatibility (logs/trading_log.csv)
    confidence_fn: optional callable(ts, state) -> confidence in [0,1]
    tau_conf: (deprecated) use tau_on/tau_off
    tau_on/tau_off: hysteresis thresholds; ACT when conf >= tau_on, HOLD when conf < tau_off
    use_stub_adapter: if True and confidence_fn is None, uses LearnerAdapter stub to supply ℓ
    """
    # backward compatibility
    if tau_conf_enter is not None and tau_conf_exit is not None and tau_conf_enter < tau_conf_exit:
        raise ValueError("tau_conf_enter should be >= tau_conf_exit for hysteresis")

    adapter = None
    current_price = None
    last_ell = float("nan")
    if confidence_fn is None and use_stub_adapter:
        adapter = LearnerAdapter(**(adapter_kwargs or {}))

        def confidence_fn(ts, state):
            nonlocal last_ell
            payload = {"state": state, "price": current_price}
            ell, _qfeat = adapter.update(ts, payload)
            last_ell = float(ell)
            return ell

    if confidence_fn is not None:
        orig_conf_fn = confidence_fn

        def confidence_fn(ts, state):
            nonlocal last_ell
            out = orig_conf_fn(ts, state)
            ell_val = out[0] if isinstance(out, tuple) and len(out) >= 1 else out
            try:
                ell_f = float(ell_val)
                if math.isfinite(ell_f):
                    last_ell = ell_f
            except (TypeError, ValueError):
                pass
            return out

    # strategy emits intents from triadic state
    strategy = TriadicStrategy(
        symbol=symbol,
        confidence_fn=confidence_fn,
        tau_on=tau_conf_enter,
        tau_off=tau_conf_exit,
    )
    executor = get_executor(symbol, mode)
    logs = []
    equity = 1.0
    prev_price = None
    prev_exposure = 0.0
    log_file = pathlib.Path(log_path) if log_path else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.unlink(missing_ok=True)
    # regime acceptance (PnL-free): rolling vol + run-length/flip-rate
    price_arr = bars["close"].to_numpy()
    rets = np.diff(price_arr, prepend=price_arr[0])
    vols = pd.Series(rets).rolling(50).std().to_numpy()
    spec = RegimeSpec()  # defaults; adjust if needed
    acceptable = np.asarray(check_regime(bars["state"].to_numpy(), vols, spec), dtype=bool)
    if acceptable.shape[0] != len(bars):
        raise ValueError(f"acceptable length {acceptable.shape[0]} != bars length {len(bars)}")
    for i, row in enumerate(bars.itertuples(index=False)):
        ts = int(getattr(row, "ts"))
        state = int(getattr(row, "state"))
        price = float(getattr(row, "close"))
        volume = float(getattr(row, "volume")) if hasattr(row, "volume") else np.nan
        current_price = price
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
                "acceptable": bool(acceptable[i]),
                "state": state,
                "intent_direction": intent.direction,
                "intent_target": intent.target_exposure,
                "urgency": intent.urgency,
                "actionability": intent.actionability,
                "ell": last_ell,
                "fill": result["filled"],
                "fill_price": result["fill_price"],
                "fee": result["fee"],
                "pnl": equity - 1.0,
                "exposure": result["exposure"],
                "slippage": result["slippage"],
                # dashboard-friendly fields
                "price": price,
                "volume": volume,
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

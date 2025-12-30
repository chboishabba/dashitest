"""
Stub adapter for LOB replay using hftbacktest.

Expected data:
  - LOB events (book updates) and trades for BTCUSDT/ETHUSDT (Binance spot or perps)
  - Timestamped in exchange time; schema to be defined when data is available.

Intents:
  {ts, side {-1,0,+1}, target_exposure, order_style, urgency, ttl}

Outputs:
  fills: [{ts, qty, price}], summary metrics (fees, slippage, fill_ratio, queue_delay, impact)

Integration plan:
  - Load prepared parquet/npz of book+trades.
  - Feed intents into hftbacktest simulator to get fills.
  - Keep strategy/state logic unchanged; this is an execution backend.
"""

from trading.base import BaseExecution


class LOBReplayExecution(BaseExecution):
    def __init__(self, data_path=None, symbol="BTCUSDT"):
        self.data_path = data_path
        self.symbol = symbol

    def execute(self, intents):
        # TODO: integrate hftbacktest when L2/trades data is available.
        # For now, return no fills.
        return [], {}

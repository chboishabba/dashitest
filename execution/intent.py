from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    """
    Strategy output (frozen contract).

    ts: integer timestamp (bar close or exchange time)
    symbol: e.g. BTCUSDT
    direction: -1, 0, +1 (SELL, HOLD, BUY). If 0, target_exposure must be 0.
    target_exposure: desired portfolio fraction [0, 1]
    urgency: [0, 1] hint for executor (0=passive, 1=aggressive). Executor may ignore.
    ttl_ms: time-to-live for passive intents. Executor may ignore.
    """

    ts: int
    symbol: str
    direction: int
    target_exposure: float
    urgency: float
    ttl_ms: int

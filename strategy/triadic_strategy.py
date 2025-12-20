"""
Triadic strategy wrapper: turns state -> Intent.
This mirrors the existing run_trader logic but emits intents instead of trades.
No learning here; purely deterministic for now.
"""

import math
from execution.intent import Intent


class TriadicStrategy:
    def __init__(self, symbol: str, base_size: float = 0.05):
        self.symbol = symbol
        self.base_size = base_size
        self.position = 0  # current thesis (direction)
        self.align_age = 0
        self.prev_state = 0

    def step(self, ts: int, state: int):
        """
        state: s_t ∈ {-1, 0, +1}
        """
        # alignment: persistence of non-zero state
        if state != 0 and state == self.prev_state:
            self.align_age += 1
        else:
            self.align_age = 0

        if state == 0:
            direction = 0
            target_exposure = 0.0
            urgency = 0.0
            ttl = 0
        else:
            direction = state
            # size ramps with alignment age, slower ramp to reduce churn
            ramp = 1.0 - math.exp(-self.align_age / 20.0)
            target_exposure = self.base_size * ramp
            # urgency increases if alignment is strong
            urgency = min(1.0, 0.3 + 0.1 * self.align_age)
            ttl = 500  # ms, advisory

        # update internal thesis
        self.position = direction if target_exposure > 0 else 0
        self.prev_state = state

        return Intent(
            ts=int(ts),
            symbol=self.symbol,
            direction=direction,
            target_exposure=max(0.0, min(1.0, target_exposure)),
            urgency=max(0.0, min(1.0, urgency)),
            ttl_ms=int(ttl),
        )

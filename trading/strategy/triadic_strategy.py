from __future__ import annotations

"""
Triadic strategy wrapper: turns state -> Intent.
This mirrors the existing run_trader logic but emits intents instead of trades.
No learning here; purely deterministic for now.
"""

import math
try:
    from trading.intent import Intent
    from trading.signals.triadic import BUY, SELL, FLAT, UNKNOWN, PARADOX
    from trading.posture import Posture
    from trading.signals.asymmetry_sensor import InfluenceTensorMonitor
except ModuleNotFoundError:
    from intent import Intent
    from signals.triadic import BUY, SELL, FLAT, UNKNOWN, PARADOX
    from posture import Posture
    from signals.asymmetry_sensor import InfluenceTensorMonitor


class TriadicStrategy:
    def __init__(
        self,
        symbol: str,
        base_size: float = 0.05,
        confidence_fn=None,
        tau_on: float = 0.0,
        tau_off: float = 0.0,
        scaffolding_a: int = 0,
        influence_monitor: InfluenceTensorMonitor | None = None,
    ):
        self.symbol = symbol
        self.base_size = base_size
        self.position = 0  # current thesis (direction)
        self.align_age = 0
        self.prev_state = 0
        self.confidence_fn = confidence_fn
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.scaffolding_a = scaffolding_a  # Pre-failure detection (A)
        self._is_holding = False
        self.influence_monitor = influence_monitor
        assert self.tau_on >= self.tau_off, "Require tau_on >= tau_off for hysteresis"

    def step(self, ts: int, state: int, posture: Posture = Posture.TRADE_NORMAL):
        """
        state: s_t ∈ {SELL, FLAT, BUY, UNKNOWN, PARADOX}
        posture: High-level engagement mode from ABC Machine.
        """
        # 0. Posture Override
        if posture == Posture.UNWIND:
            return Intent(
                ts=int(ts),
                symbol=self.symbol,
                direction=0,
                target_exposure=0.0,
                urgency=1.0,
                ttl_ms=0,
                hold=False,
                actionability=0.0,
                reason="posture_unwind (M9)",
            )
        elif posture == Posture.OBSERVE:
             # Similar to UNKNOWN, but more persistent
             direction = self.position
             target_exposure = 0.0
             urgency = 0.0
             hold_flag = True
             return Intent(
                 ts=int(ts),
                 symbol=self.symbol,
                 direction=direction,
                 target_exposure=0.0,
                 urgency=0.0,
                 ttl_ms=500,
                 hold=True,
                 actionability=0.0,
                 reason="posture_observe (M4/M6)",
             )

        # 1. Pre-failure detection (A = -1 forces Hard Closure)
        if self.scaffolding_a == -1 or state == PARADOX:
            return Intent(
                ts=int(ts),
                symbol=self.symbol,
                direction=0,
                target_exposure=0.0,
                urgency=1.0,  # Panic exit
                ttl_ms=0,
                hold=False,
                actionability=0.0,
                reason="systemic_collapse_prohibit (⚡)",
            )

        # 2. Alignment: persistence of non-zero state
        if state != FLAT and state != UNKNOWN and state == self.prev_state:
            self.align_age += 1
        else:
            self.align_age = 0

        # 3. Epistemic Gating (Confidence/Legitimacy)
        conf = 1.0
        if self.confidence_fn is not None:
            raw_conf = self.confidence_fn(ts, state)
            if isinstance(raw_conf, tuple):
                raw_conf = raw_conf[0]
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

        # 4. Hysteresis & M5 Suspension Buffer
        # P = Permission axis: Prohibited (-1), Suspended (0), Permitted (+1)
        if not self._is_holding: # Currently ACT
            if conf < self.tau_off:
                permission = -1 # Prohibit
            else:
                permission = 1  # Permitted
        else: # Currently HOLD
            if conf >= self.tau_on:
                permission = 1  # Permit
            else:
                permission = 0  # Suspended / Hold

        # 5. Intent Determination
        reason = ""
        if state == FLAT:
            direction = 0
            target_exposure = 0.0
            urgency = 0.5
            hold_flag = False
            reason = "intentional_flat"
        elif state == UNKNOWN:
            # M6 Tension: Abstain from decision. Keep current position.
            direction = self.position
            target_exposure = 0.0
            urgency = 0.0
            hold_flag = True
            reason = "epistemic_unknown (⊥)"
        elif permission == 0:
            direction = self.position
            target_exposure = 0.0
            urgency = 0.0
            hold_flag = True
            reason = "m5_suspension_buffer"
        elif permission == -1:
            direction = 0
            target_exposure = 0.0
            urgency = 0.2
            hold_flag = False
            reason = "prohibited_by_legitimacy"
        else:
            # Permitted: ACT if state is intensional
            direction = state
            ramp = 1.0 - math.exp(-self.align_age / 20.0)
            
            # TRADE_CONVEX posture increases base size
            current_base_size = self.base_size
            if posture == Posture.TRADE_CONVEX:
                current_base_size *= 2.0
                reason = "active_triadic (CONVEX)"
            else:
                reason = "active_triadic"
                
            target_exposure = current_base_size * ramp * conf
            urgency = min(1.0, (0.3 + 0.1 * self.align_age) * conf)
            hold_flag = False

        sensor_override = False
        if (
            self.influence_monitor
            and state not in (FLAT, UNKNOWN, PARADOX)
            and self.influence_monitor.should_escalate(self.symbol, state)
        ):
            sensor_override = True
            permission = 1

        if sensor_override:
            reason = f"{reason} + influence_sensor" if reason else "influence_sensor"

        # update internal thesis
        self.position = direction if hold_flag or target_exposure > 0 else 0
        self.prev_state = state
        self._is_holding = hold_flag

        return Intent(
            ts=int(ts),
            symbol=self.symbol,
            direction=direction,
            target_exposure=max(0.0, min(1.0, target_exposure)),
            urgency=max(0.0, min(1.0, urgency)),
            ttl_ms=500,
            hold=hold_flag,
            actionability=conf,
            reason=reason,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math
from collections import deque

import numpy as np

try:
    from trading.features.quotient import compute_quotient_features
except ModuleNotFoundError:
    from features.quotient import compute_quotient_features


@dataclass
class LearnerOutput:
    """Container for learner outputs; kept for future GPU-backed use."""

    ell: float  # legitimacy scalar in [0, 1]
    qfeat: Dict[str, float]


class LearnerAdapter:
    """
    Permission-only learner adapter.

    Contract:
      - update(ts, state) -> (ell, qfeat)
      - ell ∈ [0,1] gates ACT/HOLD via TriadicStrategy.confidence_fn
      - No PnL-based loss; no directional or sizing signals.
      - GPU implementation can replace the internals without touching callers.
    """

    supports_gpu: bool = False

    def __init__(
        self,
        *,
        window: int = 128,
        smoothing: int = 1,
        stub_mode: str = "constant",  # "constant" | "vol_proxy" | "schedule" | "qfeat_var"
        stub_constant: float = 0.5,
        w1: int = 64,
        w2: int = 256,
        ell_beta: float = 1.0,
        beta_var: float = 0.5,
        beta_curv: float = 0.5,
        beta_acorr: float = 0.5,
        q_hist_fast: int = 50,
        q_hist_slow: int = 200,
        warmup_min: int = 10,
    ) -> None:
        self.window = int(window)
        self.smoothing = int(smoothing)
        self.stub_mode = str(stub_mode)
        self.stub_constant = float(stub_constant)
        self._t = 0  # simple counter for schedule stub
        self._prev_price: Optional[float] = None
        self._log_returns: deque[float] = deque(maxlen=max(w1, w2, window))
        self._q_hist_fast: deque[np.ndarray] = deque(maxlen=q_hist_fast)
        self._q_hist_slow: deque[np.ndarray] = deque(maxlen=q_hist_slow)
        self.w1 = int(w1)
        self.w2 = int(w2)
        self.ell_beta = float(ell_beta)
        self.beta_var = float(beta_var)
        self.beta_curv = float(beta_curv)
        self.beta_acorr = float(beta_acorr)
        self.warmup_min = int(warmup_min)

    def update(self, ts: Any, state: Any) -> Tuple[float, Dict[str, float]]:
        """
        Current stub: deterministic ℓ for wiring/hysteresis validation.

        Later:
          - extract window W_t from state
          - compute qfeat_t (GPU)
          - predict qfeat_hat_{t+1} (GPU)
          - ell = exp(-||qfeat_hat - qfeat||)
        """
        self._t += 1

        if self.stub_mode == "constant":
            ell = self.stub_constant
            qfeat = {"stub": 1.0}

        elif self.stub_mode == "schedule":
            # simple triangle wave in [0,1] over a fixed period to test hysteresis
            period = 200
            phase = (self._t % period) / period
            ell = 2.0 * phase if phase <= 0.5 else 2.0 * (1.0 - phase)
            ell = max(0.0, min(1.0, ell))
            qfeat = {"stub_phase": phase}

        elif self.stub_mode == "vol_proxy":
            returns = _try_get_returns(state, self.window)
            if returns is None or len(returns) < 8:
                ell = self.stub_constant
                qfeat = {"stub": 1.0}
            else:
                mean = sum(returns) / len(returns)
                var = sum((x - mean) ** 2 for x in returns) / max(1, len(returns) - 1)
                vol = math.sqrt(max(0.0, var))
                # higher vol → lower legitimacy (placeholder scale)
                ell = math.exp(-10.0 * vol)
            ell = max(0.0, min(1.0, ell))
            qfeat = {"vol": float(vol)}

        elif self.stub_mode == "qfeat_var":
            price = _try_get_price(state)
            if price is None:
                ell = self.stub_constant
                qfeat = {"stub": 1.0}
            else:
                ell, qfeat = self._ell_from_qfeat(price)

        else:
            raise ValueError(f"Unknown stub_mode={self.stub_mode!r}")

        return float(ell), qfeat

    def _ell_from_qfeat(self, price: float) -> Tuple[float, Dict[str, float]]:
        if self._prev_price is not None and price > 0:
            ret = math.log(price / self._prev_price)
            self._log_returns.append(ret)
        self._prev_price = price

        rets = np.array(self._log_returns, dtype=float)
        qfeat = compute_quotient_features(list(self._log_returns), w1=self.w1, w2=self.w2)
        qvec_raw = np.array([v for v in qfeat.values()], dtype=float)
        if not np.isfinite(qvec_raw).any():
            return 0.0, {"stub": 1.0}

        # Preserve shape; keep NaNs but allow centroid via nanmean.
        qvec = np.where(np.isfinite(qvec_raw), qvec_raw, np.nan)
        self._q_hist_fast.append(qvec)
        self._q_hist_slow.append(qvec)

        # Warmup: force HOLD until we have enough history and returns.
        if len(self._q_hist_fast) < self.warmup_min or rets.size < self.warmup_min:
            return 0.0, qfeat

        hist_fast = np.stack(list(self._q_hist_fast))
        hist_slow = np.stack(list(self._q_hist_slow))
        centroid_slow = np.nanmean(hist_slow, axis=0)
        diff_q = np.nan_to_num(qvec - centroid_slow, nan=0.0, posinf=0.0, neginf=0.0)
        dist_q = np.linalg.norm(diff_q)

        var_pen = math.log1p(float(np.nanstd(rets))) if rets.size > 1 else 0.0
        curv_pen = math.log1p(float(np.nanstd(np.diff(rets)))) if rets.size > 2 else 0.0
        acorr_pen = abs(_lag1_autocorr(rets)) if rets.size > 2 else 0.0

        penalty = (
            self.ell_beta * dist_q
            + self.beta_var * var_pen
            + self.beta_curv * curv_pen
            + self.beta_acorr * acorr_pen
        )
        ell = math.exp(-penalty)
        ell = max(0.0, min(1.0, ell))
        return ell, {**qfeat, "var_pen": var_pen, "curv_pen": curv_pen, "acorr_pen": acorr_pen}


def _try_get_returns(state: Any, window: int) -> Optional[list[float]]:
    """
    Best-effort extractor for returns from a loosely-typed state object.
    Replace with the real accessor once the learner is GPU-backed.
    """
    # direct returns attribute/key
    if hasattr(state, "returns"):
        r = getattr(state, "returns")
        return list(r)[-window:]
    if isinstance(state, dict) and "returns" in state:
        return list(state["returns"])[-window:]

    # derive from prices if present
    prices = None
    if hasattr(state, "prices"):
        prices = getattr(state, "prices")
    elif isinstance(state, dict) and "prices" in state:
        prices = state["prices"]

    if prices is None:
        return None

    p = list(prices)[-window:]
    if len(p) < 2:
        return None
    return [float(p[i] / p[i - 1] - 1.0) for i in range(1, len(p))]


def _try_get_price(state: Any) -> Optional[float]:
    if isinstance(state, dict) and "price" in state:
        try:
            return float(state["price"])
        except Exception:
            return None
    if hasattr(state, "price"):
        try:
            return float(getattr(state, "price"))
        except Exception:
            return None
    if isinstance(state, (int, float)):
        return float(state)
    return None


def _lag1_autocorr(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x = x - np.nanmean(x)
    x0 = x[:-1]
    x1 = x[1:]
    denom = np.sqrt(np.nansum(x0 * x0) * np.nansum(x1 * x1))
    if denom <= 0:
        return 0.0
    return float(np.nansum(x0 * x1) / denom)

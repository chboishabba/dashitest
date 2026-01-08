#!/usr/bin/env python3
"""
Minimal operator learner that fits a contractive linear map on band-energy sequences.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Sequence, Union


def _softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus to keep band energies positive."""
    return np.logaddexp(0.0, x)


def _spectral_norm(matrix: np.ndarray) -> float:
    """Spectral norm via the 2-norm (maximum singular value)."""
    return float(np.linalg.norm(matrix, ord=2))


def _enforce_contractive(raw: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Scale `raw` so that its spectral norm is at most `alpha`."""
    norm = _spectral_norm(raw)
    if norm == 0.0:
        return raw.copy(), 0.0
    factor = min(alpha / norm, 1.0)
    return raw * factor, norm


class OperatorLearner:
    """Contractive linear operator learner on band-energy space."""

    def __init__(self, alpha: float = 0.9) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.spectral_norm: float | None = None
        self._raw_norm: float | None = None
        self.train_loss: float | None = None
        self._trained = False

    def fit(self, E_seq: Sequence[Sequence[float]]) -> None:
        """Fit W,b from E_seq where next-state prediction is the target."""
        seq = np.asarray(E_seq, dtype=np.float64)
        if seq.ndim != 2:
            raise ValueError("E_seq must be 2D (T, B)")
        if seq.shape[0] < 2:
            raise ValueError("Need at least two time steps to fit")
        X = seq[:-1]
        Y = seq[1:]
        ones = np.ones((X.shape[0], 1), dtype=seq.dtype)
        design = np.concatenate([X, ones], axis=1)
        coeff, *_ = np.linalg.lstsq(design, Y, rcond=None)
        W_raw = coeff[:-1, :]
        b = coeff[-1, :]
        W_contractive, raw_norm = _enforce_contractive(W_raw, self.alpha)
        self._raw_norm = raw_norm
        self.W = W_contractive
        self.b = b
        self.spectral_norm = _spectral_norm(self.W)
        preds = self._apply(X)
        self.train_loss = float(np.mean((preds - Y) ** 2))
        self._trained = True

    def _apply(self, inputs: np.ndarray) -> np.ndarray:
        """Apply the contractive map plus positivity."""
        if self.W is None or self.b is None:
            raise RuntimeError("Learner not fitted")
        raw = inputs @ self.W + self.b
        return _softplus(raw)

    def predict_next(self, state: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Predict the next band-energy vector from a single state."""
        if not self._trained:
            raise RuntimeError("Learner not fitted")
        state_arr = np.asarray(state, dtype=np.float64)
        if state_arr.ndim != 1:
            raise ValueError("state must be a 1D band-energy vector")
        return self._apply(state_arr.reshape(1, -1))[0]

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """Predict one-step outputs for a batch of states."""
        if not self._trained:
            raise RuntimeError("Learner not fitted")
        states_arr = np.asarray(states, dtype=np.float64)
        if states_arr.ndim != 2:
            raise ValueError("states must be 2D")
        return self._apply(states_arr)

    def rollout(self, state: Sequence[float], steps: int) -> np.ndarray:
        """Run the operator forward for a fixed number of steps."""
        if steps < 1:
            raise ValueError("rollout steps must be >= 1")
        state_arr = np.asarray(state, dtype=np.float64)
        seq = [state_arr]
        current = state_arr
        for _ in range(steps):
            current = self.predict_next(current)
            seq.append(current)
        return np.stack(seq)

    def state_dict(self) -> Dict[str, Union[Sequence[float], float]]:
        """Return weights + metadata for serialization."""
        if not self._trained or self.W is None or self.b is None:
            raise RuntimeError("Learner not fitted")
        return {
            "alpha": self.alpha,
            "spectral_norm": self.spectral_norm,
            "raw_spectral_norm": self._raw_norm,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }

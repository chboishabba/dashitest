from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class QuotientSpec:
    w_fast: int = 64
    w_slow: int = 256
    eps: float = 1e-12
    version: str = "qfeat_v1"

class QuotientMap:
    """Deterministic map from raw closes to quotient representatives."""

    def __init__(self, spec: QuotientSpec | None = None) -> None:
        self.spec = spec or QuotientSpec()

    def compute(self, close: np.ndarray) -> np.ndarray:
        close = np.asarray(close, dtype=np.float32)
        if close.ndim == 1:
            return self._compute_1d(close)
        if close.ndim == 2:
            return np.stack([self._compute_1d(close[i]) for i in range(close.shape[0])], axis=0)
        raise ValueError(f"close must be 1d or 2d, got {close.shape}")

    def _compute_1d(self, close: np.ndarray) -> np.ndarray:
        # Placeholder for the actual qfeat oracle. Replace or wrap existing logic.
        # Ensure float32, deterministic ordering, and NaN/Inf handling.
        from features.quotient import compute_qfeat

        return compute_qfeat(close[None, :])[0]

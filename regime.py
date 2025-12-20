"""
Regime specification and checking for acceptable engagement.
This is PnL-free; it only encodes epistemic/mechanical permission to act.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RegimeSpec:
    min_run_length: int = 3       # minimum consecutive identical state to be considered stable
    max_flip_rate: float = None   # optional: maximum flips per bar in a recent window
    max_vol: float = None         # optional: max normalized volatility (user-defined scale)
    window: int = 50              # window for flip-rate/vol calculations


def sign_run_lengths(states: np.ndarray) -> np.ndarray:
    """Compute run-length of identical non-zero states ending at each index."""
    runs = np.zeros(len(states), dtype=int)
    run = 0
    for i, s in enumerate(states):
        if s == 0:
            run = 0
        else:
            if i > 0 and s == states[i - 1]:
                run += 1
            else:
                run = 1
        runs[i] = run
    return runs


def check_regime(states: np.ndarray, vols: np.ndarray, spec: RegimeSpec) -> np.ndarray:
    """
    Returns a boolean array acceptable_t indicating whether engagement is allowed at each t.
    Criteria (AND):
      - run_length >= min_run_length OR (max_flip_rate and flip_rate <= max_flip_rate)
      - if max_vol is set: vol <= max_vol
    """
    states = np.asarray(states, dtype=int)
    vols = np.asarray(vols, dtype=float)
    n = len(states)
    acceptable = np.zeros(n, dtype=bool)

    runs = sign_run_lengths(states)
    flips = np.abs(np.diff(states)) > 0
    flip_rate = np.zeros(n)
    window = max(1, spec.window)
    for i in range(n):
        lo = max(0, i - window + 1)
        flip_rate[i] = flips[lo:i].sum() / max(1, i - lo)

    stable = runs >= spec.min_run_length
    if spec.max_flip_rate is not None:
        stable = stable | (flip_rate <= spec.max_flip_rate)
    vol_ok = np.ones(n, dtype=bool)
    if spec.max_vol is not None:
        vol_ok = vols <= spec.max_vol
    acceptable = stable & vol_ok
    return acceptable

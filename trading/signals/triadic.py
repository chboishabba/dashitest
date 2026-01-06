from __future__ import annotations

import numpy as np


def compute_triadic_state(prices, dz_min=5e-5, window=200):
    """
    Reproduce the triadic state logic from run_trader:
    - rolling std over returns (window)
    - EWMA of volatility-normalized returns
    - dead-zone around zero
    Returns an array of ints in {-1,0,+1} with the same length as prices.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    state = np.zeros(n, dtype=int)
    z_prev = 0.0
    recent_rets = []
    for t in range(1, n):
        ret = prices[t] / prices[t - 1] - 1.0
        recent_rets.append(ret)
        if len(recent_rets) > window:
            recent_rets.pop(0)
        sigma = np.std(recent_rets) if recent_rets else dz_min
        z_update = ret / (sigma + 1e-9)
        z_update = np.clip(z_update, -5.0, 5.0)
        z = 0.95 * z_prev + 0.05 * z_update
        dz = max(dz_min, 0.5 * sigma)
        if abs(z) < dz:
            desired = 0
        elif z > 0:
            desired = 1
        else:
            desired = -1
        state[t] = desired
        z_prev = z
    return state

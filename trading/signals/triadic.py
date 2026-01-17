import numpy as np

# Ternary Sovereignty Constants
BUY = 1
SELL = -1
FLAT = 0
UNKNOWN = -2  # Epistemic Void (⊥) - Abstain/Hold
PARADOX = -3  # Systemic Collapse Risk (⚡) - Prohibit/Exit


def compute_triadic_state(prices, dz_min=5e-5, window=200):
    """
    Reproduce the triadic state logic from run_trader:
    - rolling std over returns (window)
    - EWMA of volatility-normalized returns
    - dead-zone around zero
    Returns an array of ints in {SELL, FLAT, BUY, UNKNOWN} with the same length as prices.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    state = np.full(n, UNKNOWN, dtype=int)
    if n == 0:
        return state
        
    state[0] = FLAT
    z_prev = 0.0
    recent_rets = []
    for t in range(1, n):
        if not np.isfinite(prices[t]) or not np.isfinite(prices[t-1]):
            state[t] = UNKNOWN
            continue
            
        ret = prices[t] / prices[t - 1] - 1.0
        recent_rets.append(ret)
        if len(recent_rets) > window:
            recent_rets.pop(0)
            
        sigma = np.std(recent_rets) if recent_rets else dz_min
        if not np.isfinite(sigma) or sigma < 1e-12:
            state[t] = UNKNOWN
            continue
            
        z_update = ret / (sigma + 1e-9)
        z_update = np.clip(z_update, -5.0, 5.0)
        z = 0.95 * z_prev + 0.05 * z_update
        dz = max(dz_min, 0.5 * sigma)
        
        if abs(z) < dz:
            desired = FLAT
        elif z > 0:
            desired = BUY
        else:
            desired = SELL
            
        state[t] = desired
        z_prev = z
    return state

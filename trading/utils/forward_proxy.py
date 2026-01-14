from __future__ import annotations

import numpy as np


def forward_return_proxy(
    prices: np.ndarray,
    *,
    horizon: int = 8,
    clip_return: float = 0.01,
    use_log_return: bool = True,
) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    n = prices.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    h = max(1, int(horizon))
    if n <= h:
        return out
    for t in range(n - h):
        if use_log_return:
            ret = np.log(prices[t + h] / prices[t])
        else:
            ret = (prices[t + h] - prices[t]) / prices[t]
        out[t] = float(np.clip(ret, -clip_return, clip_return))
    return out

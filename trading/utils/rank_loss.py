from __future__ import annotations

import numpy as np


def pairwise_rank_loss(
    scores: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, int]:
    n = scores.shape[0]
    grad = np.zeros_like(scores, dtype=np.float32)
    if n < 2:
        return 0.0, grad, 0

    idx = np.arange(n)
    rng.shuffle(idx)
    pairs = idx[: (n // 2) * 2].reshape(-1, 2)

    loss = np.float32(0.0)
    used = 0
    for i, j in pairs:
        y = float(np.sign(targets[i] - targets[j]))
        if y == 0.0 or not np.isfinite(y):
            continue
        margin = float(scores[i] - scores[j])
        z = np.float32(-y * margin)
        loss += np.log1p(np.exp(z))
        sig = np.float32(1.0 / (1.0 + np.exp(-z)))
        g = -y * sig
        grad[i] += g
        grad[j] -= g
        used += 1

    if used > 0:
        loss = loss / np.float32(used)
        grad = grad / np.float32(used)
    return float(loss), grad, used

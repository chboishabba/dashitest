"""
ternary_life_ca.py
------------------
Cyclic ternary CA (no HOLD): states {0,1,2}, rule:
    Each cell advances to the next state (s -> (s+1) mod 3)
    if at least k neighbors are already in that next state.
Otherwise it stays the same. Zero-padding boundaries.

This produces traveling waves/spirals (Rock-Paper-Scissors style).
"""

import numpy as np


def step(grid: np.ndarray, k: int = 2, wrap: bool = True) -> np.ndarray:
    """Single CA step; wrap=True uses periodic boundaries for more activity."""
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]
    out = grid.copy()
    H, W = grid.shape

    def shift(arr, di, dj):
        if wrap:
            return np.roll(np.roll(arr, di, axis=0), dj, axis=1)
        else:
            padded = np.zeros_like(arr)
            src_i = slice(max(0, di), H + min(0, di))
            src_j = slice(max(0, dj), W + min(0, dj))
            dst_i = slice(max(0, -di), H - max(0, di))
            dst_j = slice(max(0, -dj), W - max(0, dj))
            padded[dst_i, dst_j] = arr[src_i, src_j]
            return padded

    for s in (0, 1, 2):
        target = (s + 1) % 3
        mask_t = np.zeros_like(grid, dtype=np.int8)
        for di, dj in shifts:
            mask_t += (shift(grid, di, dj) == target).astype(np.int8)
        advance = (grid == s) & (mask_t >= k)
        out[advance] = target
    return out


def rollout(grid, steps=20, k=2, wrap=True):
    stats = []
    g = grid.copy()
    for t in range(steps):
        counts = (g == 0).sum(), (g == 1).sum(), (g == 2).sum()
        stats.append(counts)
        g = step(g, k=k, wrap=wrap)
    return np.array(stats, dtype=int)


def demo(H=64, W=64, steps=30, seed=0, k=2, wrap=True):
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 3, size=(H, W), dtype=np.int8)
    stats = rollout(grid, steps=steps, k=k, wrap=wrap)
    print("Cyclic ternary CA (0→1→2→0, no HOLD)")
    print(f"Grid: {H}x{W}, Steps: {steps}, k={k}, wrap={wrap}")
    for t, (c0, c1, c2) in enumerate(stats):
        total = H * W
        print(f"t={t:02d}: 0 {c0/total:6.2%}  1 {c1/total:6.2%}  2 {c2/total:6.2%}")


if __name__ == "__main__":
    demo()

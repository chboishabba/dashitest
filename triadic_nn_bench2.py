"""
triadic_nn_bench2.py
--------------------
Alternate ternary dense-layer benchmark comparing:
- Baseline NumPy (unpacked int8) dense
- SWAR packed dot_product_swar from swar_test_harness

Inputs/weights use 12-lane ternary vectors in {0,1,2}. No activation.
"""

import time
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

from swar_test_harness import (
    dot_product_swar,
    LANE_SHIFTS,
)


def pack_matrix(X: np.ndarray) -> np.ndarray:
    """Pack (N,12) uint8 trits into (N,) uint64 words."""
    return np.sum(X.astype(np.uint64) << LANE_SHIFTS, axis=1, dtype=np.uint64)


def dense_ref(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Baseline dense layer using unpacked int8 math."""
    return X.astype(np.int32) @ W.T.astype(np.int32)


def dense_swar(X_packed: np.ndarray, W_packed: np.ndarray) -> np.ndarray:
    """
    Dense layer using packed dot_product_swar (normal lanes only).
    """
    N = X_packed.shape[0]
    M = W_packed.shape[0]
    out = np.empty((N, M), dtype=np.int32)
    tmp = np.empty(N, dtype=np.int32)
    for j in range(M):
        wj = W_packed[j]
        dot_product_swar(X_packed, np.full(N, wj, dtype=np.uint64), tmp)
        out[:, j] = tmp
    return out


def bench_once(X, W, iters=5):
    # warmup
    dense_ref(X, W)
    Xp = pack_matrix(X)
    Wp = pack_matrix(W)
    dense_swar(Xp, Wp)

    # correctness
    ref = dense_ref(X, W)
    sw = dense_swar(Xp, Wp)
    if not np.array_equal(ref, sw):
        idx = np.nonzero(ref != sw)
        raise AssertionError(f"Mismatch at indices {idx}")

    t0 = time.perf_counter()
    for _ in range(iters):
        dense_ref(X, W)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(iters):
        dense_swar(Xp, Wp)
    t3 = time.perf_counter()

    return (t1 - t0) / iters, (t3 - t2) / iters


def run_bench(N=100_000, M=16, seed=0, iters=5):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(N, 12), dtype=np.uint8)
    W = rng.integers(0, 3, size=(M, 12), dtype=np.uint8)

    dt_ref, dt_swar = bench_once(X, W, iters=iters)
    ops = N * M * 12
    print(f"N={N:>9}, M={M:>3}: baseline {dt_ref*1e6:8.2f} µs/call ({ops/dt_ref/1e6:8.2f} Mop/s) "
          f"SWAR {dt_swar*1e6:8.2f} µs/call ({ops/dt_swar/1e6:8.2f} Mop/s) "
          f"speedup x{dt_ref/dt_swar:5.2f}")


def main():
    print("triadic_nn_bench2: baseline NumPy vs packed SWAR dot_product_swar")
    for N, M in [(1_000, 8), (100_000, 16)]:
        iters = 20 if N == 1_000 else 5
        run_bench(N=N, M=M, seed=123 + N, iters=iters)


if __name__ == "__main__":
    main()

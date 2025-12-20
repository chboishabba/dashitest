"""
triadic_nn_bench.py
-------------------
Compare a simple ternary (trit) dense layer using:
- Baseline NumPy int8 vectors (unpacked)
- Packed SWAR dot_product_swar from swar_test_harness

Both use the same semantics: inputs/weights in {0,1,2}, single linear layer (no activation).
"""

import sys
import time
import numpy as np

from swar_test_harness import (
    HAVE_NUMBA,
    LANE_SHIFTS,
    dot_product_ref,
    dot_product_swar,
)


def pack_trits(lanes: np.ndarray) -> np.ndarray:
    """Pack shape (N,12) uint8 lanes into shape (N,) uint64 words."""
    return np.sum(lanes.astype(np.uint64) << LANE_SHIFTS, axis=1, dtype=np.uint64)


def gen_data(batch: int, neurons: int, seed: int = 0):
    """Generate ternary inputs and weights."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(batch, 12), dtype=np.uint8)
    W = rng.integers(0, 3, size=(neurons, 12), dtype=np.uint8)
    return X, W


def baseline_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Baseline dense layer using unpacked int8 math."""
    return (X.astype(np.int32) @ W.T.astype(np.int32))


def swar_forward(X_packed: np.ndarray, W_packed: np.ndarray) -> np.ndarray:
    """SWAR dense layer using dot_product_swar per neuron."""
    batch = X_packed.shape[0]
    neurons = W_packed.shape[0]
    out = np.empty((batch, neurons), dtype=np.int32)
    tmp = np.empty(batch, dtype=np.int32)
    for j in range(neurons):
        # broadcast weight word
        wp = W_packed[j]
        dot_product_swar(X_packed, np.full(batch, wp, dtype=np.uint64), tmp)
        out[:, j] = tmp
    return out


def bench(fn, *args, iters: int):
    # warmup
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    if not HAVE_NUMBA:
        from swar_test_harness import NB_IMPORT_ERROR  # type: ignore
        print("Numba not available:", NB_IMPORT_ERROR)
        sys.exit(1)

    batch_sizes = (1_000, 100_000)
    neurons = 8
    print("Triadic NN bench: baseline (unpacked int8) vs SWAR packed dot_product")
    print(f"Neurons: {neurons}, input lanes: 12, values in {{0,1,2}}")

    for N in batch_sizes:
        X, W = gen_data(N, neurons, seed=123 + N)

        baseline_out = baseline_forward(X, W)

        X_p = pack_trits(X)
        W_p = pack_trits(W)

        swar_out = swar_forward(X_p, W_p)

        if not np.array_equal(baseline_out, swar_out):
            idx = np.nonzero(baseline_out != swar_out)[0][0]
            raise AssertionError(
                f"Mismatch at sample {idx}: baseline={baseline_out[idx]} swar={swar_out[idx]}"
            )

        iters = 20 if N == 1_000 else 5
        t_base = bench(baseline_forward, X, W, iters=iters)
        t_swar = bench(swar_forward, X_p, W_p, iters=iters)

        # ops per call: batch * neurons * 12 multiplies
        ops = N * neurons * 12
        print(
            f"N={N:>9}: baseline {t_base*1e6:8.2f} µs/call ({ops/t_base/1e6:8.2f} Mop/s)  "
            f"SWAR {t_swar*1e6:8.2f} µs/call ({ops/t_swar/1e6:8.2f} Mop/s)  "
            f"speedup x{t_base/t_swar:5.1f}"
        )


if __name__ == "__main__":
    main()

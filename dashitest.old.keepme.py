"""
Legacy dashitest snapshot, now wired to the validated harness kernels.
Runs a quick correctness check and benchmark using swar_test_harness.
"""

import numpy as np
import time

from swar_test_harness import (
    C_XOR_ref,
    C_XOR_array_swar,
    random_words,
)


def bench(fn, A, B, out, flags, iters=20):
    fn(A, B, out, flags)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(A, B, out, flags)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    print("dashitest.old.keepme: harness-backed C_XOR benchmark")
    N = 100_000
    A = random_words(N, p_special=0.01, seed=1)
    B = random_words(N, p_special=0.01, seed=2)

    ref_w, ref_f = C_XOR_ref(A, B)
    out = np.empty_like(A)
    flags = np.empty(N, dtype=np.uint8)
    C_XOR_array_swar(A, B, out, flags)
    assert np.array_equal(ref_w, out) and np.array_equal(ref_f, flags)
    print("Correctness OK.")

    t = bench(C_XOR_array_swar, A, B, out, flags, iters=20)
    print(f"N={N}: {t*1e6:8.2f} µs/call  {N/t/1e6:8.2f} Mwords/s")


if __name__ == "__main__":
    main()

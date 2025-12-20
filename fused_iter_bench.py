"""
fused_iter_bench.py
-------------------
Cache-resident iterative workload to expose ALU throughput:
state = XOR(state, inp) -> threshold -> accumulate (dot) for many iterations.

Uses validated harness primitives on packed 12-lane words (no specials in this demo).
"""

import time
import numpy as np
from swar_test_harness import (
    C_XOR_array_swar,
    dot_product_swar,
    threshold_count_swar,
    random_words,
)


def fused_step(state, inp, thresh, tmp_flags, tmp_word, tmp_counts):
    """
    Perform one fused step over the whole array:
    state = XOR(state, inp)
    counts = threshold(state, thresh)
    acc = dot(state, state)  # self-dot as a cheap accumulate surrogate
    Returns updated state and accumulated counts/dots.
    """
    # XOR
    C_XOR_array_swar(state, inp, tmp_word, tmp_flags)
    state[:] = tmp_word

    # threshold counts
    threshold_count_swar(state, thresh, tmp_counts)

    # dot (self-dot)
    dot_product_swar(state, state, tmp_counts)  # reuse tmp_counts as int32 buffer

    return state


def bench_fused(N=1024, iters=256, thresh=10, seed=0):
    state = random_words(N, p_special=0.0, seed=seed)
    inp = random_words(N, p_special=0.0, seed=seed + 1)

    tmp_flags = np.empty(N, dtype=np.uint8)
    tmp_word = np.empty_like(state)
    tmp_counts = np.empty(N, dtype=np.int32)

    # warmup
    fused_step(state, inp, thresh, tmp_flags, tmp_word, tmp_counts)

    t0 = time.perf_counter()
    for _ in range(iters):
        fused_step(state, inp, thresh, tmp_flags, tmp_word, tmp_counts)
    t1 = time.perf_counter()

    dt = (t1 - t0) / iters
    # effective trit operations ~ (XOR + threshold + dot) per word
    ops = N * 12 * 3
    print(f"Fused iter bench: N={N}, iters={iters}, {dt*1e6:8.2f} µs/iter, {ops/dt/1e6:8.2f} Mop/s")


def main():
    print("fused_iter_bench: XOR -> threshold -> dot loop (cache-resident)")
    bench_fused(N=1024, iters=256, thresh=10, seed=123)
    bench_fused(N=1024, iters=1024, thresh=10, seed=123)


if __name__ == "__main__":
    main()

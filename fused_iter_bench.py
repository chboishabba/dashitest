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
    extract_lanes,
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


def fused_step_baseline(lanes_state, lanes_inp, thresh):
    """
    Unpacked baseline on lanes (N,12) uint8.
    """
    lanes_state = (lanes_state + lanes_inp) % 3
    counts = (lanes_state > thresh).sum(axis=1)
    _ = counts  # unused
    _ = np.sum(lanes_state * lanes_state, axis=1)
    return lanes_state


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
    # baseline unpacked
    lanes_state = extract_lanes(state)
    lanes_inp = extract_lanes(inp)
    t0b = time.perf_counter()
    ls = lanes_state
    for _ in range(iters):
        ls = fused_step_baseline(ls, lanes_inp, thresh)
    t1b = time.perf_counter()
    dt_base = (t1b - t0b) / iters
    print(f"Fused iter bench: N={N}, iters={iters}, baseline {dt_base*1e6:8.2f} µs/iter, "
          f"SWAR {dt*1e6:8.2f} µs/iter, speedup x{dt_base/dt:5.2f}, "
          f"ops {ops/dt/1e6:8.2f} Mop/s")


def main():
    print("fused_iter_bench: XOR -> threshold -> dot loop (cache-resident)")
    bench_fused(N=1024, iters=256, thresh=10, seed=123)
    bench_fused(N=1024, iters=1024, thresh=10, seed=123)


if __name__ == "__main__":
    main()

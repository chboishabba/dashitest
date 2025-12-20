"""
potts3_bench.py
----------------
Simple 1D 3-state lattice (Potts-like) update:
  new = (center + left + right) % 3
States are packed into 12-lane words (0..2 only, no specials).
Compare baseline (unpacked) vs SWAR using harness C_XOR_array_swar composed twice.
"""

import time
import numpy as np
from swar_test_harness import (
    random_words,
    extract_lanes,
    C_XOR_array_swar,
)


def baseline_step(words):
    lanes = extract_lanes(words)
    left = np.roll(lanes, 1, axis=0)
    right = np.roll(lanes, -1, axis=0)
    new_lanes = (lanes + left + right) % 3
    return np.sum(new_lanes.astype(np.uint64) << np.array([5*i for i in range(12)], dtype=np.uint64),
                  axis=1, dtype=np.uint64)


def swar_step(words):
    # new = (center + left + right) mod3 == C_XOR(C_XOR(center,left), right) with no specials
    tmp = np.empty_like(words)
    flags = np.empty(words.shape[0], dtype=np.uint8)
    left = np.roll(words, 1)
    right = np.roll(words, -1)
    C_XOR_array_swar(words, left, tmp, flags)
    out = np.empty_like(words)
    C_XOR_array_swar(tmp, right, out, flags)
    return out


def bench(fn, state, iters, label):
    # warmup
    fn(state)
    t0 = time.perf_counter()
    for _ in range(iters):
        state = fn(state)
    t1 = time.perf_counter()
    return (t1 - t0) / iters, state


def main():
    N = 4096
    iters = 256
    state = random_words(N, p_special=0.0, seed=1) % 3  # ensure 0..2

    t_base, final_base = bench(baseline_step, state.copy(), iters, "baseline")
    t_swar, final_swar = bench(swar_step, state.copy(), iters, "swar")

    # validate
    if not np.array_equal(final_base, final_swar):
        raise AssertionError("Potts benchmark diverged between baseline and SWAR")

    print("Potts/3-state 1D lattice update (center+left+right mod3):")
    print(f"N={N}, iters={iters}: baseline {t_base*1e3:8.2f} ms/iter   SWAR {t_swar*1e3:8.2f} ms/iter   speedup x{t_base/t_swar:5.2f}")


if __name__ == "__main__":
    main()

"""
gf3_parity_bench.py
-------------------
GF(3) parity (sum mod 3) over packed lanes.
Compare baseline (unpacked sum) vs SWAR using dot_product_swar with an all-ones word.
"""

import time
import numpy as np
from swar_test_harness import (
    extract_lanes,
    dot_product_swar,
    random_words,
)

# precompute all-ones word (lanes=1)
ONES_WORD = np.sum(
    (np.ones(12, dtype=np.uint64) << np.array([5*i for i in range(12)], dtype=np.uint64)),
    dtype=np.uint64
)


def baseline_parity(words):
    lanes = extract_lanes(words)
    return (lanes.sum(axis=1) % 3).astype(np.uint8)


def swar_parity(words):
    out = np.empty(words.shape[0], dtype=np.int32)
    dot_product_swar(words, np.full(words.shape[0], ONES_WORD, dtype=np.uint64), out)
    return (out % 3).astype(np.uint8)


def bench(fn, words, iters=10):
    fn(words)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(words)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 100_000
    words = random_words(N, p_special=0.0, seed=5) % 3  # ensure 0..2
    b = baseline_parity(words)
    s = swar_parity(words)
    if not np.array_equal(b, s):
        raise AssertionError("GF(3) parity mismatch between baseline and SWAR")
    t_base = bench(baseline_parity, words, iters=5)
    t_swar = bench(swar_parity, words, iters=5)
    print("GF(3) parity (sum mod 3 over lanes):")
    print(f"N={N}: baseline {t_base*1e3:8.2f} ms/call   SWAR {t_swar*1e3:8.2f} ms/call   speedup x{t_base/t_swar:5.2f}")


if __name__ == "__main__":
    main()

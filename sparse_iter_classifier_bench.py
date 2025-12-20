"""
sparse_iter_classifier_bench.py
-------------------------------
Sparse-ish iterative classifier loop:
    for k in range(K):
        state ^= input
        counts = threshold(state)
        score  = dot(state, state)
        pred   = counts > decision

Benchmarks:
- FP32 baseline on unpacked lanes
- SWAR packed path using harness kernels

This stresses XOR->threshold->dot iteratively on cache-resident data.
"""

import time
import numpy as np

from swar_test_harness import (
    C_XOR_array_swar,
    threshold_count_swar,
    dot_product_swar,
    extract_lanes,
)


def random_sparse_words(N, density=0.1, seed=0):
    """
    Generate packed words with ternary lanes mostly zero, with occasional 1/2.
    """
    rng = np.random.default_rng(seed)
    lanes = np.zeros((N, 12), dtype=np.uint8)
    mask = rng.random((N, 12)) < density
    lanes = np.where(mask, rng.integers(1, 3, size=(N, 12), dtype=np.uint8), lanes)
    words = np.sum(lanes.astype(np.uint64) << np.array([5*i for i in range(12)], dtype=np.uint64),
                   axis=1, dtype=np.uint64)
    return words


def baseline_epoch(state_words, inp_words, K=128, thresh=0, decision=2):
    """
    Unpacked FP32 baseline.
    """
    state = extract_lanes(state_words).astype(np.float32)
    inp = extract_lanes(inp_words).astype(np.float32)
    for _ in range(K):
        state = (state + inp) % 3  # xor-like mod3
        counts = (state > thresh).sum(axis=1)
        preds = counts > decision
        # simple accumulation to simulate work
        score = np.sum(state * state, axis=1)
        _ = np.sum(score * preds)  # prevent dead code
    # repack to match interface (not used further)
    return state


def swar_epoch(state_words, inp_words, K=128, thresh=0, decision=2):
    """
    Packed SWAR path using harness kernels.
    """
    state = state_words.copy()
    inp = inp_words
    tmp_flags = np.empty(state.shape[0], dtype=np.uint8)
    tmp_word = np.empty_like(state)
    counts = np.empty(state.shape[0], dtype=np.uint32)
    dots = np.empty(state.shape[0], dtype=np.int32)
    for _ in range(K):
        C_XOR_array_swar(state, inp, tmp_word, tmp_flags)
        state[:] = tmp_word
        threshold_count_swar(state, thresh, counts)
        preds = counts > decision
        dot_product_swar(state, state, dots)
        _ = int(np.sum(dots * preds))  # prevent dead code
    return state


def bench(fn, *args, iters=5):
    # warmup
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 2048
    K_short = 128
    K_long = 512
    thresh = 0
    decision = 2

    state = random_sparse_words(N, density=0.1, seed=1)
    inp = random_sparse_words(N, density=0.1, seed=2)

    print("Sparse iterative classifier loop (XOR -> threshold -> dot) on cache-resident data.")
    for K in (K_short, K_long):
        t_base = bench(baseline_epoch, state, inp, K, thresh, decision, iters=3 if K > 128 else 5)
        t_swar = bench(swar_epoch, state, inp, K, thresh, decision, iters=3 if K > 128 else 5)
        print(f"K={K:4d}: baseline {t_base*1e3:8.2f} ms/epoch   SWAR {t_swar*1e3:8.2f} ms/epoch   speedup x{t_base/t_swar:5.2f}")


if __name__ == "__main__":
    main()

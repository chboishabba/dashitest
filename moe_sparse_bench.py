"""
moe_sparse_bench.py
-------------------
Toy sparse/MoE-style CPU benchmark with ternary packed inputs/experts.

Flow per token:
  - gate: count lanes > gate_thresh
  - route: expert_idx = count % num_experts (simple sparse routing)
  - compute: dot(token, expert[expert_idx]) on packed ternary lanes
  - update: write per-token score (no global atomics)

Compares:
  - FP32 baseline on unpacked lanes
  - Fused Numba SWAR kernel on packed words
"""

import time
import numpy as np
import numba as nb
from swar_test_harness import extract_lanes, random_words


def random_sparse_words(N, density=0.1, seed=0):
    rng = np.random.default_rng(seed)
    lanes = np.zeros((N, 12), dtype=np.uint8)
    mask = rng.random((N, 12)) < density
    lanes = np.where(mask, rng.integers(1, 3, size=(N, 12), dtype=np.uint8), lanes)
    words = np.sum(lanes.astype(np.uint64) << np.array([5*i for i in range(12)], dtype=np.uint64),
                   axis=1, dtype=np.uint64)
    return words


def baseline_epoch(tokens, experts, gate_thresh, iters=64):
    """
    FP32 baseline: unpacked lanes, simple routing and dot.
    """
    X = extract_lanes(tokens).astype(np.float32)
    W = extract_lanes(experts).astype(np.float32)
    num_experts = W.shape[0]
    scores = np.empty(tokens.shape[0], dtype=np.float32)
    for _ in range(iters):
        counts = (X > gate_thresh).sum(axis=1)
        idx = counts % num_experts
        for i in range(X.shape[0]):
            scores[i] = float(np.dot(X[i], W[idx[i]]))
    return scores


@nb.njit(inline="always")
def lane_value(v):
    # lane is 0..26 normal
    return v


@nb.njit(parallel=True, fastmath=True)
def swar_epoch(tokens, experts, gate_thresh, iters):
    """
    Packed fused path: gate + route + dot per token.
    """
    N = tokens.shape[0]
    M = experts.shape[0]
    scores = np.empty(N, dtype=np.int32)
    for _ in range(iters):
        for i in nb.prange(N):
            x = tokens[i]
            # gate: count lanes > gate_thresh
            cnt = 0
            for lane in range(12):
                sh = np.uint64(5*lane)
                v = (x >> sh) & np.uint64(0x1F)
                if v < np.uint64(27) and v > np.uint64(gate_thresh):
                    cnt += 1
            idx = cnt % M
            w = experts[idx]
            acc = 0
            for lane in range(12):
                sh = np.uint64(5*lane)
                a = (x >> sh) & np.uint64(0x1F)
                b = (w >> sh) & np.uint64(0x1F)
                acc += int(a) * int(b)
            scores[i] = acc
    return scores


def bench(fn, *args, iters=3):
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 4096        # tokens
    M = 8           # experts
    iters = 128     # iterations per epoch
    gate_thresh = 0

    tokens = random_sparse_words(N, density=0.1, seed=1)
    experts = random_sparse_words(M, density=0.5, seed=2)

    print("MoE-style sparse ternary benchmark (gate + route + dot) on CPU.")
    t_base = bench(baseline_epoch, tokens, experts, gate_thresh, iters)
    t_swar = bench(swar_epoch, tokens, experts, gate_thresh, iters)
    print(f"N={N}, M={M}, iters={iters}: baseline {t_base*1e3:8.2f} ms/epoch   SWAR {t_swar*1e3:8.2f} ms/epoch   speedup x{t_base/t_swar:5.2f}")


if __name__ == "__main__":
    main()

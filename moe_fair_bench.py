"""
moe_fair_bench.py
-----------------
A "fair NN" style ternary MoE benchmark that keeps work masked/fused:
  - ternary gate (bitwise count > threshold)
  - route to two experts (top-2 surrogate)
  - run several expert steps in-place (no per-step emission)
  - emit once at the end of the block

Compares:
  - FP32 baseline on unpacked lanes
  - Fused Numba SWAR kernel on packed words
"""

import time
import numpy as np
import numba as nb


LANES = 12
LANE_SHIFTS = (np.uint64(1) << np.arange(0, 5 * LANES, 5, dtype=np.uint64))


def pack_words(vals: np.ndarray) -> np.ndarray:
    """vals: (N, LANES) in {0,1,2} -> packed uint64 words."""
    return (vals.astype(np.uint64) * LANE_SHIFTS).sum(axis=1, dtype=np.uint64)


def random_ternary_words(N, density=1.0, seed=0):
    rng = np.random.default_rng(seed)
    lanes = rng.integers(0, 3, size=(N, LANES), dtype=np.uint8)
    if density < 1.0:
        mask = rng.random((N, LANES)) < density
        lanes = np.where(mask, lanes, 0)
    return pack_words(lanes)


def baseline_epoch(tokens, experts, gate_thresh, steps, iters):
    X = np.unpackbits(tokens.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    X = X[:, : 5 * LANES].reshape(-1, LANES, 5)
    # decode 5-bit lanes
    powers = (1 << np.arange(5, dtype=np.uint8))
    X = (X * powers).sum(axis=2).astype(np.float32)
    W = np.unpackbits(experts.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    W = W[:, : 5 * LANES].reshape(-1, LANES, 5)
    W = (W * powers).sum(axis=2).astype(np.float32)

    N = X.shape[0]
    M = W.shape[0]
    scores = np.zeros(N, dtype=np.float32)
    for _ in range(iters):
        counts = (X > gate_thresh).sum(axis=1).astype(np.int32)
        idx0 = counts % M
        idx1 = (idx0 + 1) % M
        for i in range(N):
            acc = 0.0
            for _ in range(steps):
                acc += float(np.dot(X[i], W[idx0[i]]))
                acc += float(np.dot(X[i], W[idx1[i]]))
            scores[i] = acc
    return scores


@nb.njit(inline="always")
def dot_packed(a, b):
    acc = 0
    for lane in range(LANES):
        sh = np.uint64(5 * lane)
        va = (a >> sh) & np.uint64(0x1F)
        vb = (b >> sh) & np.uint64(0x1F)
        acc += int(va) * int(vb)
    return acc


@nb.njit(parallel=True, fastmath=True)
def swar_epoch(tokens, experts, gate_thresh, steps, iters):
    N = tokens.shape[0]
    M = experts.shape[0]
    scores = np.zeros(N, dtype=np.int32)
    for _ in range(iters):
        for i in nb.prange(N):
            x = tokens[i]
            cnt = 0
            for lane in range(LANES):
                sh = np.uint64(5 * lane)
                v = (x >> sh) & np.uint64(0x1F)
                if v < np.uint64(27) and v > np.uint64(gate_thresh):
                    cnt += 1
            idx0 = cnt % M
            idx1 = (idx0 + 1) % M
            acc = 0
            for _ in range(steps):
                acc += dot_packed(x, experts[idx0])
                acc += dot_packed(x, experts[idx1])
            scores[i] = acc
    return scores


def bench(fn, *args, iters=3):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 4096        # tokens
    M = 8           # experts
    steps = 4       # expert steps per routing
    iters = 128     # iterations per epoch
    gate_thresh = 0

    tokens = random_ternary_words(N, density=1.0, seed=1)
    experts = random_ternary_words(M, density=1.0, seed=2)

    print("Fair NN-style ternary MoE benchmark (fused, masked, emit once).")
    t_base = bench(baseline_epoch, tokens, experts, gate_thresh, steps, iters)
    t_swar = bench(swar_epoch, tokens, experts, gate_thresh, steps, iters)
    print(f"N={N}, M={M}, steps={steps}, iters={iters}:")
    print(f"baseline {t_base*1e3:8.2f} ms/epoch   SWAR {t_swar*1e3:8.2f} ms/epoch   speedup x{t_base/t_swar:5.2f}")


if __name__ == "__main__":
    main()

"""
cluster_cull_pqn_bench.py
-------------------------
Order-ternary cluster culling using vertical P/Q/N planes (one-hot):
  P: state == 2
  Q: state == 1
  N: state == 0

Baseline: numpy boolean masks
PQN: bitset ops over uint64 blocks
"""

import time
import numpy as np


def make_states(N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=N, dtype=np.uint8)


def baseline_cull(states, thresh=1):
    # keep states >= thresh
    return states >= thresh


def pqn_build(states):
    N = states.size
    blocks = (N + 63) // 64
    P = np.zeros(blocks, dtype=np.uint64)
    Q = np.zeros(blocks, dtype=np.uint64)
    Np = np.zeros(blocks, dtype=np.uint64)
    idx = np.arange(N, dtype=np.int64)
    blk = idx >> 6
    off = idx & 63
    shifts = np.left_shift(np.uint64(1), off.astype(np.uint64))
    selP = states == 2
    selQ = states == 1
    selN = states == 0
    if np.any(selP):
        np.bitwise_or.at(P, blk[selP], shifts[selP])
    if np.any(selQ):
        np.bitwise_or.at(Q, blk[selQ], shifts[selQ])
    if np.any(selN):
        np.bitwise_or.at(Np, blk[selN], shifts[selN])
    valid = np.full(blocks, np.uint64(~np.uint64(0)), dtype=np.uint64)
    rem = N & 63
    if rem:
        valid[-1] = (np.uint64(1) << rem) - np.uint64(1)
    return P, Q, Np, valid, N


def pqn_cull(P, Q, Np, valid, thresh=1):
    # states >= thresh:
    # thresh=1: Q or P
    # thresh=2: P only
    if thresh <= 1:
        mask = (P | Q) & valid
    else:
        mask = P & valid
    return mask


def mask_to_bool(mask, total):
    out = np.zeros(total, dtype=bool)
    for i, v in enumerate(mask):
        base = i * 64
        val = int(v)
        while val:
            lsb = val & -val
            pos = base + (lsb.bit_length() - 1)
            if pos < total:
                out[pos] = True
            val ^= lsb
    return out


def bench(fn, *args, reps=5):
    fn(*args)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        dur = (t1 - t0) * 1e3
        best = dur if best is None or dur < best else best
    return best


def main():
    N = 1_000_000
    thresh = 1
    states = make_states(N, seed=0)
    P, Q, Np, valid, total = pqn_build(states)

    # correctness
    ref = baseline_cull(states, thresh)
    mask = pqn_cull(P, Q, Np, valid, thresh)
    pqn_bool = mask_to_bool(mask, total)
    assert np.array_equal(ref, pqn_bool)

    t_base = bench(baseline_cull, states, thresh)
    t_pqn = bench(lambda: pqn_cull(P, Q, Np, valid, thresh))

    print("Cluster cull (order ternary) using P/Q/N bitplanes")
    print(f"N={N}, thresh={thresh}")
    print(f"baseline: {t_base:8.2f} ms/call")
    print(f"PQN     : {t_pqn:8.2f} ms/call   speedup x{t_base/t_pqn:5.2f}")


if __name__ == "__main__":
    main()

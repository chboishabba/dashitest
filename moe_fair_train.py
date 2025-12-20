"""
moe_fair_train.py
-----------------
Tiny ternary MoE training loop to illustrate model-level usage of the
packed SWAR kernels. We generate synthetic data from a hidden ternary
MoE and train a simple model with learnable expert weights. Gate is
count-based; experts are trained with SGD on MSE.

This is intentionally minimal and CPU-only.
"""

import time
import numpy as np
import numba as nb

LANES = 12
LANE_SHIFTS = (np.uint64(1) << np.arange(0, 5 * LANES, 5, dtype=np.uint64))


def pack_words(vals: np.ndarray) -> np.ndarray:
    return (vals.astype(np.uint64) * LANE_SHIFTS).sum(axis=1, dtype=np.uint64)


def random_ternary_words(N, density=1.0, seed=0):
    rng = np.random.default_rng(seed)
    lanes = rng.integers(0, 3, size=(N, LANES), dtype=np.uint8)
    if density < 1.0:
        mask = rng.random((N, LANES)) < density
        lanes = np.where(mask, lanes, 0)
    return pack_words(lanes), lanes  # packed and unpacked views


def synth_data(N=4096, M=8, seed=0):
    tokens_packed, tokens_unpacked = random_ternary_words(N, density=1.0, seed=seed)
    rng = np.random.default_rng(seed + 1)
    true_experts = rng.normal(size=(M, LANES)).astype(np.float32)
    # gate: count lanes > 0, route to idx = count % M
    counts = (tokens_unpacked > 0).sum(axis=1)
    idx = counts % M
    targets = np.array([np.dot(tokens_unpacked[i].astype(np.float32), true_experts[idx[i]]) for i in range(N)], dtype=np.float32)
    return tokens_packed, tokens_unpacked, targets, true_experts


@nb.njit(inline="always")
def dot_packed_float(x_packed, w):
    acc = 0.0
    for lane in range(LANES):
        sh = np.uint64(5 * lane)
        v = (x_packed >> sh) & np.uint64(0x1F)
        acc += float(v) * w[lane]
    return acc


@nb.njit(parallel=True, fastmath=True)
def swar_forward(tokens, experts):
    N = tokens.shape[0]
    M = experts.shape[0]
    out = np.empty(N, dtype=np.float32)
    for i in nb.prange(N):
        x = tokens[i]
        cnt = 0
        for lane in range(LANES):
            sh = np.uint64(5 * lane)
            v = (x >> sh) & np.uint64(0x1F)
            if v > 0:
                cnt += 1
        idx = cnt % M
        out[i] = dot_packed_float(x, experts[idx])
    return out


def train_epoch(tokens_packed, tokens_unpacked, targets, experts, lr=1e-2):
    # forward (packed, SWAR)
    preds = swar_forward(tokens_packed, experts)
    err = preds - targets
    loss = float((err ** 2).mean())
    # backward: histogram-based aggregation per expert per lane
    M = experts.shape[0]
    grad = np.zeros_like(experts)
    counts = (tokens_unpacked > 0).sum(axis=1)
    idx = counts % M
    err_per_exp = np.zeros(M, dtype=np.float32)
    for i in range(tokens_unpacked.shape[0]):
        err_per_exp[idx[i]] += err[i]
        # accumulate per-lane counts weighted by error
        for lane in range(LANES):
            v = tokens_unpacked[i, lane]
            if v == 1:
                grad[idx[i], lane] += err[i]
            elif v == 2:
                grad[idx[i], lane] += 2 * err[i]
    grad /= tokens_unpacked.shape[0]
    experts -= lr * grad
    return loss


def main():
    N = 4096
    M = 8
    epochs = 1000
    tokens_packed, tokens_unpacked, targets, true_experts = synth_data(N, M, seed=0)
    experts = np.zeros((M, LANES), dtype=np.float32)

    print("Ternary MoE training demo (packed forward, SGD experts)")
    for e in range(epochs):
        t0 = time.perf_counter()
        loss = train_epoch(tokens_packed, tokens_unpacked, targets, experts, lr=5e-3)
        t1 = time.perf_counter()
        print(f"epoch {e+1:2d}: loss={loss:8.4f}  time={(t1-t0)*1e3:6.2f} ms")


if __name__ == "__main__":
    main()

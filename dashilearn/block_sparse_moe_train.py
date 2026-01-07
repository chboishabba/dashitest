"""
block_sparse_moe_train.py
-------------------------
Block-sparse MoE training demo:
  - Gate selects active tiles (block-level)
  - Tile masks are derived from per-output gate activity
  - Dense int8 matmul microkernel runs only on active tiles
  - Emit once per block
"""

import time
import numpy as np


def gate_prob_for_tile_density(tile_active, tile):
    if tile_active <= 0.0:
        return 0.0
    if tile_active >= 1.0:
        return 1.0
    return 1.0 - (1.0 - tile_active) ** (1.0 / (tile * tile))


def tiles_from_gate_mask(gate_mask, tile=32):
    M, N = gate_mask.shape
    tiles = np.zeros(((M + tile - 1) // tile, (N + tile - 1) // tile), dtype=bool)
    for ti in range(tiles.shape[0]):
        for tj in range(tiles.shape[1]):
            i0, j0 = ti * tile, tj * tile
            i1, j1 = min(i0 + tile, M), min(j0 + tile, N)
            tiles[ti, tj] = np.any(gate_mask[i0:i1, j0:j1])
    return tiles


def make_data(M=256, K=256, N=256, tiles_active=0.5, tile=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(M, K), dtype=np.int8)
    W = rng.integers(0, 3, size=(K, N), dtype=np.int8)
    # gate mask -> active tiles (tile-level any-activation)
    gate_prob = gate_prob_for_tile_density(tiles_active, tile)
    gate_mask = rng.random((M, N)) < gate_prob
    tiles = tiles_from_gate_mask(gate_mask, tile=tile)
    return X, W, tiles


def dense_matmul(X, W):
    return X.astype(np.int32) @ W.astype(np.int32)


def block_sparse_matmul(X, W, tiles, tile=32):
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.int32)
    for ti in range(tiles.shape[0]):
        for tj in range(tiles.shape[1]):
            if not tiles[ti, tj]:
                continue
            i0, j0 = ti * tile, tj * tile
            i1, j1 = min(i0 + tile, M), min(j0 + tile, N)
            for k0 in range(0, K, tile):
                k1 = min(k0 + tile, K)
                Ablk = X[i0:i1, k0:k1].astype(np.int32)
                Bblk = W[k0:k1, j0:j1].astype(np.int32)
                C[i0:i1, j0:j1] += Ablk @ Bblk
    return C


def train_epoch(X, W, tiles, lr=1e-3, tile=32):
    # forward block-sparse
    C = block_sparse_matmul(X, W, tiles, tile=tile)
    # fake target: zeros
    err = C
    # backward: simple gradient on W for active tiles
    gradW = np.zeros_like(W, dtype=np.int32)
    for ti in range(tiles.shape[0]):
        for tj in range(tiles.shape[1]):
            if not tiles[ti, tj]:
                continue
            i0, j0 = ti * tile, tj * tile
            i1, j1 = min(i0 + tile, X.shape[0]), min(j0 + tile, W.shape[1])
            # X_blk: (tile, K) -> transpose -> (K, tile)
            X_blk = X[i0:i1, :].astype(np.int32)  # (tile, K)
            err_blk = err[i0:i1, j0:j1]           # (tile, tile)
            gradW[:, j0:j1] += X_blk.T @ err_blk
    W = W - lr * gradW.astype(W.dtype)
    loss = float((err ** 2).mean())
    return W, loss


def bench(fn, *args, reps=3, **kwargs):
    fn(*args, **kwargs)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        dur = (t1 - t0) * 1e3
        best = dur if best is None or dur < best else best
    return best


def main():
    M = N = K = 256
    tile_active = 0.5
    tile = 32
    epochs = 3
    X, W, tiles = make_data(M, K, N, tile_active, tile=tile)

    # Baseline dense timing
    t_dense = bench(dense_matmul, X, W)

    # Block-sparse timing
    t_bs = bench(block_sparse_matmul, X, W, tiles, tile=tile)

    active_frac = float(tiles.mean()) if tiles.size else 0.0
    print("Block-sparse MoE-style matmul")
    print(f"M=N=K={M}, active tiles ~{active_frac*100:.1f}% (target {tile_active*100:.1f}%)")
    print(f"dense matmul      : {t_dense:6.2f} ms/call")
    print(f"block-sparse matmul: {t_bs:6.2f} ms/call   speedup x{t_dense/t_bs:5.2f}")

    # Tiny training loop (for illustration)
    for e in range(epochs):
        t0 = time.perf_counter()
        W, loss = train_epoch(X, W, tiles, lr=1e-5, tile=tile)
        t1 = time.perf_counter()
        print(f"epoch {e+1}: loss={loss:8.2e}  time={(t1-t0)*1e3:6.2f} ms")


if __name__ == "__main__":
    main()

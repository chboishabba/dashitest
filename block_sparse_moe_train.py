"""
block_sparse_moe_train.py
-------------------------
Block-sparse MoE training demo:
  - Gate selects active tiles (block-level)
  - Dense int8 matmul microkernel runs only on active tiles
  - Emit once per block
"""

import time
import numpy as np


def make_data(M=256, K=256, N=256, tiles_active=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(M, K), dtype=np.int8)
    W = rng.integers(0, 3, size=(K, N), dtype=np.int8)
    # tile mask: True means compute this tile
    tiles = rng.random((M // 32, N // 32)) < tiles_active
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


def train_epoch(X, W, tiles, lr=1e-3):
    # forward block-sparse
    C = block_sparse_matmul(X, W, tiles)
    # fake target: zeros
    err = C
    # backward: simple gradient on W for active tiles
    gradW = np.zeros_like(W, dtype=np.int32)
    tile = 32
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


def bench(fn, *args, reps=3):
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
    M = N = K = 256
    tile_active = 0.5
    epochs = 3
    X, W, tiles = make_data(M, K, N, tile_active)

    # Baseline dense timing
    t_dense = bench(dense_matmul, X, W)

    # Block-sparse timing
    t_bs = bench(block_sparse_matmul, X, W, tiles)

    print("Block-sparse MoE-style matmul")
    print(f"M=N=K={M}, active tiles ~{tile_active*100:.1f}%")
    print(f"dense matmul      : {t_dense:6.2f} ms/call")
    print(f"block-sparse matmul: {t_bs:6.2f} ms/call   speedup x{t_dense/t_bs:5.2f}")

    # Tiny training loop (for illustration)
    for e in range(epochs):
        t0 = time.perf_counter()
        W, loss = train_epoch(X, W, tiles, lr=1e-5)
        t1 = time.perf_counter()
        print(f"epoch {e+1}: loss={loss:8.2e}  time={(t1-t0)*1e3:6.2f} ms")


if __name__ == "__main__":
    main()

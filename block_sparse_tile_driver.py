"""
block_sparse_tile_driver.py
---------------------------
Build a block/tile mask from order-ternary P/Q/N planes, then run a dense
int8 microkernel (dp-style) only on active tiles. This demonstrates the
"mask early, compute dense inside the tile" pattern.
"""

import time
import numpy as np


def make_states(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=shape, dtype=np.uint8)


def pqn_build_tiles(states, tile_m=32, tile_n=32, thresh=1):
    """
    states: 2D array of {0,1,2}
    Return list of active tile indices (i,j) where tile has any state >= thresh.
    """
    H, W = states.shape
    tiles = []
    for i in range(0, H, tile_m):
        for j in range(0, W, tile_n):
            block = states[i : i + tile_m, j : j + tile_n]
            if thresh <= 1:
                active = np.any(block >= 1)
            else:
                active = np.any(block == 2)
            if active:
                tiles.append((i, j))
    return tiles


def dense_tile_dot(A, B, tile_m=32, tile_n=32, tile_k=32):
    """
    Dense int8 microkernel over one tile: C += A_tile @ B_tile
    """
    # expects A: (tile_m, tile_k), B: (tile_k, tile_n)
    return A.astype(np.int32) @ B.astype(np.int32)


def block_sparse_run(A, B, tiles, tile_m=32, tile_n=32, tile_k=32):
    """
    Run dense microkernel over active tiles only.
    NOTE: This only computes active tiles; other regions remain zero.
    A: (M,K), B: (K,N)
    tiles: list of (i,j) tile offsets in output space
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.int32)
    for (i0, j0) in tiles:
        i1 = min(i0 + tile_m, M)
        j1 = min(j0 + tile_n, N)
        # slide k in chunks to cover full K
        for k0 in range(0, K, tile_k):
            k1 = min(k0 + tile_k, K)
            Ablk = A[i0:i1, k0:k1]
            Bblk = B[k0:k1, j0:j1]
            C[i0:i1, j0:j1] += dense_tile_dot(Ablk, Bblk, tile_m, tile_n, k1 - k0)
    return C


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
    tile = 32
    states = make_states((M, N), seed=0)
    tiles = pqn_build_tiles(states, tile_m=tile, tile_n=tile, thresh=1)
    A = np.random.randint(0, 3, size=(M, K), dtype=np.int8)
    B = np.random.randint(0, 3, size=(K, N), dtype=np.int8)

    # correctness vs dense
    C_dense = A.astype(np.int32) @ B.astype(np.int32)
    C_bs = block_sparse_run(A, B, tiles, tile_m=tile, tile_n=tile, tile_k=tile)
    # tiles cover only active regions; for fairness, compare only those tiles
    for (i, j) in tiles:
        i1 = min(i + tile, M)
        j1 = min(j + tile, N)
        assert np.array_equal(C_dense[i:i1, j:j1], C_bs[i:i1, j:j1])

    t_dense = bench(lambda: A.astype(np.int32) @ B.astype(np.int32))
    t_bs = bench(block_sparse_run, A, B, tiles, tile, tile, tile)

    print("Block-sparse driver with dense microkernel on active tiles")
    print(f"M=N=K={M}, tile={tile}, active tiles={len(tiles)}/{(M//tile)*(N//tile)}")
    print(f"dense full matmul: {t_dense:8.2f} ms")
    print(f"block-sparse     : {t_bs:8.2f} ms   speedup x{t_dense/t_bs:5.2f}")


if __name__ == "__main__":
    main()

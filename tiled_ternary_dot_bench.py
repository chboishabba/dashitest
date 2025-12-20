"""
tiled_ternary_dot_bench.py
--------------------------
Control experiment: reuse the *same* blocked/tiling structure as an
int8/FP32 microkernel, but swap the inner arithmetic for ternary GF(3)
to test "structure vs algebra".
"""

import time
import numpy as np


def make_data(M=256, K=256, N=256, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 3, size=(M, K), dtype=np.int8)
    B = rng.integers(0, 3, size=(K, N), dtype=np.int8)
    return A, B


def blocked_int8_dot(A, B, block=32):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.int32)
    for i0 in range(0, M, block):
        for j0 in range(0, N, block):
            for k0 in range(0, K, block):
                i1 = min(i0 + block, M)
                j1 = min(j0 + block, N)
                k1 = min(k0 + block, K)
                # microkernel: straight multiply-accumulate
                for i in range(i0, i1):
                    for k in range(k0, k1):
                        a = int(A[i, k])
                        C[i, j0:j1] += a * B[k, j0:j1].astype(np.int32)
    return C


def blocked_ternary_gf3(A, B, block=32):
    """
    Same blocking as blocked_int8_dot, but arithmetic is GF(3):
      multiply mod 3, accumulate mod 3.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.int8)
    for i0 in range(0, M, block):
        for j0 in range(0, N, block):
            for k0 in range(0, K, block):
                i1 = min(i0 + block, M)
                j1 = min(j0 + block, N)
                k1 = min(k0 + block, K)
                for i in range(i0, i1):
                    for k in range(k0, k1):
                        a = int(A[i, k])
                        if a == 0:
                            continue
                        prod = (a * B[k, j0:j1].astype(np.int16)) % 3
                        C[i, j0:j1] = (C[i, j0:j1].astype(np.int16) + prod) % 3
    return C.astype(np.int8)


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


def roofline_stats_int8(M, K, N):
    # int32 accumulates of int8 muls
    ops = M * K * N * 2  # mul + add
    bytes_in = M * K + K * N  # int8 inputs
    bytes_out = M * N * 4     # int32 outputs
    bytes_total = bytes_in + bytes_out
    return ops, bytes_total


def roofline_stats_gf3(M, K, N):
    # mul/add mod3 per element
    ops = M * K * N * 2  # count similarly
    bytes_in = M * K + K * N  # int8 inputs
    bytes_out = M * N         # int8 outputs
    bytes_total = bytes_in + bytes_out
    return ops, bytes_total


def main():
    M = N = K = 256
    block = 32
    A, B = make_data(M, K, N)

    # Correctness sanity
    C_int = blocked_int8_dot(A, B, block)
    C_gf3 = blocked_ternary_gf3(A, B, block)
    ref_gf3 = (A.astype(np.int16) @ B.astype(np.int16)) % 3
    assert np.array_equal(ref_gf3.astype(np.int8), C_gf3)

    t_int = bench(blocked_int8_dot, A, B, block)
    t_gf3 = bench(blocked_ternary_gf3, A, B, block)

    int_ops, int_bytes = roofline_stats_int8(M, K, N)
    gf3_ops, gf3_bytes = roofline_stats_gf3(M, K, N)
    int_gflops = (int_ops / (t_int / 1e3)) / 1e9
    gf3_gops = (gf3_ops / (t_gf3 / 1e3)) / 1e9
    int_Bps = (int_bytes / (t_int / 1e3))
    gf3_Bps = (gf3_bytes / (t_gf3 / 1e3))
    int_opb = int_ops / int_bytes
    gf3_opb = gf3_ops / gf3_bytes

    print("Tiled dot product (same blocking, different algebra)")
    print(f"M=N=K={M}, block={block}")
    print(f"int8 dot : {t_int:8.2f} ms   {int_gflops:8.2f} Gops/s   {int_Bps/1e9:6.2f} GB/s   ops/byte={int_opb:4.2f}")
    print(f"GF3 dot  : {t_gf3:8.2f} ms   {gf3_gops:8.2f} Gops/s   {gf3_Bps/1e9:6.2f} GB/s   ops/byte={gf3_opb:4.2f}   speedup x{t_int/t_gf3:5.2f}")


if __name__ == "__main__":
    main()

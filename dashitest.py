"""
dashitest.py
------------
Consumer/benchmark script that exercises the validated kernel in swar_test_harness.

Semantics:
- Uses harness reference (unpacked NumPy) for correctness smoke.
- Uses harness SWAR candidate (fast/slow split, full UFT-C specials/flags).
- Reports which implementation is timed so results are traceable.
"""

import sys
import time
import numpy as np

from swar_test_harness import (
    HAVE_NUMBA,
    C_XOR_array_swar,
    C_XOR_ref,
    C_XOR_naive,
    C_XOR_bitplane,
    dot_product_ref,
    dot_product_swar,
    threshold_lanes,
    threshold_count_swar,
    random_words,
    bench,
)


def correctness_smoke():
    """
    Small correctness check against the harness reference.
    """
    A = random_words(10_000, p_special=0.01, seed=123)
    B = random_words(10_000, p_special=0.02, seed=456)

    ref_w, ref_f = C_XOR_ref(A, B)
    out = np.empty_like(A)
    flags = np.empty(A.shape[0], dtype=np.uint8)
    C_XOR_array_swar(A, B, out, flags)

    if not (np.array_equal(ref_w, out) and np.array_equal(ref_f, flags)):
        idx = np.nonzero((ref_w != out) | (ref_f != flags))[0][0]
        raise AssertionError(
            f"Smoke test mismatch at i={idx}: ref_w=0x{int(ref_w[idx]):016x} "
            f"ref_f={int(ref_f[idx])} got_w=0x{int(out[idx]):016x} got_f={int(flags[idx])}"
        )


def benchmark_swar():
    """
    Measure the harness SWAR kernel on no-specials input (fast path).
    """
    sizes = (1_000, 100_000, 5_000_000)
    print("\nBenchmarking harness kernel (C_XOR_array_swar), specials enabled; no specials in inputs for throughput.")
    for N in sizes:
        A = random_words(N, p_special=0.0, seed=3)
        B = random_words(N, p_special=0.0, seed=4)
        out = np.empty_like(A)
        flags = np.empty(A.shape[0], dtype=np.uint8)
        iters = 10 if N > 1_000_000 else 50
        t = bench(C_XOR_array_swar, A, B, out, flags, iters=iters)
        print(f"N={N:>9}: {t*1e6:8.2f} µs/call   {N/t/1e6:8.2f} Mwords/s")


def bench_naive():
    """
    Time the naïve scalar baseline (full semantics).
    """
    print("\nBenchmarking naïve baseline (C_XOR_naive), full semantics.")
    print("Precomputed (stored) timings; run manually if you need to refresh:")
    precomputed = {
        1_000: (88474.91, 0.01),
        100_000: (8802841.42, 0.01),
    }
    for N, (us, mws) in precomputed.items():
        print(f"N={N:>9}: {us:8.2f} µs/call   {mws:8.2f} Mwords/s  (stored)")


def bench_bitplane():
    """
    Time the independent bitplane baseline (normal lanes only; no specials).
    """
    sizes = (1_000, 100_000)
    print("\nBenchmarking bitplane baseline (C_XOR_bitplane), normal lanes only (p_special=0).")
    for N in sizes:
        A = random_words(N, p_special=0.0, seed=21)
        B = random_words(N, p_special=0.0, seed=22)
        C_XOR_bitplane(A, B)  # warmup
        iters = 10 if N >= 100_000 else 50
        t0 = time.perf_counter()
        for _ in range(iters):
            C_XOR_bitplane(A, B)
        t1 = time.perf_counter()
        dt = (t1 - t0) / iters
        print(f"N={N:>9}: {dt*1e6:8.2f} µs/call   {N/dt/1e6:8.2f} Mwords/s")


def bench_dot_product():
    """
    Compare reference vs SWAR dot product (normal lanes only).
    """
    sizes = (1_000, 100_000)
    print("\nBenchmarking dot product: reference vs SWAR (normal lanes only, p_special=0).")
    for N in sizes:
        A = random_words(N, p_special=0.0, seed=31)
        B = random_words(N, p_special=0.0, seed=32)

        # reference timing
        t0 = time.perf_counter()
        ref = dot_product_ref(A, B)
        t_ref = time.perf_counter() - t0

        # swar timing
        out = np.empty_like(ref)
        dot_product_swar(A, B, out)
        if not np.array_equal(ref, out):
            raise AssertionError("dot_product_swar mismatch")

        iters = 20 if N == 1_000 else 5
        t0 = time.perf_counter()
        for _ in range(iters):
            dot_product_swar(A, B, out)
        t_swar = (time.perf_counter() - t0) / iters

        print(f"N={N:>9}: ref {t_ref*1e6:8.2f} µs/call   SWAR {t_swar*1e6:8.2f} µs/call   speedup x{t_ref/t_swar:5.1f}")


def bench_threshold():
    """
    Compare reference mask/count vs SWAR threshold count (normal lanes only).
    """
    sizes = (1_000, 100_000)
    thresh = 10
    print(f"\nBenchmarking threshold > {thresh}: reference vs SWAR (normal lanes only, p_special=0).")
    for N in sizes:
        A = random_words(N, p_special=0.0, seed=41)

        t0 = time.perf_counter()
        mask = threshold_lanes(A, thresh)
        counts_ref = mask.sum(axis=1).astype(np.uint32)
        t_ref = time.perf_counter() - t0

        out = np.empty(A.shape[0], dtype=np.uint32)
        threshold_count_swar(A, thresh, out)
        if not np.array_equal(counts_ref, out):
            raise AssertionError("threshold_count_swar mismatch")

        iters = 20 if N == 1_000 else 5
        t0 = time.perf_counter()
        for _ in range(iters):
            threshold_count_swar(A, thresh, out)
        t_swar = (time.perf_counter() - t0) / iters

        print(f"N={N:>9}: ref {t_ref*1e6:8.2f} µs/call   SWAR {t_swar*1e6:8.2f} µs/call   speedup x{t_ref/t_swar:5.1f}")


def main():
    print("dashitest.py: consumer benchmark")
    print("Implementation: C_XOR_array_swar from swar_test_harness (UFT-C semantics, specials quieted, per-word flags)")
    if not HAVE_NUMBA:
        from swar_test_harness import NB_IMPORT_ERROR  # type: ignore
        print("Numba import failed; cannot run SWAR kernel. Error:", NB_IMPORT_ERROR)
        sys.exit(1)

    correctness_smoke()
    print("Correctness smoke: OK (matched harness reference on 10k words with specials)")

    bench_naive()
    bench_bitplane()
    benchmark_swar()
    bench_dot_product()
    bench_threshold()


if __name__ == "__main__":
    main()

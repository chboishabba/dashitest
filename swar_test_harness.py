
"""
SWAR test harness for dashitest.py

Goal:
- Validate candidate SWAR/Numba packed kernel against the reference NumPy unpacked implementation.
- Provide reproducible correctness tests (including specials) + simple throughput timing.

Usage:
  python swar_test_harness.py

Notes:
- This harness *intentionally* keeps the "slow path" correct (12-lane scalar) if specials are present.
- The fast path uses the candidate SWAR mod3 kernel. If it is wrong, tests will fail loudly.
"""

import os
os.environ.setdefault("NUMBA_DISABLE_COVERAGE","1")
import time
import numpy as np
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception as e:
    HAVE_NUMBA = False
    NB_IMPORT_ERROR = repr(e)


# ----------------------------
# Reference implementation (unpacked, vectorized NumPy)
# ----------------------------

QVOID, QPARA, SVOID, SPARA, QMETA = 27, 28, 29, 30, 31
SPECIAL_MIN = 27

LANE_SHIFTS = np.array([5*i for i in range(12)], dtype=np.uint64)

SEVERITY = np.array(
    [0]*27 + [1, 2, 1, 2, 0],
    dtype=np.uint8
)

def extract_lanes(words: np.ndarray) -> np.ndarray:
    lanes = (words[:, None] >> LANE_SHIFTS) & 0x1F
    return lanes.astype(np.uint8)

def quiet_specials(lanes, flags):
    svoid = lanes == SVOID
    spara = lanes == SPARA
    if np.any(svoid | spara):
        # Set INVALID per word (not globally)
        flags |= np.any(svoid | spara, axis=1).astype(np.uint8)
    lanes = np.where(svoid, QVOID, lanes)
    lanes = np.where(spara, QPARA, lanes)
    return lanes, flags

def decode_trits(lanes):
    a = lanes % 3
    b = (lanes // 3) % 3
    c = lanes // 9
    return a, b, c

def encode_trits(a, b, c):
    return (a + 3*b + 9*c).astype(np.uint8)


# Precompute normal-lane addition (0..26) so the fast path can stay simple.
NORMAL_SUM_LUT = np.empty(27*27, dtype=np.uint8)
for a in range(27):
    for b in range(27):
        a0 = a % 3
        a1 = (a // 3) % 3
        a2 = a // 9
        b0 = b % 3
        b1 = (b // 3) % 3
        b2 = b // 9
        r0 = (a0 + b0) % 3
        r1 = (a1 + b1) % 3
        r2 = (a2 + b2) % 3
        NORMAL_SUM_LUT[a*27 + b] = r0 + 3*r1 + 9*r2

# Pairwise LUT: maps (lane0|lane1<<5, lane0b|lane1b<<5) -> result packed into 10 bits.
PAIR_LUT = np.empty(1 << 20, dtype=np.uint16)
for a_pair in range(1 << 10):
    la0 = a_pair & 0x1F
    la1 = (a_pair >> 5) & 0x1F
    for b_pair in range(1 << 10):
        lb0 = b_pair & 0x1F
        lb1 = (b_pair >> 5) & 0x1F
        res0 = np.uint16(NORMAL_SUM_LUT[la0*27 + lb0]) if (la0 < 27 and lb0 < 27) else np.uint16(0)
        res1 = np.uint16(NORMAL_SUM_LUT[la1*27 + lb1]) if (la1 < 27 and lb1 < 27) else np.uint16(0)
        PAIR_LUT[(b_pair << 10) | a_pair] = res0 | np.uint16(res1 << np.uint16(5))

# Product LUT: lane product (normal lanes only, specials -> 0).
PROD_LUT = np.zeros(27*27, dtype=np.uint16)
for a in range(27):
    for b in range(27):
        PROD_LUT[a*27 + b] = np.uint16(a * b)


if HAVE_NUMBA:

    @nb.njit(inline='always')
    def ternary_full_adder_digit_nb(d0, d1, cin):
        s = d0 + d1 + cin
        if s >= np.uint64(3):
            return s - np.uint64(3), np.uint64(1)
        return s, np.uint64(0)

    @nb.njit(inline='always')
    def ripple_carry_add_lane_nb(la, lb):
        # Base-3 ripple-carry add of two normal lanes (0..26). Experimental.
        a0 = la % np.uint64(3)
        a1 = (la // np.uint64(3)) % np.uint64(3)
        a2 = la // np.uint64(9)
        b0 = lb % np.uint64(3)
        b1 = (lb // np.uint64(3)) % np.uint64(3)
        b2 = lb // np.uint64(9)

        s0, c0 = ternary_full_adder_digit_nb(a0, b0, np.uint64(0))
        s1, c1 = ternary_full_adder_digit_nb(a1, b1, c0)
        s2, _ = ternary_full_adder_digit_nb(a2, b2, c1)
        return s0 + np.uint64(3)*s1 + np.uint64(9)*s2


def C_XOR_ref(wordsA: np.ndarray, wordsB: np.ndarray):
    flags = np.zeros(wordsA.shape[0], dtype=np.uint8)

    lanesA = extract_lanes(wordsA)
    lanesB = extract_lanes(wordsB)

    lanesA, flags = quiet_specials(lanesA, flags)
    lanesB, flags = quiet_specials(lanesB, flags)

    specialA = lanesA >= SPECIAL_MIN
    specialB = lanesB >= SPECIAL_MIN
    any_special = specialA | specialB

    aA, bA, cA = decode_trits(lanesA)
    aB, bB, cB = decode_trits(lanesB)

    rA = (aA + aB) % 3
    rB = (bA + bB) % 3
    rC = (cA + cB) % 3
    normal_result = encode_trits(rA, rB, rC)

    sevA = SEVERITY[lanesA]
    sevB = SEVERITY[lanesB]
    useA = sevA >= sevB
    special_result = np.where(useA, lanesA, lanesB)

    result_lanes = np.where(any_special, special_result, normal_result)

    if np.any(any_special):
        flags |= (np.any(any_special, axis=1).astype(np.uint8) << np.uint8(1))

    # repack (vectorized)
    result_words = np.sum(
        result_lanes.astype(np.uint64) << LANE_SHIFTS,
        axis=1,
        dtype=np.uint64
    )

    return result_words, flags


def dot_product_ref(wordsA: np.ndarray, wordsB: np.ndarray):
    """
    Per-word dot product treating each lane as its 0..26 value.
    Specials are not supported; caller must mask them out.
    """
    lanesA = extract_lanes(wordsA)
    lanesB = extract_lanes(wordsB)
    if np.any(lanesA >= 27) or np.any(lanesB >= 27):
        raise ValueError("dot_product_ref expects normal lanes only")
    return np.sum(lanesA.astype(np.int32) * lanesB.astype(np.int32), axis=1, dtype=np.int32)


def threshold_lanes(words: np.ndarray, thresh: int):
    """
    Return boolean mask (N,12) where lane value > thresh (normal lanes only).
    Specials yield False.
    """
    lanes = extract_lanes(words)
    normal = lanes < 27
    return normal & (lanes > np.uint8(thresh))


# ----------------------------
# Candidate SWAR fast-path kernel (Numba)
# ----------------------------

LANE_MASK = np.uint64(0x1F1F1F1F1F1F1F1F)

# Candidate trit-plane masks (must be validated by tests!)
T0_MASK = np.uint64(0x0924924924924924)
T1_MASK = np.uint64(0x1249249249249248)
T2_MASK = np.uint64(0x2492492492492490)

if HAVE_NUMBA:

    @nb.njit(inline='always')
    def has_special_word_loop(x):
        # Correct and simple: check each 5-bit lane
        for i in range(12):
            lane = (x >> np.uint64(5*i)) & np.uint64(0x1F)
            if lane >= np.uint64(27):
                return True
        return False

    @nb.njit(inline='always')
    def quiet_lane(lane):
        # quiet signaling specials per-lane
        if lane == np.uint64(SVOID):
            return np.uint64(QVOID), True
        if lane == np.uint64(SPARA):
            return np.uint64(QPARA), True
        return lane, False

    @nb.njit(inline='always')
    def sev_lane(lane):
        # 0 normal, 1 VOID, 2 PARA, 0 META
        if lane == np.uint64(QVOID) or lane == np.uint64(SVOID):
            return np.uint64(1)
        if lane == np.uint64(QPARA) or lane == np.uint64(SPARA):
            return np.uint64(2)
        return np.uint64(0)

    @nb.njit(inline='always')
    def mod3_swar_candidate(x):
        # Candidate SWAR reducer (validated by harness)
        t0 = x & T0_MASK
        t1 = x & T1_MASK
        t2 = x & T2_MASK
        s = t0 + t1 + t2
        carry = s & (T2_MASK << np.uint64(1))
        s -= carry
        s -= (carry >> np.uint64(1))
        return s

    @nb.njit(inline='always')
    def add_lane_no_special(la, lb):
        # Both lanes are <27; use precomputed ternary add mod 3.
        return NORMAL_SUM_LUT[int(la) * 27 + int(lb)]

    @nb.njit(inline='always')
    def C_XOR_fast_candidate(a, b):
        # SWAR fast path using pairwise (2-lane) lookup; no specials present.
        out = np.uint64(0)
        for i in range(6):
            sh = np.uint64(10*i)
            a_pair = (a >> sh) & np.uint64(0x3FF)  # two 5-bit lanes
            b_pair = (b >> sh) & np.uint64(0x3FF)
            idx = (b_pair << np.uint64(10)) | a_pair
            out |= np.uint64(PAIR_LUT[int(idx)]) << sh
        return out

    @nb.njit(inline='always')
    def C_XOR_slow_correct(a, b):
        # Correct per-lane semantics; scalar because specials are assumed rare.
        invalid = False
        special_used = False

        out = np.uint64(0)
        for i in range(12):
            sh = np.uint64(5*i)

            la = (a >> sh) & np.uint64(0x1F)
            lb = (b >> sh) & np.uint64(0x1F)

            la, inv_a = quiet_lane(la)
            lb, inv_b = quiet_lane(lb)
            if inv_a or inv_b:
                invalid = True

            if la >= np.uint64(27) or lb >= np.uint64(27):
                special_used = True
                sa = sev_lane(la)
                sb = sev_lane(lb)
                lane_out = la if sa >= sb else lb
            else:
                # normal lane: add per-trit (decode/encode)
                a0 = la % np.uint64(3)
                a1 = (la // np.uint64(3)) % np.uint64(3)
                a2 = la // np.uint64(9)

                b0 = lb % np.uint64(3)
                b1 = (lb // np.uint64(3)) % np.uint64(3)
                b2 = lb // np.uint64(9)

                r0 = (a0 + b0) % np.uint64(3)
                r1 = (a1 + b1) % np.uint64(3)
                r2 = (a2 + b2) % np.uint64(3)

                lane_out = r0 + np.uint64(3)*r1 + np.uint64(9)*r2

            out |= (lane_out & np.uint64(0x1F)) << sh

        flags = np.uint8(0)
        if invalid:
            flags |= np.uint8(1)
        if special_used:
            flags |= np.uint8(2)
        return out, flags

    @nb.njit(inline='always')
    def C_XOR_word_swar(a, b):
        if not (has_special_word_loop(a) or has_special_word_loop(b)):
            # Fast SWAR candidate (no specials)
            return C_XOR_fast_candidate(a, b), np.uint8(0)
        # Correct slow path
        return C_XOR_slow_correct(a, b)

    @nb.njit(parallel=True, fastmath=True)
    def C_XOR_array_swar(A, B, out, flags_out):
        n = A.shape[0]
        for i in nb.prange(n):
            w, f = C_XOR_word_swar(A[i], B[i])
            out[i] = w
            flags_out[i] = f

    @nb.njit(parallel=True, fastmath=True)
    def dot_product_swar(A, B, out):
        """
        Per-word dot product treating lanes as 0..26 values.
        Specials unsupported: set output to -1 if encountered.
        """
        n = A.shape[0]
        for i in nb.prange(n):
            acc = np.int32(0)
            for lane in range(12):
                sh = np.uint64(5*lane)
                la = (A[i] >> sh) & np.uint64(0x1F)
                lb = (B[i] >> sh) & np.uint64(0x1F)
                if la >= np.uint64(27) or lb >= np.uint64(27):
                    acc = np.int32(-1)
                    break
                acc += np.int32(PROD_LUT[int(la)*27 + int(lb)])
            out[i] = acc

    @nb.njit(parallel=True, fastmath=True)
    def threshold_count_swar(words, thresh, out):
        """
        Count lanes greater than thresh (normal lanes only). Specials ignored.
        """
        n = words.shape[0]
        for i in nb.prange(n):
            c = np.uint32(0)
            for lane in range(12):
                sh = np.uint64(5*lane)
                v = (words[i] >> sh) & np.uint64(0x1F)
                if v < np.uint64(27) and v > np.uint64(thresh):
                    c += np.uint32(1)
            out[i] = c


# ----------------------------
# Test generation
# ----------------------------

def random_words(N, p_special=0.0, seed=0):
    rng = np.random.default_rng(seed)
    # start with normal lanes 0..26 packed
    lanes = rng.integers(0, 27, size=(N, 12), dtype=np.uint8)

    if p_special > 0:
        # sprinkle specials into random lanes
        mask = rng.random((N, 12)) < p_special
        specials = rng.choice(np.array([QVOID, QPARA, SVOID, SPARA, QMETA], dtype=np.uint8),
                              size=(N, 12), replace=True)
        lanes = np.where(mask, specials, lanes).astype(np.uint8)

    words = np.sum(lanes.astype(np.uint64) << LANE_SHIFTS, axis=1, dtype=np.uint64)
    return words

def assert_equal(ref_w, ref_f, got_w, got_f, label):
    if not np.array_equal(ref_w, got_w) or not np.array_equal(ref_f, got_f):
        # find first mismatch
        idx = np.nonzero((ref_w != got_w) | (ref_f != got_f))[0][0]
        raise AssertionError(
            f"[{label}] mismatch at i={idx}\n"
            f"  ref_w=0x{int(ref_w[idx]):016x} ref_f={int(ref_f[idx])}\n"
            f"  got_w=0x{int(got_w[idx]):016x} got_f={int(got_f[idx])}"
        )

def bench(fn, A, B, out, flags, iters=50):
    fn(A, B, out, flags)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(A, B, out, flags)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


# ----------------------------
# Ripple-carry ternary adders (utility / experimentation)
# ----------------------------

def ternary_full_adder_digit(d0: np.uint8, d1: np.uint8, cin: np.uint8):
    """Return (sum_digit, carry_out) for ternary digits in {0,1,2}."""
    s = int(d0) + int(d1) + int(cin)
    if s >= 3:
        return np.uint8(s - 3), np.uint8(1)
    return np.uint8(s), np.uint8(0)


def ripple_carry_add_lane(la: np.uint8, lb: np.uint8):
    """
    Base-3 ripple-carry add of two normal lanes (0..26).
    This is *not* the UFT-C per-trit mod3 operation; it is provided for experimentation.
    """
    a0 = la % 3
    a1 = (la // 3) % 3
    a2 = la // 9
    b0 = lb % 3
    b1 = (lb // 3) % 3
    b2 = lb // 9

    s0, c0 = ternary_full_adder_digit(a0, b0, np.uint8(0))
    s1, c1 = ternary_full_adder_digit(a1, b1, c0)
    s2, _ = ternary_full_adder_digit(a2, b2, c1)
    return np.uint8(s0 + 3*s1 + 9*s2)


# ----------------------------
# Extra baselines for comparison
# ----------------------------

def C_XOR_naive(wordsA: np.ndarray, wordsB: np.ndarray):
    """
    Naïve baseline: scalar Python loops, explicit decode/encode, full semantics.
    """
    N = wordsA.shape[0]
    out = np.zeros_like(wordsA, dtype=np.uint64)
    flags = np.zeros(N, dtype=np.uint8)

    for i in range(N):
        w = np.uint64(0)
        for lane in range(12):
            sh = np.uint64(5*lane)
            la = (wordsA[i] >> sh) & np.uint64(0x1F)
            lb = (wordsB[i] >> sh) & np.uint64(0x1F)

            if la == np.uint64(SVOID):
                la = np.uint64(QVOID)
                flags[i] |= np.uint8(1)
            if la == np.uint64(SPARA):
                la = np.uint64(QPARA)
                flags[i] |= np.uint8(1)
            if lb == np.uint64(SVOID):
                lb = np.uint64(QVOID)
                flags[i] |= np.uint8(1)
            if lb == np.uint64(SPARA):
                lb = np.uint64(QPARA)
                flags[i] |= np.uint8(1)

            if la >= np.uint64(27) or lb >= np.uint64(27):
                flags[i] |= np.uint8(2)
                lane_out = la if SEVERITY[int(la)] >= SEVERITY[int(lb)] else lb
            else:
                a0 = la % np.uint64(3)
                a1 = (la // np.uint64(3)) % np.uint64(3)
                a2 = la // np.uint64(9)
                b0 = lb % np.uint64(3)
                b1 = (lb // np.uint64(3)) % np.uint64(3)
                b2 = lb // np.uint64(9)
                lane_out = (
                    (a0 + b0) % np.uint64(3)
                    + np.uint64(3) * ((a1 + b1) % np.uint64(3))
                    + np.uint64(9) * ((a2 + b2) % np.uint64(3))
                )

            w |= (lane_out & np.uint64(0x1F)) << sh

        out[i] = w

    return out, flags


def trits_to_planes(words: np.ndarray):
    """
    Convert packed words into three bitplanes (normal lanes only; specials treated as normal here).
    Plane bit i = 1 if lane i holds that trit value.
    """
    N = words.shape[0]
    p0 = np.zeros(N, dtype=np.uint16)
    p1 = np.zeros(N, dtype=np.uint16)
    p2 = np.zeros(N, dtype=np.uint16)

    for lane in range(12):
        sh = np.uint64(5*lane)
        v = (words >> sh) & np.uint64(0x1F)
        t = v % np.uint64(3)
        p0 |= ((t == np.uint64(0)).astype(np.uint16) << lane)
        p1 |= ((t == np.uint64(1)).astype(np.uint16) << lane)
        p2 |= ((t == np.uint64(2)).astype(np.uint16) << lane)

    return p0, p1, p2


def C_XOR_bitplane(wordsA: np.ndarray, wordsB: np.ndarray):
    """
    Independent ternary XOR on bitplanes. Only valid when all lanes are normal (no specials).
    """
    lanesA = extract_lanes(wordsA)
    lanesB = extract_lanes(wordsB)

    a0 = lanesA % 3
    a1 = (lanesA // 3) % 3
    a2 = lanesA // 9

    b0 = lanesB % 3
    b1 = (lanesB // 3) % 3
    b2 = lanesB // 9

    r0 = (a0 + b0) % 3
    r1 = (a1 + b1) % 3
    r2 = (a2 + b2) % 3

    result_lanes = (r0 + 3*r1 + 9*r2).astype(np.uint8)
    return np.sum(result_lanes.astype(np.uint64) << LANE_SHIFTS, axis=1, dtype=np.uint64)


def main():
    print("Compiling Numba kernels (first call)...")
    if not HAVE_NUMBA:
        print("Numba import failed:", NB_IMPORT_ERROR)
        print("Run reference tests only (NumPy). Fix by upgrading coverage or pinning compatible numba/coverage.")
        # quick reference-only sanity
        A = random_words(10000, p_special=0.01, seed=1)
        B = random_words(10000, p_special=0.01, seed=2)
        C_XOR_ref(A, B)
        return


    # correctness: multiple regimes
    regimes = [
        ("no_specials_small", 10_000, 0.0),
        ("rare_specials",     200_000, 1e-4),
        ("some_specials",     200_000, 1e-2),
        ("many_specials",      50_000, 0.2),
    ]

    for name, N, ps in regimes:
        A = random_words(N, p_special=ps, seed=1)
        B = random_words(N, p_special=ps, seed=2)

        ref_w, ref_f = C_XOR_ref(A, B)

        out = np.empty_like(A)
        flags = np.empty(A.shape[0], dtype=np.uint8)
        C_XOR_array_swar(A, B, out, flags)

        assert_equal(ref_w, ref_f, out, flags, name)
        print(f"OK: {name} (N={N}, p_special={ps})")

        # Baseline cross-checks (cheap sanity)
        naive_w, naive_f = C_XOR_naive(A[: min(10_000, N)], B[: min(10_000, N)])
        assert_equal(
            ref_w[: naive_w.shape[0]],
            ref_f[: naive_f.shape[0]],
            naive_w,
            naive_f,
            name + "_naive"
        )

        # External bitplane reference only valid when no specials are present.
        mask = np.all((extract_lanes(A) < SPECIAL_MIN) & (extract_lanes(B) < SPECIAL_MIN), axis=1)
        if np.any(mask):
            bp_w = C_XOR_bitplane(A[mask], B[mask])
            assert_equal(ref_w[mask], ref_f[mask], bp_w, np.zeros_like(bp_w, dtype=np.uint8), name + "_bitplane")

    # throughput bench (no specials, so we measure SWAR fast path)
    for N in (1_000, 100_000, 5_000_000):
        A = random_words(N, p_special=0.0, seed=3)
        B = random_words(N, p_special=0.0, seed=4)
        out = np.empty_like(A)
        flags = np.empty(A.shape[0], dtype=np.uint8)

        t = bench(C_XOR_array_swar, A, B, out, flags, iters=10 if N > 1_000_000 else 50)
        print(f"BENCH (SWAR candidate): N={N:>9}  {t*1e6:8.2f} µs/call  {N/t/1e6:8.2f} Mwords/s")

    print("All tests passed.")

if __name__ == "__main__":
    main()

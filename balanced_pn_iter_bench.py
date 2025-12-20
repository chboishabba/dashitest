"""
balanced_pn_iter_bench.py
-------------------------
Prototype 2-bit balanced-ternary (P,N) bitplane kernels and an iterative p-adic-style benchmark.

Representation:
- Two uint64 bitplanes: P has bit=1 for +1, N has bit=1 for -1. (0,0)=0, (1,1)=invalid.
- 64 lanes per word.

Addition (balanced with carry):
- We use carry-save style: first combine two words to produce sum bits and carry bits (for +/-2).
- Propagate carries iteratively by shifting carry planes left (×3) and re-adding until no carry.

Benchmark:
- Run K iterations of state = state + input (balanced ternary) with carry propagation.
- Measures ALU-heavy iterative loop on a cache-resident dataset.
"""

import time
import numpy as np

try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


def random_pn_words(N, p_neg=0.33, p_pos=0.33, seed=0):
    rng = np.random.default_rng(seed)
    # Values: -1, 0, +1 with given probs (remaining mass to 0)
    vals = rng.choice(np.array([-1, 0, 1], dtype=np.int8),
                      size=N*64,
                      p=[p_neg, 1 - p_neg - p_pos, p_pos]).reshape(N, 64)
    P = np.zeros(N, dtype=np.uint64)
    Nn = np.zeros(N, dtype=np.uint64)
    for lane in range(64):
        mask = np.uint64(1) << np.uint64(lane)
        P |= mask * (vals[:, lane] == 1)
        Nn |= mask * (vals[:, lane] == -1)
    return P, Nn


def add_balanced_carry_save(Pa, Na, Pb, Nb):
    """
    Carry-save balanced add of two PN words.
    Returns (sumP, sumN, carryP, carryN) where carry planes encode +/-2.
    """
    # +2 and -2 carry bits
    carryP = Pa & Pb
    carryN = Na & Nb

    # lanes with opposite signs cancel to 0
    cancel = (Pa & Nb) | (Na & Pb)

    # remaining single +/-1 lanes
    sumP = (Pa ^ Pb) & ~ (Na | Nb)
    sumN = (Na ^ Nb) & ~ (Pa | Pb)

    # zero out cancelled lanes in sums
    sumP &= ~cancel
    sumN &= ~cancel

    return sumP, sumN, carryP, carryN


def add_balanced_with_carry(Pa, Na, Pb, Nb):
    """
    Full balanced add with carry propagation (iterative).
    """
    sumP, sumN, carryP, carryN = add_balanced_carry_save(Pa, Na, Pb, Nb)
    # Propagate carries: a carry corresponds to +/-2 == +/-1 at current digit and carry of +/-1 to next digit (shift by 1 lane).
    while carryP != 0 or carryN != 0:
        # carry contributes +/-1 at current lane, and a shifted +/-1 to next lane
        shiftedP = carryP << np.uint64(1)
        shiftedN = carryN << np.uint64(1)
        # add current sum with shifted carry (carry magnitude becomes +1/-1)
        sumP, sumN, carryP, carryN = add_balanced_carry_save(sumP | carryP, sumN | carryN, shiftedP, shiftedN)
    return sumP, sumN


def iterative_accumulate(P_state, N_state, P_in, N_in, iters=64):
    """
    Run iterative p-adic-style accumulation: state = state + input with carry.
    """
    for _ in range(iters):
        P_state, N_state = add_balanced_with_carry(P_state, N_state, P_in, N_in)
    return P_state, N_state


def bench_iterative(N=1024, iters=128, seed=0):
    P_in, N_in = random_pn_words(1, seed=seed)
    P_state, N_state = random_pn_words(N, seed=seed + 1)

    # warmup
    iterative_accumulate(P_state[0], N_state[0], P_in[0], N_in[0], iters=4)

    t0 = time.perf_counter()
    for i in range(N):
        iterative_accumulate(P_state[i], N_state[i], P_in[0], N_in[0], iters=iters)
    t1 = time.perf_counter()

    dt = (t1 - t0) / N
    ops = iters * 64  # trit updates per word
    print(f"Iterative balanced PN add: N={N}, iters={iters}, {dt*1e6:8.2f} µs/word, {ops/dt/1e6:8.2f} Mtrits/s")


if HAVE_NUMBA:
    @nb.njit(inline="always")
    def add_balanced_carry_save_nb(Pa, Na, Pb, Nb):
        carryP = Pa & Pb
        carryN = Na & Nb
        cancel = (Pa & Nb) | (Na & Pb)
        sumP = (Pa ^ Pb) & ~ (Na | Nb)
        sumN = (Na ^ Nb) & ~ (Pa | Pb)
        sumP &= ~cancel
        sumN &= ~cancel
        return sumP, sumN, carryP, carryN

    @nb.njit(inline="always")
    def add_balanced_with_carry_nb(Pa, Na, Pb, Nb):
        sumP, sumN, carryP, carryN = add_balanced_carry_save_nb(Pa, Na, Pb, Nb)
        while carryP != 0 or carryN != 0:
            shiftedP = carryP << np.uint64(1)
            shiftedN = carryN << np.uint64(1)
            sumP, sumN, carryP, carryN = add_balanced_carry_save_nb(sumP | carryP, sumN | carryN, shiftedP, shiftedN)
        return sumP, sumN

    @nb.njit(parallel=True, fastmath=True)
    def iterative_accumulate_nb(P_state, N_state, P_in, N_in, iters):
        N = P_state.shape[0]
        outP = np.empty_like(P_state)
        outN = np.empty_like(N_state)
        for i in nb.prange(N):
            p = P_state[i]
            n = N_state[i]
            for _ in range(iters):
                p, n = add_balanced_with_carry_nb(p, n, P_in, N_in)
            outP[i] = p
            outN[i] = n
        return outP, outN


def bench_iterative_nb(N=1024, iters=128, seed=0):
    if not HAVE_NUMBA:
        print("Numba not available; skipping NB benchmark.")
        return
    P_in, N_in = random_pn_words(1, seed=seed)
    P_state, N_state = random_pn_words(N, seed=seed + 1)
    P_in0 = np.uint64(P_in[0])
    N_in0 = np.uint64(N_in[0])

    # warmup compile
    iterative_accumulate_nb(P_state[:1], N_state[:1], P_in0, N_in0, 4)

    t0 = time.perf_counter()
    iterative_accumulate_nb(P_state, N_state, P_in0, N_in0, iters)
    t1 = time.perf_counter()

    dt = (t1 - t0) / N
    ops = iters * 64
    print(f"[NB] Iterative balanced PN add: N={N}, iters={iters}, {dt*1e6:8.2f} µs/word, {ops/dt/1e6:8.2f} Mtrits/s")


def main():
    print("balanced_pn_iter_bench: 2-bit (P,N) balanced add with carry, iterative loop")
    bench_iterative(N=1024, iters=128, seed=123)
    bench_iterative(N=1024, iters=512, seed=123)
    bench_iterative_nb(N=1024, iters=128, seed=123)
    bench_iterative_nb(N=1024, iters=512, seed=123)


if __name__ == "__main__":
    main()

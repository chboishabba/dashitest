"""
z3_phase_fixedpoint_bench.py
----------------------------
Bench a simple ℤ3 phase-lattice update in three representations:
- Unpacked int8 lanes (-1,0,1)
- P/N bitplanes (balanced ternary), staying packed

Update rule (1D ring):
    state = state ⊕ left ⊕ right  (balanced add with carry, mod 3)
Iterated many times to keep data hot in cache.
"""

import time
import numpy as np
import numba as nb


# ----------------------------
# Helpers: balanced bitplanes
# ----------------------------

def random_balanced_int8(N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=N)


def pack_pn_from_int8(arr):
    N = arr.shape[0]
    P = np.zeros((N // 64,), dtype=np.uint64)
    Nn = np.zeros((N // 64,), dtype=np.uint64)
    for i in range(N // 64):
        block = arr[64*i:64*(i+1)]
        wordP = np.uint64(0)
        wordN = np.uint64(0)
        for lane in range(64):
            mask = np.uint64(1) << np.uint64(lane)
            v = block[lane]
            if v > 0:
                wordP |= mask
            elif v < 0:
                wordN |= mask
        P[i] = wordP
        Nn[i] = wordN
    return P, Nn


def unpack_pn_to_int8(P, Nn):
    N = P.shape[0] * 64
    out = np.zeros(N, dtype=np.int8)
    idx = 0
    for wordP, wordN in zip(P, Nn):
        for lane in range(64):
            mask = np.uint64(1) << np.uint64(lane)
            out[idx] = np.int8((1 if (wordP & mask) else 0) - (1 if (wordN & mask) else 0))
            idx += 1
    return out


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
def pn_update(P_state, N_state, iters):
    nwords = P_state.shape[0]
    curP = P_state.copy()
    curN = N_state.copy()
    nextP = np.empty_like(curP)
    nextN = np.empty_like(curN)
    for _ in range(iters):
        for i in nb.prange(nwords):
            p = curP[i]
            n = curN[i]
            leftP = curP[(i - 1) % nwords]
            leftN = curN[(i - 1) % nwords]
            rightP = curP[(i + 1) % nwords]
            rightN = curN[(i + 1) % nwords]
            p, n = add_balanced_with_carry_nb(p, n, leftP, leftN)
            p, n = add_balanced_with_carry_nb(p, n, rightP, rightN)
            nextP[i] = p
            nextN[i] = n
        curP, nextP = nextP, curP
        curN, nextN = nextN, curN
    return curP, curN


# ----------------------------
# Baseline (reference, slow)
# ----------------------------

def baseline_update(state, iters):
    # Use bitplane add with carry in Python loops for correctness baseline.
    P, Nn = pack_pn_from_int8(state)
    for _ in range(iters):
        curP = P.copy()
        curN = Nn.copy()
        for i in range(P.shape[0]):
            p = curP[i]
            n = curN[i]
            leftP = curP[(i - 1) % curP.shape[0]]
            leftN = curN[(i - 1) % curN.shape[0]]
            rightP = curP[(i + 1) % curP.shape[0]]
            rightN = curN[(i + 1) % curN.shape[0]]
            p, n = add_balanced_with_carry_nb(np.uint64(p), np.uint64(n), np.uint64(leftP), np.uint64(leftN))
            p, n = add_balanced_with_carry_nb(np.uint64(p), np.uint64(n), np.uint64(rightP), np.uint64(rightN))
            P[i] = p
            Nn[i] = n
    return unpack_pn_to_int8(P, Nn)


def bench(fn, *args, iters=1):
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 4096  # 64 * 64 lanes
    iters = 64
    state = random_balanced_int8(N, seed=0)

    # Baseline
    t_base = bench(baseline_update, state, iters, iters=1)
    out_base = baseline_update(state, iters)

    # P/N packed
    P, Nn = pack_pn_from_int8(state)
    pn_update(P[:1], Nn[:1], 1)  # compile
    t_pn = bench(pn_update, P, Nn, iters, iters=3)
    outP, outN = pn_update(P, Nn, iters)
    out_pn = unpack_pn_to_int8(outP, outN)

    if not np.array_equal(out_base, out_pn):
        raise AssertionError("Mismatch between baseline and P/N packed update")

    ops = N * iters * 3  # three adds per site (center+left+right)
    print("ℤ3 phase-lattice update (1D ring, balanced add with carry):")
    print(f"N={N}, iters={iters}: baseline {t_base*1e3:8.2f} ms/call ({ops/t_base/1e6:8.2f} Mop/s) "
          f"P/N SWAR {t_pn*1e3:8.2f} ms/call ({ops/t_pn/1e6:8.2f} Mop/s) "
          f"speedup x{t_base/t_pn:5.2f}")


if __name__ == "__main__":
    main()

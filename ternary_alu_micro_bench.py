"""
ternary_alu_micro_bench.py
--------------------------
Ternary ALU microkernel benchmark:
- Apply the same ternary op repeatedly over a working set (iterations).
- Includes specials (VOID/PARA) in the input.
- Compares packed SWAR kernel vs a per-lane emulator (instruction-by-instruction style).
"""

import time
import numpy as np
from swar_test_harness import (
    C_XOR_array_swar,
    random_words,
    QVOID, QPARA, SVOID, SPARA, SPECIAL_MIN,
)

SEVERITY = np.array(
    [0]*27 + [1, 2, 1, 2, 0],
    dtype=np.uint8
)


def emulator_step(A, B):
    """
    Per-word, per-lane emulator of C_XOR semantics (instruction-style).
    Returns (out, flags).
    """
    N = A.shape[0]
    out = np.zeros_like(A, dtype=np.uint64)
    flags = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        w_out = np.uint64(0)
        inv = False
        special_used = False
        for lane in range(12):
            sh = np.uint64(5*lane)
            la = (A[i] >> sh) & np.uint64(0x1F)
            lb = (B[i] >> sh) & np.uint64(0x1F)
            # quiet specials
            if la == np.uint64(SVOID):
                la = np.uint64(QVOID); inv = True
            if la == np.uint64(SPARA):
                la = np.uint64(QPARA); inv = True
            if lb == np.uint64(SVOID):
                lb = np.uint64(QVOID); inv = True
            if lb == np.uint64(SPARA):
                lb = np.uint64(QPARA); inv = True

            if la >= np.uint64(SPECIAL_MIN) or lb >= np.uint64(SPECIAL_MIN):
                special_used = True
                sa = SEVERITY[int(la)]
                sb = SEVERITY[int(lb)]
                lane_out = la if sa >= sb else lb
            else:
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
            w_out |= (lane_out & np.uint64(0x1F)) << sh
        f = np.uint8(0)
        if inv:
            f |= np.uint8(1)
        if special_used:
            f |= np.uint8(2)
        out[i] = w_out
        flags[i] = f
    return out, flags


def bench(fn, state, inp, tmp_out, tmp_flags, iters):
    # warmup
    fn(state, inp, tmp_out, tmp_flags)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(state, inp, tmp_out, tmp_flags)
        state[:] = tmp_out
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def bench_emulator(state, inp, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        out, flags = emulator_step(state, inp)
        state[:] = out
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 4_096
    iters = 128
    A = random_words(N, p_special=0.01, seed=1)
    B = random_words(N, p_special=0.01, seed=2)

    # SWAR path
    state_swar = A.copy()
    tmp_out = np.empty_like(state_swar)
    tmp_flags = np.empty(N, dtype=np.uint8)
    t_swar = bench(C_XOR_array_swar, state_swar, B, tmp_out, tmp_flags, iters)

    # Emulator path
    state_emu = A.copy()
    t_emu = bench_emulator(state_emu, B, iters)

    # sanity
    assert np.array_equal(state_swar, state_emu), "Final states diverged between SWAR and emulator"

    print("Ternary ALU microkernel (iterative XOR with specials):")
    print(f"N={N}, iters={iters}: SWAR {t_swar*1e3:8.2f} ms/iter   Emulator {t_emu*1e3:8.2f} ms/iter   speedup x{t_emu/t_swar:6.1f}")


if __name__ == "__main__":
    main()

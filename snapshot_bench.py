"""
snapshot_bench.py
-----------------
Demonstrate hot compute in P/N bitplanes with optional cold snapshots in 5-trit-per-byte form.

Workflow:
  - Keep state in packed 12-lane words (P/N implicitly via swar_test_harness extract/pack)
  - Iterate K times with XOR + threshold + dot (hot loop)
  - Optionally snapshot every M iters to 5-trit/byte packed bytes
Compares:
  - Hot loop only (no snapshots)
  - Hot loop with snapshot overhead
Reports throughput and snapshot size.
"""

import time
import numpy as np
from swar_test_harness import (
    C_XOR_array_swar,
    threshold_count_swar,
    dot_product_swar,
    extract_lanes,
    random_words,
)

LANE_SHIFTS = np.array([5*i for i in range(12)], dtype=np.uint64)


def pack_5trit_bytes(lanes):
    """
    Pack lanes (N,12) uint8 (0..26) into bytes holding 5 trits each (~5% waste).
    We only pack the low 5 trits here for simplicity (12 -> 3 bytes, last 3 lanes dropped).
    """
    lanes16 = lanes.astype(np.uint16, copy=False)
    # Use first 10 trits (2 bytes) to avoid complexity; adjust as needed
    N = lanes.shape[0]
    out = np.zeros((N, 2), dtype=np.uint8)
    for i in range(N):
        t0 = lanes16[i, 0] + 3*lanes16[i, 1] + 9*lanes16[i, 2] + 27*lanes16[i, 3] + 81*lanes16[i, 4]
        t1 = lanes16[i, 5] + 3*lanes16[i, 6] + 9*lanes16[i, 7] + 27*lanes16[i, 8] + 81*lanes16[i, 9]
        out[i, 0] = t0
        out[i, 1] = t1
    return out


def hot_loop(words_state, words_inp, iters=128, thresh=0):
    state = words_state.copy()
    tmp_out = np.empty_like(state)
    tmp_flags = np.empty(state.shape[0], dtype=np.uint8)
    counts = np.empty(state.shape[0], dtype=np.uint32)
    dots = np.empty(state.shape[0], dtype=np.int32)
    for _ in range(iters):
        C_XOR_array_swar(state, words_inp, tmp_out, tmp_flags)
        state[:] = tmp_out
        threshold_count_swar(state, thresh, counts)
        dot_product_swar(state, state, dots)
    return state


def hot_loop_with_snapshots(words_state, words_inp, iters=128, snap_every=16, thresh=0):
    state = words_state.copy()
    tmp_out = np.empty_like(state)
    tmp_flags = np.empty(state.shape[0], dtype=np.uint8)
    counts = np.empty(state.shape[0], dtype=np.uint32)
    dots = np.empty(state.shape[0], dtype=np.int32)
    total_snap_bytes = 0
    for i in range(iters):
        C_XOR_array_swar(state, words_inp, tmp_out, tmp_flags)
        state[:] = tmp_out
        threshold_count_swar(state, thresh, counts)
        dot_product_swar(state, state, dots)
        if (i + 1) % snap_every == 0:
            lanes = extract_lanes(state)
            snaps = pack_5trit_bytes(lanes)
            total_snap_bytes += snaps.nbytes
    return state, total_snap_bytes


def bench(fn, *args, iters=3):
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 8192
    iters = 128
    snap_every = 16
    state = random_words(N, p_special=0.0, seed=0)
    inp = random_words(N, p_special=0.0, seed=1)

    t_hot = bench(hot_loop, state, inp, iters, 0, iters=3)
    out_hot = hot_loop(state, inp, iters, 0)

    def run_snap():
        return hot_loop_with_snapshots(state, inp, iters, snap_every, 0)

    run_snap()  # warmup
    t_snap = time.perf_counter()
    out_snap, snap_bytes = run_snap()
    t_snap = (time.perf_counter() - t_snap)

    if not np.array_equal(out_hot, out_snap):
        raise AssertionError("Snapshot path diverged from hot loop")

    ops = N * iters * 12 * 3
    print("Snapshot benchmark (hot P/N compute, optional 5-trit/byte snapshots):")
    print(f"N={N}, iters={iters}, snap_every={snap_every}")
    print(f"Hot only: {t_hot*1e3:8.2f} ms/call ({ops/t_hot/1e6:8.2f} Mop/s)")
    print(f"Hot+snap: {t_snap*1e3:8.2f} ms/call ({ops/t_snap/1e6:8.2f} Mop/s), snapshots={snap_bytes/1024:.2f} KiB")


if __name__ == "__main__":
    main()

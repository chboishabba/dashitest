"""
packing_ablation_bench.py
-------------------------
Compare three representations on the same iterative workload (mod3 add + threshold + dot):
1) Unpacked lanes (int8)
2) Radix-packed wire format: pack/unpack every iteration (overhead visible)
3) Packed SWAR staying packed (C_XOR_array_swar)
"""

import time
import numpy as np
from swar_test_harness import (
    C_XOR_array_swar,
    extract_lanes,
    random_words,
)

LANE_SHIFTS = np.array([5*i for i in range(12)], dtype=np.uint64)


def pack_words(lanes):
    return np.sum(lanes.astype(np.uint64) << LANE_SHIFTS, axis=1, dtype=np.uint64)


def unpack_words(words):
    lanes = (words[:, None] >> LANE_SHIFTS) & np.uint64(0x1F)
    return lanes.astype(np.uint8)


def baseline_iter(lanes_state, lanes_inp, thresh, iters):
    cur = lanes_state.copy()
    for _ in range(iters):
        cur = (cur + lanes_inp) % 3
        counts = (cur > thresh).sum(axis=1)
        _ = counts
        _ = np.sum(cur * cur, axis=1)
    return cur


def radix_iter(words_state, words_inp, thresh, iters):
    cur = words_state.copy()
    for _ in range(iters):
        lanes_state = unpack_words(cur)
        lanes_inp = unpack_words(words_inp)
        lanes_state = (lanes_state + lanes_inp) % 3
        counts = (lanes_state > thresh).sum(axis=1)
        _ = counts
        _ = np.sum(lanes_state * lanes_state, axis=1)
        cur = pack_words(lanes_state)
    return cur


def swar_iter(words_state, words_inp, thresh, iters):
    cur = words_state.copy()
    tmp_out = np.empty_like(cur)
    tmp_flags = np.empty(cur.shape[0], dtype=np.uint8)
    counts = np.empty(cur.shape[0], dtype=np.uint32)
    for _ in range(iters):
        C_XOR_array_swar(cur, words_inp, tmp_out, tmp_flags)  # mod3 add packed
        cur[:] = tmp_out
        # threshold (unpacked)
        lanes = extract_lanes(cur)
        counts = (lanes > thresh).sum(axis=1)
        _ = counts
        _ = np.sum(lanes * lanes, axis=1)
    return cur


def bench(fn, *args, iters=1):
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 4096
    iters = 64
    thresh = 1
    words_state = random_words(N, p_special=0.0, seed=0) % 3
    words_inp = random_words(N, p_special=0.0, seed=1) % 3
    lanes_state = extract_lanes(words_state)
    lanes_inp = extract_lanes(words_inp)

    out_base = baseline_iter(lanes_state, lanes_inp, thresh, iters)
    t_base = bench(baseline_iter, lanes_state, lanes_inp, thresh, iters, iters=1)

    out_radix = radix_iter(words_state, words_inp, thresh, iters)
    lanes_radix = unpack_words(out_radix)
    t_radix = bench(radix_iter, words_state, words_inp, thresh, iters, iters=1)

    out_swar = swar_iter(words_state, words_inp, thresh, iters)
    lanes_swar = extract_lanes(out_swar)
    t_swar = bench(swar_iter, words_state, words_inp, thresh, iters, iters=3)

    if not np.array_equal(out_base, lanes_radix):
        raise AssertionError("Radix-packed result mismatch")
    if not np.array_equal(out_base, lanes_swar):
        raise AssertionError("SWAR result mismatch")

    ops = N * iters * 12 * 3
    print(f"Packing ablation: N={N}, iters={iters}, threshold={thresh}")
    print(f"Unpacked:  {t_base*1e3:8.2f} ms/call ({ops/t_base/1e6:8.2f} Mop/s)")
    print(f"Radix (pack/unpack): {t_radix*1e3:8.2f} ms/call ({ops/t_radix/1e6:8.2f} Mop/s)")
    print(f"Packed SWAR: {t_swar*1e3:8.2f} ms/call ({ops/t_swar/1e6:8.2f} Mop/s)")
    print(f"Speedup SWAR vs unpacked: x{t_base/t_swar:5.2f}, vs radix: x{t_radix/t_swar:5.2f}")


if __name__ == "__main__":
    main()

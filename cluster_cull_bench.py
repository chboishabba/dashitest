"""
cluster_cull_bench.py
---------------------
Synthetic cluster culling benchmark (Nanite-like "classify -> mask -> emit"):
- States per cluster: 12 lanes with values {0 empty, 1 refine, 2 emit}.
- Per iteration:
    mask_refine = state > 0
    mask_emit   = state > 1
    count_refine, count_emit aggregated
- Compare baseline (unpacked int8) vs packed SWAR using threshold_count_swar.
"""

import time
import numpy as np
from swar_test_harness import (
    threshold_count_swar,
    extract_lanes,
    random_words,
)


def random_states(N, seed=0):
    rng = np.random.default_rng(seed)
    lanes = rng.integers(0, 3, size=(N, 12), dtype=np.uint8)
    words = np.sum(lanes.astype(np.uint64) << np.array([5*i for i in range(12)], dtype=np.uint64),
                   axis=1, dtype=np.uint64)
    return words, lanes


def baseline_epoch(lanes_state, iters=64, rng=None):
    total_refine = 0
    total_emit = 0
    cur = lanes_state.copy()
    if rng is None:
        rng = np.random.default_rng(0)
    for _ in range(iters):
        mask_refine = cur > 0
        mask_emit = cur > 1
        total_refine += int(mask_refine.sum())
        total_emit += int(mask_emit.sum())
        # simple update: flip some lanes from refine to emit to simulate work
        rnd = rng.random(cur.shape)
        cur = np.where(mask_refine & (~mask_emit) & (rnd < 0.1), 2, cur)
    return total_refine, total_emit


def swar_epoch(words_state, iters=64, rng=None):
    total_refine = 0
    total_emit = 0
    cur = words_state.copy()
    tmp_counts = np.empty(cur.shape[0], dtype=np.uint32)
    if rng is None:
        rng = np.random.default_rng(0)
    for _ in range(iters):
        threshold_count_swar(cur, 0, tmp_counts)
        total_refine += int(tmp_counts.sum())
        threshold_count_swar(cur, 1, tmp_counts)
        total_emit += int(tmp_counts.sum())
        # simple update: promote some refine lanes to emit (random mask)
        lanes = extract_lanes(cur)
        rnd = rng.random(lanes.shape)
        promote = (lanes == 1) & (rnd < 0.1)
        lanes = np.where(promote, 2, lanes)
        cur = np.sum(lanes.astype(np.uint64) << np.array([5*i for i in range(12)], dtype=np.uint64),
                     axis=1, dtype=np.uint64)
    return total_refine, total_emit


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
    words, lanes = random_states(N, seed=0)
    # Baseline
    t_base = bench(baseline_epoch, lanes, iters, iters=1)
    ref_refine, ref_emit = baseline_epoch(lanes, iters)
    # SWAR
    t_swar = bench(swar_epoch, words, iters, iters=3)
    sw_refine, sw_emit = swar_epoch(words, iters)

    if (ref_refine != sw_refine) or (ref_emit != sw_emit):
        raise AssertionError(f"Mismatch: baseline refine={ref_refine} emit={ref_emit}, "
                             f"SWAR refine={sw_refine} emit={sw_emit}")

    print("Cluster culling (classify -> mask -> emit) synthetic benchmark:")
    print(f"N={N}, iters={iters}: baseline {t_base*1e3:8.2f} ms/epoch   "
          f"SWAR {t_swar*1e3:8.2f} ms/epoch   speedup x{t_base/t_swar:5.2f}")


if __name__ == "__main__":
    main()

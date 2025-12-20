"""
five_trit_pack_bench.py
-----------------------
Microbench to measure 5-trit-per-byte packing/unpacking cost vs staying packed words.
Not a compute kernel—just raw (un)pack throughput for 5 trits/byte (storage) vs 12-lane words (compute).
"""

import time
import numpy as np
from swar_test_harness import extract_lanes, random_words


LANE_SHIFTS = np.array([5*i for i in range(12)], dtype=np.uint64)


def pack_5trit_bytes(lanes):
    """
    Pack 10 trits into 2 bytes (5 trits/byte). Drops lanes >=10.
    """
    lanes16 = lanes.astype(np.uint16, copy=False)
    N = lanes.shape[0]
    out = np.zeros((N, 2), dtype=np.uint8)
    for i in range(N):
        t0 = lanes16[i, 0] + 3*lanes16[i, 1] + 9*lanes16[i, 2] + 27*lanes16[i, 3] + 81*lanes16[i, 4]
        t1 = lanes16[i, 5] + 3*lanes16[i, 6] + 9*lanes16[i, 7] + 27*lanes16[i, 8] + 81*lanes16[i, 9]
        out[i, 0] = t0
        out[i, 1] = t1
    return out


def unpack_5trit_bytes(packed):
    """
    Unpack 2 bytes into 10 trits.
    """
    N = packed.shape[0]
    out = np.zeros((N, 10), dtype=np.uint8)
    pow3 = np.array([1, 3, 9, 27, 81], dtype=np.uint8)
    for i in range(N):
        t0 = packed[i, 0]
        t1 = packed[i, 1]
        for j in range(5):
            out[i, j] = (t0 // pow3[j]) % 3
            out[i, 5 + j] = (t1 // pow3[j]) % 3
    return out


def bench(fn, *args, iters=10):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    N = 100_000
    words = random_words(N, p_special=0.0, seed=0) % 3
    lanes = extract_lanes(words)
    t_pack = bench(pack_5trit_bytes, lanes, iters=5)
    packed = pack_5trit_bytes(lanes)
    t_unpack = bench(unpack_5trit_bytes, packed, iters=5)
    print("5-trit/byte pack/unpack throughput (10 trits handled per item):")
    print(f"N={N}: pack {t_pack*1e3:8.2f} ms/call, unpack {t_unpack*1e3:8.2f} ms/call, packed size {packed.nbytes/1024:.2f} KiB")


if __name__ == "__main__":
    main()

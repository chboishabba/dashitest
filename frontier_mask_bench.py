import time
import numpy as np


LANES_PER_WORD = 12


def build_masks_from_state(state: np.ndarray):
    """
    Build three disjoint bit masks (value == 0/1/2) from a (words, lanes) array.
    Masks are uint64 arrays; lane i is bit (i % 64) in mask[i//64].
    Returns (mask0, mask1, mask2, valid_mask, total_lanes).
    """
    flat = state.ravel()
    total = flat.size
    blocks = (total + 63) // 64
    idx = np.arange(total, dtype=np.int64)
    blk = idx >> 6
    off = idx & 63
    shifts = np.left_shift(np.uint64(1), off.astype(np.uint64))

    m0 = np.zeros(blocks, dtype=np.uint64)
    m1 = np.zeros(blocks, dtype=np.uint64)
    m2 = np.zeros(blocks, dtype=np.uint64)

    for val, mask in ((0, m0), (1, m1), (2, m2)):
        sel = flat == val
        if np.any(sel):
            np.bitwise_or.at(mask, blk[sel], shifts[sel])

    valid = np.full(blocks, np.uint64(~np.uint64(0)), dtype=np.uint64)
    rem = total & 63
    if rem:
        valid[-1] = (np.uint64(1) << rem) - np.uint64(1)

    return m0, m1, m2, valid, total


def build_sparse_masks(state: np.ndarray):
    """
    Sparse variant: pack all lanes into Python ints (bitsets).
    Returns (m0, m1, m2, valid_mask_int, total_lanes).
    """
    flat = state.ravel()
    total = flat.size
    m0 = 0
    m1 = 0
    m2 = 0
    for pos, v in enumerate(flat):
        if v == 0:
            m0 |= 1 << pos
        elif v == 1:
            m1 |= 1 << pos
        else:
            m2 |= 1 << pos
    valid = (1 << total) - 1
    return m0, m1, m2, valid, total


def mask_to_bool(mask: np.ndarray, total: int) -> np.ndarray:
    """Convert a uint64 bit mask array back to a flat bool array of length total."""
    out = np.zeros(total, dtype=bool)
    for i, v in enumerate(mask):
        base = i * 64
        val = int(v)
        while val:
            lsb = val & -val
            bit = (lsb.bit_length() - 1)
            pos = base + bit
            if pos < total:
                out[pos] = True
            val ^= lsb
    return out


def frontier_baseline(state: np.ndarray, iters: int) -> np.ndarray:
    """Baseline per-lane frontier evolution."""
    s = state.copy()
    active = s != 2
    for _ in range(iters):
        refine = s == 1
        done = s == 2
        active = (active & ~done) | refine
        s = (s + 1) % 3
    return active


def frontier_mask_variant(state: np.ndarray, iters: int) -> np.ndarray:
    """Mask-native frontier evolution: no per-iter lane extraction."""
    m0, m1, m2, valid, total = build_masks_from_state(state)
    active = (~m2) & valid
    for _ in range(iters):
        refine = m1
        done = m2
        active = ((active & ~done) | refine) & valid
        m0, m1, m2 = m2, m0, m1  # rotate values: 0->1->2->0
    return mask_to_bool(active, total)


def frontier_mask_variant_blocks(state: np.ndarray, iters: int) -> np.ndarray:
    """Block-sparse: operate only on blocks with any activity."""
    m0, m1, m2, valid, total = build_masks_from_state(state)
    active_blocks = ((m0 | m1 | m2) & valid) != 0
    idx = np.flatnonzero(active_blocks)
    if idx.size == 0:
        return np.zeros(total, dtype=bool)
    m0b = m0[idx].copy()
    m1b = m1[idx].copy()
    m2b = m2[idx].copy()
    validb = valid[idx]
    activeb = (~m2b) & validb
    for _ in range(iters):
        refine = m1b
        done = m2b
        activeb = ((activeb & ~done) | refine) & validb
        m0b, m1b, m2b = m2b, m0b, m1b
    full_active = np.zeros_like(m0)
    full_active[idx] = activeb
    return mask_to_bool(full_active, total)


def frontier_mask_variant_sparse(state: np.ndarray, iters: int) -> np.ndarray:
    """Sparse bitset frontier evolution using Python bigints."""
    m0, m1, m2, valid, total = build_sparse_masks(state)
    active = (~m2) & valid
    for _ in range(iters):
        refine = m1
        done = m2
        active = ((active & ~done) | refine) & valid
        m0, m1, m2 = m2, m0, m1
    out = np.zeros(total, dtype=bool)
    a = active
    while a:
        lsb = a & -a
        pos = lsb.bit_length() - 1
        out[pos] = True
        a ^= lsb
    return out


def bench(n_words=4096, iters=128, lanes=LANES_PER_WORD, reps=5, mode="dense", state=None):
    rng = np.random.default_rng(0)
    state = state if state is not None else rng.integers(0, 3, size=(n_words, lanes), dtype=np.int8)

    # correctness
    ref_active = frontier_baseline(state, iters)
    if mode == "dense":
        mask_active = frontier_mask_variant(state, iters)
    elif mode == "block":
        mask_active = frontier_mask_variant_blocks(state, iters)
    else:
        mask_active = frontier_mask_variant_sparse(state, iters)
    if not np.array_equal(ref_active.ravel(), mask_active):
        raise AssertionError("Mask-native frontier does not match baseline.")

    def time_fn(fn):
        best = None
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            dur = (time.perf_counter() - start) * 1000.0
            best = dur if best is None or dur < best else best
        return best

    baseline_ms = time_fn(lambda: frontier_baseline(state, iters))
    if mode == "dense":
        mask_ms = time_fn(lambda: frontier_mask_variant(state, iters))
    elif mode == "block":
        mask_ms = time_fn(lambda: frontier_mask_variant_blocks(state, iters))
    else:
        mask_ms = time_fn(lambda: frontier_mask_variant_sparse(state, iters))

    print(f"Frontier evolution (n_words={n_words}, lanes={lanes}, iters={iters}, mode={mode})")
    print(f"baseline : {baseline_ms:8.2f} ms/epoch")
    print(f"mask     : {mask_ms:8.2f} ms/epoch   speedup x {baseline_ms / mask_ms:5.2f}")


if __name__ == "__main__":
    import os

    # Mode selection: one-time density discovery, no per-iter scans.
    # Modes:
    # - dense  (bitset SWAR) for high occupancy
    # - block  (active word list) for block-sparse occupancy
    # - sparse (bigint queue) only when words are ultra-sparse
    env_sparse = os.environ.get("SPARSE", "")
    force_sparse = env_sparse not in ("", "0", "false", "False")

    rng = np.random.default_rng(0)
    state = rng.integers(0, 3, size=(4096, LANES_PER_WORD), dtype=np.int8)
    active_word_mask = (state != 2).any(axis=1)
    active_ratio = active_word_mask.mean()

    # Hysteresis-style thresholds to avoid mode flapping; PREV_MODE can bias.
    prev_mode = os.environ.get("PREV_MODE", "").lower()
    dense_hi, dense_lo = 0.25, 0.10
    block_hi, block_lo = 0.20, 0.005

    if force_sparse:
        mode = "sparse"
    else:
        if prev_mode == "dense" and active_ratio > dense_lo:
            mode = "dense"
        elif prev_mode == "block" and block_lo < active_ratio < block_hi:
            mode = "block"
        else:
            if active_ratio < 0.001:
                mode = "sparse"
            elif active_ratio < 0.15:
                mode = "block"
            else:
                mode = "dense"

    bench(mode=mode, state=state)

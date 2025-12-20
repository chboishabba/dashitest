import time
import numpy as np

try:
    import numba
    from numba import types
    from numba.extending import intrinsic
except ImportError:  # pragma: no cover
    numba = None

LANES_PER_WORD = 64  # use full bit-width for simple packing


def pack_bitplanes(vals: np.ndarray):
    """
    vals: (words, lanes) with entries in {-1,0,1}
    returns (P, N) uint64 arrays, one word per element.
    """
    words, lanes = vals.shape
    assert lanes == LANES_PER_WORD, f"lanes must be {LANES_PER_WORD}"
    weights = (np.uint64(1) << np.arange(lanes, dtype=np.uint64))
    pos = (vals == 1).astype(np.uint64)
    neg = (vals == -1).astype(np.uint64)
    P = pos @ weights
    N = neg @ weights
    return P, N


def baseline_checksum(vals: np.ndarray, iters: int):
    res = None
    for _ in range(iters):
        res = vals.sum(axis=1) % 3
    return res


def swar_checksum(P: np.ndarray, N: np.ndarray, iters: int):
    res = None
    # vectorized popcount via byte lookup (fallback)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount_fallback(arr: np.ndarray) -> np.ndarray:
        env_hw = os.environ.get("USE_HW_POPCNT", "")
        if env_hw not in ("", "0", "false", "False"):
            return np.fromiter((int(x).bit_count() for x in arr), dtype=np.int64, count=arr.size)
        bytes_view = arr.view(np.uint8).reshape(arr.size, 8)
        return table[bytes_view].sum(axis=1, dtype=np.int64)

    pop_fn = None
    if numba is not None:

        @intrinsic
        def popcount_u64(typingctx, x):
            if x != types.uint64:
                raise TypeError("popcount_u64 expects uint64")
            sig = types.uint64(x)

            def codegen(context, builder, signature, args):
                (val,) = args
                return builder.ctpop(val)

            return sig, codegen

        @numba.njit
        def popcount_numba(arr):
            out = np.empty(arr.shape[0], dtype=np.int64)
            for i in range(arr.shape[0]):
                out[i] = popcount_u64(arr[i])
            return out

        pop_fn = popcount_numba

    def popcount(arr: np.ndarray) -> np.ndarray:
        if pop_fn is not None:
            return pop_fn(arr)
        return popcount_fallback(arr)

    for _ in range(iters):
        pc = popcount(P)
        nc = popcount(N)
        res = (pc - nc) % 3
    return res


def bench(n_words=4096, iters=128, reps=5):
    rng = np.random.default_rng(0)
    vals = rng.integers(-1, 2, size=(n_words, LANES_PER_WORD), dtype=np.int8)
    P, N = pack_bitplanes(vals)

    # correctness
    ref = baseline_checksum(vals, 1)
    test = swar_checksum(P, N, 1)
    if not np.array_equal(ref, test):
        raise AssertionError("SWAR checksum mismatch.")

    def time_fn(fn):
        best = None
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            dur = (time.perf_counter() - start) * 1000.0
            best = dur if best is None or dur < best else best
        return best

    baseline_ms = time_fn(lambda: baseline_checksum(vals, iters))
    swar_ms = time_fn(lambda: swar_checksum(P, N, iters))

    print(f"GF(3) checksum (n_words={n_words}, lanes={LANES_PER_WORD}, iters={iters})")
    print(f"baseline : {baseline_ms:8.2f} ms/epoch")
    print(f"SWAR     : {swar_ms:8.2f} ms/epoch   speedup x {baseline_ms / swar_ms:5.2f}")


if __name__ == "__main__":
    bench()

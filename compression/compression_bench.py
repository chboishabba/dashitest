import argparse
import lzma
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# --- Small triadic CA generator (no plotting; deterministic, fast) ---

def _neighbor_counts_ternary(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return counts of values {0,1,2} in a 3x3 Moore neighborhood (including self)."""
    shifts = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    c0 = np.zeros_like(x, dtype=np.int16)
    c1 = np.zeros_like(x, dtype=np.int16)
    c2 = np.zeros_like(x, dtype=np.int16)
    for dy, dx in shifts:
        rolled = np.roll(np.roll(x, dy, axis=0), dx, axis=1)
        c0 += rolled == 0
        c1 += rolled == 1
        c2 += rolled == 2
    return c0, c1, c2


def _step_tri_ca(G: np.ndarray, F: np.ndarray, A: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One CA step (triadic gate/flow with anchors and fatigue).

    G: gate in {-1,0,+1} mapped to {2,0,1} for storage (int8)
    F: flow in {0,1,2}
    A: anchor in {-1,0,+1} mapped to {2,0,1}
    u: fatigue in [0, 15]
    """
    # Anchor and flow neighborhood influence
    pA, nA, _ = _neighbor_counts_ternary(A)
    pF, nF, _ = _neighbor_counts_ternary(F)

    anchor_score = pA - nA  # positive supports ACT, negative supports BAN
    flow_score = pF - nF
    conflict = np.minimum(pF, nF)

    G_next = G.copy()
    u_next = u.copy()

    # Motifs: shutdown (M9), fatigue rim (M7), corridor (M4)
    m9 = (anchor_score <= -2) & (conflict >= 3) & (G == 1)
    G_next[m9] = -1

    m7 = (G == 1) & (u >= 6) & (anchor_score < 2) & (~m9)
    G_next[m7] = 0

    m4 = (anchor_score >= 3) & (G != -1)
    G_next[m4] = 1

    recover = (G == -1) & (anchor_score >= 4) & (conflict <= 1)
    G_next[recover] = 1

    engaged = (G_next == 1) & (F != 0)
    u_next[engaged] = np.minimum(u_next[engaged] + 1, 15)
    u_next[~engaged] = np.maximum(u_next[~engaged] - 1, 0)

    F_next = F.copy()
    F_next[G_next == -1] = 0

    hold = G_next == 0
    if hold.any():
        mask = ((flow_score + anchor_score) & 1) == 0
        F_next[hold & (F != 0) & mask] = 0

    act = G_next == 1
    if act.any():
        score = flow_score + 0.6 * anchor_score
        plus = score > 2.0
        minus = score < -2.0
        F_next[act & plus] = 1
        F_next[act & minus] = 2
        F_next[act & ~(plus | minus)] = 0

    return G_next, F_next, A, u_next


def generate_tri_ca_trace(
    height: int = 64, width: int = 64, steps: int = 128, seed: int = 0
) -> np.ndarray:
    """Generate a reproducible CA trace of gate states (encoded to {0,1,2})."""
    rng = np.random.default_rng(seed)
    A = rng.choice([0, 1, 2], size=(height, width), p=[0.25, 0.5, 0.25]).astype(np.int8)
    for _ in range(4):
        pA, nA, _ = _neighbor_counts_ternary(A)
        majority = np.where(pA > nA, 1, np.where(nA > pA, 2, 0))
        A = majority.astype(np.int8)

    G = np.where(A != 2, 1, 0).astype(np.int8)  # act when anchor non-negative
    F = rng.integers(0, 3, size=(height, width), dtype=np.int8)
    u = np.zeros((height, width), dtype=np.int8)

    frames: List[np.ndarray] = []
    for _ in range(steps):
        frames.append((G + 1).astype(np.uint8))  # encode -1/0/+1 -> 0/1/2 for compression
        G, F, A, u = _step_tri_ca(G, F, A, u)
    return np.stack(frames, axis=0)  # shape: (steps, H, W), dtype uint8 in {0,1,2}


# --- Compression utilities ---

def stream_entropy(symbols: np.ndarray) -> float:
    """Empirical entropy in bits/symbol."""
    counts = np.bincount(symbols, minlength=int(symbols.max()) + 1)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0


def to_residual_stream(frames: np.ndarray) -> np.ndarray:
    """Residuals mod 3; 0 means 'no change'."""
    first = frames[0].ravel()
    if len(frames) == 1:
        return first
    rest = (frames[1:] - frames[:-1]) % 3
    residuals = np.concatenate([first, rest.ravel()])
    return residuals.astype(np.uint8)


@dataclass
class CodecResult:
    bytes_out: int
    ms: float


def measure_stream(name: str, symbols: np.ndarray, codecs: List[str]) -> Dict:
    payload = symbols.astype(np.uint8).tobytes()
    ent = stream_entropy(symbols)
    compressed = {}
    for codec in codecs:
        if codec == "lzma":
            start = time.perf_counter()
            data = lzma.compress(payload, preset=6)
            elapsed = (time.perf_counter() - start) * 1000.0
            compressed[codec] = CodecResult(bytes_out=len(data), ms=elapsed)
        elif codec == "gzip":
            import gzip

            start = time.perf_counter()
            data = gzip.compress(payload, compresslevel=6)
            elapsed = (time.perf_counter() - start) * 1000.0
            compressed[codec] = CodecResult(bytes_out=len(data), ms=elapsed)
        elif codec == "zlib":
            import zlib

            start = time.perf_counter()
            data = zlib.compress(payload, level=6)
            elapsed = (time.perf_counter() - start) * 1000.0
            compressed[codec] = CodecResult(bytes_out=len(data), ms=elapsed)
        else:
            raise ValueError(f"unknown codec {codec}")

    return {
        "name": name,
        "symbols": symbols.size,
        "entropy_bits_per_symbol": ent,
        "compressed": compressed,
    }


def run_benchmark(height: int = 64, width: int = 64, steps: int = 128, seed: int = 0) -> List[Dict]:
    frames = generate_tri_ca_trace(height=height, width=width, steps=steps, seed=seed)
    raw_stream = frames.ravel().astype(np.uint8)
    residual_stream = to_residual_stream(frames)

    codecs = ["lzma", "gzip", "zlib"]
    results = [
        measure_stream("raw_gate", raw_stream, codecs),
        measure_stream("residual_mod3", residual_stream, codecs),
    ]
    return results


def _format_results(results: List[Dict], height: int, width: int, steps: int) -> str:
    rows = []
    cells = height * width * steps
    header = "stream          entropy  lzma (bytes/bpc/ms)    gzip (bytes/bpc/ms)    zlib (bytes/bpc/ms)"
    rows.append(header)
    for res in results:
        ent = res["entropy_bits_per_symbol"]
        line = f"{res['name']:<14} {ent:6.3f}  "
        for codec in ["lzma", "gzip", "zlib"]:
            c = res["compressed"][codec]
            bpc = (c.bytes_out * 8) / cells
            line += f"{c.bytes_out:6d}/{bpc:4.2f}/{c.ms:5.1f}   "
        rows.append(line.rstrip())
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark compression efficiency on a triadic CA trace.")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_benchmark(height=args.height, width=args.width, steps=args.steps, seed=args.seed)
    print(_format_results(results, args.height, args.width, args.steps))


if __name__ == "__main__":
    main()

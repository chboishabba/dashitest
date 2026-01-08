#!/usr/bin/env python3
"""
Generate DNA Task B `E_seq.npy` from sliding windows over a FASTA file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def load_fasta(path: Path) -> str:
    tokens: List[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            tokens.append(line.upper())
    return "".join(tokens)


def one_hot(seq: str) -> np.ndarray:
    mat = np.zeros((4, len(seq)), dtype=np.float32)
    for idx, base in enumerate(seq):
        if base in BASE_TO_IDX:
            mat[BASE_TO_IDX[base], idx] = 1.0
    return mat


def haar_detail(signal: np.ndarray, levels: int) -> List[np.ndarray]:
    arr = signal.astype(np.float32)
    details: List[np.ndarray] = []
    for _ in range(levels):
        length = arr.shape[0]
        if length < 2:
            details.append(np.zeros(1, dtype=np.float32))
            break
        even = (length // 2) * 2
        arr = arr[:even]
        avg = (arr[0::2] + arr[1::2]) / 2.0
        diff = (arr[0::2] - arr[1::2]) / 2.0
        details.append(diff)
        arr = avg
    while len(details) < levels:
        details.append(np.zeros(1, dtype=np.float32))
    return details


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump DNA band energies to E_seq.npy.")
    ap.add_argument("--fasta", type=Path, required=True, help="Input genome FASTA.")
    ap.add_argument("--window", type=int, default=512, help="Window length.")
    ap.add_argument("--step", type=int, default=128, help="Window stride.")
    ap.add_argument("--levels", type=int, default=5, help="Number of Haar detail levels per base.")
    ap.add_argument("--out", type=Path, required=True, help="E_seq.npy output.")
    args = ap.parse_args()

    seq = load_fasta(args.fasta)
    if len(seq) < args.window:
        raise ValueError("FASTA shorter than window length")
    num_windows = 1 + (len(seq) - args.window) // args.step
    bands = 4 * args.levels
    energies = np.zeros((num_windows, bands), dtype=np.float32)

    for t in range(num_windows):
        start = t * args.step
        window = seq[start : start + args.window]
        X = one_hot(window)
        band_idx = 0
        for channel in range(4):
            details = haar_detail(X[channel], args.levels)
            for detail in details:
                energies[t, band_idx] = float(np.mean(np.abs(detail)))
                band_idx += 1

    norms = energies.mean(axis=0, keepdims=True) + 1e-8
    energies /= norms

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, energies)

    print(f"[dna] wrote {args.out}")
    print(f"shape={energies.shape}, bands={bands}, windows={num_windows}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate the codec Task B `E_seq.npy` from stored balanced-ternary planes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_planes(path: Path) -> np.ndarray:
    data = np.load(path)
    if "planes" not in data:
        raise ValueError("NPZ must contain a 'planes' array")
    planes = data["planes"]
    if planes.ndim != 4:
        raise ValueError("planes must have shape [T, B, H, W]")
    return planes


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump codec band energies to E_seq.npy.")
    ap.add_argument("--planes", type=Path, required=True, help="Input npz with 'planes' [T,B,H,W].")
    ap.add_argument("--out", type=Path, required=True, help="Output E_seq.npy path.")
    args = ap.parse_args()

    planes = load_planes(args.planes)
    energies = np.mean(np.abs(planes), axis=(2, 3))
    energies = energies.astype(np.float32)
    norms = energies.mean(axis=0, keepdims=True) + 1e-8
    energies /= norms

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, energies)

    print(f"[codec] wrote {args.out}")
    print(f"shape={energies.shape}, bands={energies.shape[1]}, steps={energies.shape[0]}")


if __name__ == "__main__":
    main()

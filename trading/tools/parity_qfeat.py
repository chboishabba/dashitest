from __future__ import annotations

import argparse
import pathlib

import numpy as np

from features.quotient import compute_qfeat
from vk_qfeat import build_feature_tape
from trading_io.prices import find_btc_csv, load_prices


def parity_check(
    prices: np.ndarray,
    tape_path: str,
    *,
    w1: int = 64,
    w2: int = 256,
    eps: float = 1e-6,
    nan_squash: float = 0.0,
    tol: float = 1e-4,
    nsamples: int = 200,
    seed: int = 0,
    force: bool = False,
    backend: str = "vulkan",
    shader_path: str = "vulkan_shaders/qfeat.comp",
    spv_path: str = "vulkan_shaders/qfeat.spv",
    vk_icd: str | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float, float] | None]]:
    prices = np.asarray(prices, dtype=np.float32, order="C")
    if prices.ndim != 2:
        raise ValueError("prices must be 2D (series, timesteps)")
    S, T = prices.shape
    if T <= w2:
        raise ValueError("T must be greater than w2 for parity sampling")

    tape = build_feature_tape(
        prices=prices,
        out_path=tape_path,
        w1=w1,
        w2=w2,
        eps=eps,
        nan_squash=nan_squash,
        force=force,
        backend=backend,
        shader_path=shader_path,
        spv_path=spv_path,
        vk_icd=vk_icd,
    )

    rng = np.random.default_rng(seed)
    worst = np.zeros(6, dtype=np.float32)
    worst_idx: list[tuple[int, int, float, float] | None] = [None] * 6

    for _ in range(nsamples):
        series = int(rng.integers(0, S))
        timestep = int(rng.integers(w2, T))
        window = prices[series, (timestep - w2) : (timestep + 1)]
        cpu = compute_qfeat(window, w1=w1, w2=w2, eps=eps)
        gpu = tape.mm[series, timestep, :6]
        diff = np.abs(cpu - gpu)
        for idx in range(6):
            if diff[idx] > worst[idx]:
                worst[idx] = diff[idx]
                worst_idx[idx] = (series, timestep, float(cpu[idx]), float(gpu[idx]))

    return worst, worst_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Parity check for qfeat tape vs CPU")
    parser.add_argument(
        "--prices-csv",
        type=pathlib.Path,
        default=find_btc_csv(),
        help="CSV with [ts, close, volume] for a single symbol",
    )
    parser.add_argument("--tape", type=pathlib.Path, required=True, help="Output qfeat tape (.memmap)")
    parser.add_argument("--w1", type=int, default=64)
    parser.add_argument("--w2", type=int, default=256)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--force", action="store_true", help="Rebuild tape if it already exists")
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="vulkan")
    parser.add_argument("--shader", type=pathlib.Path, default=pathlib.Path("vulkan_shaders/qfeat.comp"))
    parser.add_argument("--spv", type=pathlib.Path, default=pathlib.Path("vulkan_shaders/qfeat.spv"))
    parser.add_argument("--vk-icd", type=str, default=None)
    args = parser.parse_args()

    price, volume, ts = load_prices(args.prices_csv, return_time=True)
    prices = price[np.newaxis, :]
    worst, worst_idx = parity_check(
        prices,
        str(args.tape),
        w1=args.w1,
        w2=args.w2,
        eps=1e-6,
        tol=args.tol,
        nsamples=args.nsamples,
        force=args.force,
        backend=args.backend,
        shader_path=str(args.shader),
        spv_path=str(args.spv),
        vk_icd=args.vk_icd,
    )

    print("worst diffs:", worst)
    for idx, record in enumerate(worst_idx):
        if record is None:
            continue
        series, timestep, cpu_val, gpu_val = record
        print(f"feature {idx}: series={series}, t={timestep}, cpu={cpu_val}, gpu={gpu_val}")


if __name__ == "__main__":
    main()

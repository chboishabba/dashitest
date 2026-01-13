from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for path in (str(ROOT), str(PARENT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from trading.features.quotient import compute_qfeat
    from trading.vk_qfeat import build_feature_tape
    from trading.trading_io.prices import find_btc_csv, load_prices
except ModuleNotFoundError:
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
    fp64_returns: bool = True,
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
        fp64_returns=fp64_returns,
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


def _debug_window(prices: np.ndarray, series: int, timestep: int, w2: int) -> None:
    start = timestep - w2
    end = timestep + 1
    window = prices[series, start:end]
    returns_len = max(0, window.size - 1)
    head = window[:3].tolist()
    tail = window[-3:].tolist()
    print(
        "debug window:",
        f"series={series}",
        f"t={timestep}",
        f"start={start}",
        f"end={end}",
        f"prices={window.size}",
        f"returns={returns_len}",
    )
    print("debug prices head:", head)
    print("debug prices tail:", tail)


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
    parser.add_argument("--no-fp64-returns", action="store_true", help="Disable FP64 log-return path")
    parser.add_argument(
        "--debug-feature",
        type=int,
        default=None,
        help="Print CPU window details for the worst diff of this feature index (0-5)",
    )
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
        fp64_returns=not args.no_fp64_returns,
    )

    print("worst diffs:", worst)
    for idx, record in enumerate(worst_idx):
        if record is None:
            continue
        series, timestep, cpu_val, gpu_val = record
        print(f"feature {idx}: series={series}, t={timestep}, cpu={cpu_val}, gpu={gpu_val}")
    if args.debug_feature is not None:
        if not 0 <= args.debug_feature < 6:
            raise ValueError("--debug-feature must be between 0 and 5")
        record = worst_idx[args.debug_feature]
        if record is None:
            print("debug window: no record available for requested feature")
        else:
            series, timestep, _cpu_val, _gpu_val = record
            _debug_window(prices, series, timestep, args.w2)


if __name__ == "__main__":
    main()

"""
Replays historical bars through the Vulkan qfeat tape so the runner sees
the GPU-derived ell stream without touching strategy semantics.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for path in (str(ROOT), str(PARENT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from trading.runner import run_bars
    from trading.signals.triadic import compute_triadic_state
    from trading.trading_io.prices import load_prices
    from trading.vk_qfeat import QFeatTape, build_feature_tape
    from trading.strategy.vulkan_tape_adapter import VulkanTapeAdapter
except ModuleNotFoundError:
    from runner import run_bars
    from signals.triadic import compute_triadic_state
    from trading_io.prices import load_prices
    from vk_qfeat import QFeatTape, build_feature_tape
    from strategy.vulkan_tape_adapter import VulkanTapeAdapter


def make_ts_map(ts: np.ndarray) -> dict[int, int]:
    return {int(t): i for i, t in enumerate(ts.astype(np.int64))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bars with Vulkan qfeat tape")
    parser.add_argument("--prices-csv", type=pathlib.Path, required=True)
    parser.add_argument("--tape", type=pathlib.Path, required=True)
    parser.add_argument("--w1", type=int, default=64)
    parser.add_argument("--w2", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--nan-squash", type=float, default=0.0)
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="vulkan")
    parser.add_argument("--shader", type=pathlib.Path, default=pathlib.Path("vulkan_shaders/qfeat.comp"))
    parser.add_argument("--spv", type=pathlib.Path, default=pathlib.Path("vulkan_shaders/qfeat.spv"))
    parser.add_argument("--vk-icd", type=str, default=None)
    parser.add_argument("--no-fp64-returns", action="store_true", help="Disable FP64 log-return path")
    parser.add_argument("--force", action="store_true", help="Overwrite tape if it exists")
    parser.add_argument("--reuse-tape", action="store_true", help="Reuse existing tape instead of rebuilding")
    parser.add_argument("--log", type=pathlib.Path, default=pathlib.Path("logs/trading_log_vulkan.csv"))
    parser.add_argument("--symbol", type=str, default="VKN")
    parser.add_argument("--tau-on", type=float, default=0.52, help="Entry threshold for ACT")
    parser.add_argument("--tau-off", type=float, default=0.47, help="Exit threshold for HOLD")
    args = parser.parse_args()

    if args.force and args.reuse_tape:
        raise SystemExit(
            "Invalid flags: --force and --reuse-tape cannot be used together. "
            "--force rebuilds and overwrites the tape; --reuse-tape preserves learned ell."
        )

    if args.force:
        print(
            "WARNING: --force will rebuild and overwrite the qfeat tape. "
            "Any learned ell in slot 6 will be LOST.",
            file=sys.stderr,
        )

    price, volume, ts = load_prices(args.prices_csv, return_time=True)
    state = compute_triadic_state(price)
    ts_int = pd.to_datetime(ts).astype("int64") if ts is not None else np.arange(price.size, dtype=np.int64)
    bars = pd.DataFrame(
        {
            "ts": ts_int,
            "close": price,
            "state": state,
            "volume": volume,
        }
    )

    if args.reuse_tape:
        if not args.tape.exists():
            raise SystemExit(f"{args.tape} not found for reuse")
        tape = QFeatTape.from_existing(str(args.tape))
        if tape.T != price.size:
            raise SystemExit(
                f"Tape length (T={tape.T}) does not match prices length (T={price.size}). "
                "Refuse to replay with --reuse-tape."
            )
    else:
        tape = build_feature_tape(
            prices=price[np.newaxis, :],
            out_path=str(args.tape),
            w1=args.w1,
            w2=args.w2,
            eps=args.eps,
            nan_squash=args.nan_squash,
            force=args.force,
            backend=args.backend,
            shader_path=str(args.shader),
            spv_path=str(args.spv),
            vk_icd=args.vk_icd,
            fp64_returns=not args.no_fp64_returns,
        )

    ts_map = make_ts_map(bars["ts"].to_numpy())
    adapter = VulkanTapeAdapter(tape=tape, series_index=0, ts_to_index=ts_map)

    def confidence_fn(ts: int, state: int):
        return adapter.update(ts, {"state": state})

    run_bars(
        bars,
        symbol=args.symbol,
        mode="bar",
        log_path=str(args.log),
        confidence_fn=confidence_fn,
        tau_conf_enter=args.tau_on,
        tau_conf_exit=args.tau_off,
    )


if __name__ == "__main__":
    main()

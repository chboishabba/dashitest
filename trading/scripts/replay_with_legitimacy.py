"""
Replay a price CSV through the triadic strategy using the learner stub for
legitimacy gating. Intended to validate hysteresis behavior end-to-end.
"""

import argparse
import os
import pathlib
import subprocess
import sys

import pandas as pd

try:
    from trading.signals.triadic import compute_triadic_state
    from trading.trading_io.prices import load_prices
    from trading.runner import run_bars
except ModuleNotFoundError:
    from signals.triadic import compute_triadic_state
    from trading_io.prices import load_prices
    from runner import run_bars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=pathlib.Path, help="Price CSV to replay")
    parser.add_argument("--log", default="logs/replay_legitimacy.csv", type=pathlib.Path, help="Output log CSV")
    parser.add_argument(
        "--stub-mode",
        default="schedule",
        choices=["schedule", "constant", "vol_proxy", "qfeat_var"],
        help="Legitimacy mode; qfeat_var uses quotient features for a real ℓ",
    )
    parser.add_argument("--stub-constant", default=0.5, type=float, help="ell value when stub-mode=constant or fallback")
    parser.add_argument("--tau-on", dest="tau_on", default=5, type=int, help="Windows ell must stay above theta_on to allow ACT")
    parser.add_argument("--tau-off", dest="tau_off", default=10, type=int, help="Windows ell must stay below theta_off to force HOLD")
    parser.add_argument("--theta-on", dest="theta_on", default=0.7, type=float, help="ell threshold to begin ACT persistence")
    parser.add_argument("--theta-off", dest="theta_off", default=0.3, type=float, help="ell threshold to begin HOLD persistence")
    parser.add_argument(
        "--plots",
        dest="plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit default timestamped plots from the generated log",
    )
    args = parser.parse_args()

    price, volume, ts = load_prices(args.csv, return_time=True)
    state = compute_triadic_state(price)
    if ts is None:
        ts = list(range(len(price)))
    bars = pd.DataFrame({"ts": ts, "close": price, "volume": volume, "state": state})
    # Ensure ts is numeric for runner; convert datetime64 to seconds since epoch.
    if pd.api.types.is_datetime64_any_dtype(bars["ts"]):
        bars["ts"] = pd.to_datetime(bars["ts"]).astype("int64") // 1_000_000_000

    df = run_bars(
        bars=bars,
        symbol=args.csv.stem.upper(),
        mode="bar",
        log_path=args.log,
        confidence_fn=None,
        tau_conf_enter=args.theta_on,
        tau_conf_exit=args.theta_off,
        use_stub_adapter=True,
        adapter_kwargs={
            "stub_mode": args.stub_mode,
            "stub_constant": args.stub_constant,
        },
    )
    print(f"[done] wrote {len(df)} rows to {args.log}")

    if args.plots:
        # Produce default timestamped plots to satisfy the "always emit images" requirement.
        venv_python = pathlib.Path(__file__).resolve().parent.parent.parent / "venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else sys.executable
        # Ensure parent-of-package is on PYTHONPATH so plot scripts can import trading.*
        base_root = pathlib.Path(__file__).resolve().parent.parent.parent
        existing = os.environ.get("PYTHONPATH", "")
        env_path = f"{base_root}{os.pathsep}{existing}" if existing else str(base_root)
        env = {**os.environ, "PYTHONPATH": env_path}
        scripts = [
            (
                "plot_hysteresis_phase.py",
                [
                    "--log",
                    str(args.log),
                    "--tau_on",
                    str(args.theta_on),
                    "--tau_off",
                    str(args.theta_off),
                    "--save",
                    "logs/hysteresis_phase.png",
                ],
            ),
            (
                "plot_legitimacy_margin.py",
                [
                    "--log",
                    str(args.log),
                    "--min_run_length",
                    "3",
                    "--max_flip_rate",
                    "0.2",
                    "--window",
                    "50",
                    "--save",
                    "logs/legitimacy_margin.png",
                ],
            ),
            (
                "plot_acceptability.py",
                [
                    "--log",
                    str(args.log),
                    "--time_bins",
                    "200",
                    "--act_bins",
                    "40",
                    "--save",
                    "logs/acceptable.png",
                ],
            ),
        ]
        for script, extra_args in scripts:
            script_path = pathlib.Path(__file__).resolve().parent / script
            if not script_path.exists():
                print(f"[warn] plot script missing: {script_path}")
                continue
            cmd = [python, str(script_path), *extra_args]
            try:
                subprocess.run(cmd, check=False, env=env, stdout=sys.stdout, stderr=sys.stderr)
            except Exception as exc:
                print(f"[warn] plot failed for {script}: {exc}")


if __name__ == "__main__":
    main()

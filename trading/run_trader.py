"""
run_trader.py
--------------
Streams a simple ternary trading loop over cached Stooq data and logs
to logs/trading_log.csv for the training_dashboard to visualize.

Usage:
  python trading/run_trader.py
Then in another terminal:
  python trading/training_dashboard.py --log logs/trading_log.csv --refresh 0.5
"""

import time
import pathlib
import os
import sys
import subprocess
import pandas as pd
import numpy as np

from engine.loop import (
    CONTRACT_MULT,
    DEFAULT_RISK_FRAC,
    EDGE_EMA_ALPHA,
    EST_TAX_RATE,
    GOAL_CASH_X,
    GOAL_EPS,
    LOG,
    MDL_NOISE_MULT,
    MDL_SWITCH_PENALTY,
    MDL_TRADE_PENALTY,
    SIGMA_TARGET,
    THESIS_A_MAX,
    THESIS_BENCHMARK_X,
    THESIS_COOLDOWN,
    THESIS_DEPTH_MAX,
    THESIS_MEMORY_DEFAULT,
    THESIS_PBAD_HI,
    THESIS_PBAD_LO,
    THESIS_STRESS_HI,
    THESIS_STRESS_LO,
    THESIS_TC_K,
    run_trading_loop,
)
from signals.triadic import compute_triadic_state
from trading_io.prices import find_btc_csv, find_stooq_csv, list_price_csvs, load_prices

def main(
    max_steps=None,
    max_trades=None,
    max_seconds=None,
    sleep_s=0.0,
    risk_frac: float = DEFAULT_RISK_FRAC,
    contract_mult: float = CONTRACT_MULT,
    sigma_target: float = SIGMA_TARGET,
    est_tax_rate: float = EST_TAX_RATE,
    goal_cash_x: float = GOAL_CASH_X,
    goal_eps: float = GOAL_EPS,
    mdl_noise_mult: float = MDL_NOISE_MULT,
    mdl_switch_penalty: float = MDL_SWITCH_PENALTY,
    mdl_trade_penalty: float = MDL_TRADE_PENALTY,
    progress_every: int = 0,
    log_level: str = "info",
    run_all: bool = False,
    raw_root: str = "data/raw",
    log_prefix: str = "logs/trading_log",
    inter_run_sleep: float = 0.5,
    log_combined: bool = False,
    edge_gate: bool = False,
    edge_decay: float = 0.9,
    edge_alpha: float = EDGE_EMA_ALPHA,
    thesis_depth_max: int = THESIS_DEPTH_MAX,
    thesis_memory: bool = THESIS_MEMORY_DEFAULT,
    thesis_a_max: int = THESIS_A_MAX,
    thesis_cooldown: int = THESIS_COOLDOWN,
    thesis_pbad_lo: float = THESIS_PBAD_LO,
    thesis_pbad_hi: float = THESIS_PBAD_HI,
    thesis_stress_lo: float = THESIS_STRESS_LO,
    thesis_stress_hi: float = THESIS_STRESS_HI,
    tc_k: float = THESIS_TC_K,
    benchmark_x: float = THESIS_BENCHMARK_X,
    geometry_plots: bool = True,
    geometry_overlay: bool = True,
    geometry_simplex: bool = True,
    geometry_dir: str = "logs/geometry",
):
    run_ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def slugify(value: str) -> str:
        keep = []
        for ch in value:
            if ch.isalnum() or ch in ("-", "_"):
                keep.append(ch)
            else:
                keep.append("_")
        return "".join(keep)

    def emit_geometry(log_path: pathlib.Path, source_name: str):
        if not geometry_plots:
            return
        geometry_root = pathlib.Path(geometry_dir)
        geometry_root.mkdir(parents=True, exist_ok=True)
        prefix = geometry_root / f"{slugify(source_name)}_{run_ts}"
        cmd = [
            sys.executable,
            "scripts/plot_decision_geometry.py",
            "--csv",
            str(log_path),
            "--save-prefix",
            str(prefix),
            "--no-show",
        ]
        if geometry_overlay:
            cmd.append("--overlay")
        if geometry_simplex:
            cmd.append("--simplex")
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f".{os.pathsep}{existing}" if existing else "."
        try:
            subprocess.run(cmd, check=False, env=env)
        except Exception as exc:
            print(f"[warn] geometry plots failed for {log_path}: {exc}")

    source = "stooq"
    try:
        if run_all:
            raw_root_path = pathlib.Path(raw_root)
            csv_paths = list_price_csvs(raw_root_path)
            if not csv_paths:
                raise FileNotFoundError(f"No CSVs found under {raw_root_path}.")
            combined_path = None
            if log_combined:
                combined_path = pathlib.Path(f"{log_prefix}_all.csv")
                combined_path.parent.mkdir(parents=True, exist_ok=True)
                combined_path.unlink(missing_ok=True)
            for idx, csv_path in enumerate(csv_paths, 1):
                try:
                    price, volume, ts = load_prices(csv_path, return_time=True)
                except Exception as exc:
                    print(f"[skip] {csv_path} ({exc})")
                    continue
                source = f"{csv_path.parent.name}:{csv_path.stem}"
                if log_combined and combined_path is not None:
                    log_path = combined_path
                    trade_log_path = pathlib.Path(f"{log_prefix}_trades_all.csv")
                else:
                    log_path = pathlib.Path(f"{log_prefix}_{csv_path.stem}.csv")
                    trade_log_path = pathlib.Path(f"{log_prefix}_trades_{csv_path.stem}.csv")
                print(f"[run {idx}/{len(csv_paths)}] {csv_path} -> {log_path}")
                run_trading_loop(
                    price=price,
                    volume=volume,
                    source=source,
                    time_index=ts,
                    max_steps=max_steps,
                    max_trades=max_trades,
                    max_seconds=max_seconds,
                    sleep_s=sleep_s,
                    risk_frac=risk_frac,
                    contract_mult=contract_mult,
                    sigma_target=sigma_target,
                    est_tax_rate=est_tax_rate,
                    goal_cash_x=goal_cash_x,
                    goal_eps=goal_eps,
                    mdl_noise_mult=mdl_noise_mult,
                    mdl_switch_penalty=mdl_switch_penalty,
                    mdl_trade_penalty=mdl_trade_penalty,
                    log_path=log_path,
                    trade_log_path=trade_log_path,
                    progress_every=progress_every,
                    log_level=log_level,
                    log_append=bool(log_combined),
                    tape_id=source if log_combined else None,
                    edge_gate=edge_gate,
                    edge_decay=edge_decay,
                    edge_ema_alpha=edge_alpha,
                    thesis_depth_max=thesis_depth_max,
                    thesis_memory=thesis_memory,
                    thesis_a_max=thesis_a_max,
                    thesis_cooldown=thesis_cooldown,
                    thesis_pbad_lo=thesis_pbad_lo,
                    thesis_pbad_hi=thesis_pbad_hi,
                    thesis_stress_lo=thesis_stress_lo,
                    thesis_stress_hi=thesis_stress_hi,
                    tc_k=tc_k,
                    benchmark_x=benchmark_x,
                )
                if not log_combined:
                    emit_geometry(log_path, source)
                if inter_run_sleep > 0:
                    time.sleep(inter_run_sleep)
            if log_combined and combined_path is not None:
                emit_geometry(combined_path, "combined")
            return

        csv_path = find_btc_csv()
        if csv_path is not None:
            price, volume, ts = load_prices(csv_path, return_time=True)
            source = "btc"
            print(f"Using BTC data: {csv_path}")
        else:
            csv_path = find_stooq_csv()
            price, volume, ts = load_prices(csv_path, return_time=True)
            print(f"Using Stooq data: {csv_path}")
    except Exception as e:
        print(f"Falling back to synthetic prices: {e}")
        source = "synthetic"
        rng = np.random.default_rng(0)
        steps = rng.normal(loc=0.0, scale=0.01, size=1000)
        price = 100 + np.cumsum(steps)
        volume = np.ones_like(price) * 1e6
        ts = None

    run_trading_loop(
        price=price,
        volume=volume,
        source=source,
        time_index=ts,
        max_steps=max_steps,
        max_trades=max_trades,
        max_seconds=max_seconds,
        sleep_s=sleep_s,
        risk_frac=risk_frac,
        contract_mult=contract_mult,
        sigma_target=sigma_target,
        est_tax_rate=est_tax_rate,
        goal_cash_x=goal_cash_x,
        goal_eps=goal_eps,
        mdl_noise_mult=mdl_noise_mult,
        mdl_switch_penalty=mdl_switch_penalty,
        mdl_trade_penalty=mdl_trade_penalty,
        log_path=LOG,
        trade_log_path=pathlib.Path("logs/trade_log.csv"),
        progress_every=progress_every,
        log_level=log_level,
        edge_gate=edge_gate,
        edge_decay=edge_decay,
        edge_ema_alpha=edge_alpha,
        thesis_depth_max=thesis_depth_max,
        thesis_memory=thesis_memory,
        thesis_a_max=thesis_a_max,
        thesis_cooldown=thesis_cooldown,
        thesis_pbad_lo=thesis_pbad_lo,
        thesis_pbad_hi=thesis_pbad_hi,
        thesis_stress_lo=thesis_stress_lo,
        thesis_stress_hi=thesis_stress_hi,
        tc_k=tc_k,
        benchmark_x=benchmark_x,
    )
    emit_geometry(LOG, source)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None, help="Limit steps (debug).")
    ap.add_argument("--max-trades", type=int, default=None, help="Stop after N trades.")
    ap.add_argument("--max-seconds", type=float, default=None, help="Stop after N seconds per asset.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep per step.")
    ap.add_argument("--risk-frac", type=float, default=DEFAULT_RISK_FRAC, help="Risk fraction.")
    ap.add_argument("--sigma-target", type=float, default=SIGMA_TARGET, help="Target vol.")
    ap.add_argument("--est-tax-rate", type=float, default=EST_TAX_RATE, help="Estimated tax rate on realized PnL.")
    ap.add_argument("--goal-cash-x", type=float, default=GOAL_CASH_X, help="Spendable cash target.")
    ap.add_argument("--goal-eps", type=float, default=GOAL_EPS, help="Tail fraction for expected shortfall.")
    ap.add_argument("--mdl-noise-mult", type=float, default=MDL_NOISE_MULT, help="Sigma multiplier for active-trit threshold.")
    ap.add_argument("--mdl-switch-penalty", type=float, default=MDL_SWITCH_PENALTY, help="Side cost for state switches.")
    ap.add_argument("--mdl-trade-penalty", type=float, default=MDL_TRADE_PENALTY, help="Side cost for executed trades.")
    ap.add_argument("--progress-every", type=int, default=0, help="Print progress every N steps.")
    ap.add_argument("--all", action="store_true", help="Run over all CSVs under data/raw.")
    ap.add_argument("--raw-root", type=str, default="data/raw", help="Root directory to scan when --all set.")
    ap.add_argument(
        "--log-prefix",
        type=str,
        default="logs/trading_log",
        help="Log prefix when --all set (per-file log suffix added).",
    )
    ap.add_argument("--inter-run-sleep", type=float, default=0.5, help="Pause between runs when --all set.")
    ap.add_argument(
        "--log-combined",
        action="store_true",
        help="Write all tapes into a single log file when --all set.",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["quiet", "info", "trades", "verbose"],
        help="Output verbosity.",
    )
    ap.add_argument("--edge-gate", action="store_true", help="Enable edge-based cap decay gate.")
    ap.add_argument("--edge-decay", type=float, default=0.9, help="Cap multiplier when edge gate triggers.")
    ap.add_argument("--edge-alpha", type=float, default=EDGE_EMA_ALPHA, help="EMA alpha for edge metric.")
    ap.add_argument("--thesis-depth-max", type=int, default=THESIS_DEPTH_MAX, help="Max thesis memory depth.")
    ap.add_argument("--thesis-memory", action="store_true", help="Enable thesis memory FSM.")
    ap.add_argument("--thesis-a-max", type=int, default=THESIS_A_MAX, help="Max thesis age.")
    ap.add_argument("--thesis-cooldown", type=int, default=THESIS_COOLDOWN, help="Cooldown steps after thesis exit.")
    ap.add_argument("--thesis-pbad-lo", type=float, default=THESIS_PBAD_LO, help="Low p_bad threshold for thesis risk trit.")
    ap.add_argument("--thesis-pbad-hi", type=float, default=THESIS_PBAD_HI, help="High p_bad threshold for thesis risk trit.")
    ap.add_argument("--thesis-stress-lo", type=float, default=THESIS_STRESS_LO, help="Low stress threshold for thesis risk trit.")
    ap.add_argument("--thesis-stress-hi", type=float, default=THESIS_STRESS_HI, help="High stress threshold for thesis risk trit.")
    ap.add_argument("--tc-k", type=float, default=THESIS_TC_K, help="Transaction cost coefficient for reward_regret.")
    ap.add_argument("--benchmark-x", type=float, default=THESIS_BENCHMARK_X, help="Benchmark exposure for reward_regret.")
    ap.add_argument("--geometry-dir", type=str, default="logs/geometry", help="Output directory for geometry plots.")
    ap.add_argument("--geometry-simplex", action="store_true", help="Render simplex plot in geometry output.")
    ap.add_argument("--no-geometry-simplex", dest="geometry_simplex", action="store_false")
    ap.add_argument("--geometry-overlay", action="store_true", help="Render overlay plot in geometry output.")
    ap.add_argument("--no-geometry-overlay", dest="geometry_overlay", action="store_false")
    ap.add_argument("--geometry-plots", dest="geometry_plots", action="store_true", help="Emit geometry plots.")
    ap.add_argument("--no-geometry-plots", dest="geometry_plots", action="store_false", help="Skip geometry plots.")
    ap.set_defaults(geometry_plots=True, geometry_overlay=True, geometry_simplex=True)
    args = ap.parse_args()
    main(
        max_steps=args.max_steps,
        max_trades=args.max_trades,
        max_seconds=args.max_seconds,
        sleep_s=args.sleep,
        risk_frac=args.risk_frac,
        sigma_target=args.sigma_target,
        est_tax_rate=args.est_tax_rate,
        goal_cash_x=args.goal_cash_x,
        goal_eps=args.goal_eps,
        mdl_noise_mult=args.mdl_noise_mult,
        mdl_switch_penalty=args.mdl_switch_penalty,
        mdl_trade_penalty=args.mdl_trade_penalty,
        progress_every=args.progress_every,
        log_level=args.log_level,
        run_all=args.all,
        raw_root=args.raw_root,
        log_prefix=args.log_prefix,
        inter_run_sleep=args.inter_run_sleep,
        log_combined=args.log_combined,
        edge_gate=args.edge_gate,
        edge_decay=args.edge_decay,
        edge_alpha=args.edge_alpha,
        thesis_depth_max=args.thesis_depth_max,
        thesis_memory=args.thesis_memory,
        thesis_a_max=args.thesis_a_max,
        thesis_cooldown=args.thesis_cooldown,
        thesis_pbad_lo=args.thesis_pbad_lo,
        thesis_pbad_hi=args.thesis_pbad_hi,
        thesis_stress_lo=args.thesis_stress_lo,
        thesis_stress_hi=args.thesis_stress_hi,
        tc_k=args.tc_k,
        benchmark_x=args.benchmark_x,
        geometry_plots=args.geometry_plots,
        geometry_overlay=args.geometry_overlay,
        geometry_simplex=args.geometry_simplex,
        geometry_dir=args.geometry_dir,
    )

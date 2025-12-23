"""
run_all_two_pointO.py
---------------------
One-stop orchestrator to run:
 1) Multi-market trading summaries (reuse run_trader loop; no dashboard).
 2) Tau sweep with PR/PnL metrics (optional live plots).
 3) CA epistemic tape preview (market-driven CA with incremental stats).

Usage (common):
  PYTHONPATH=. python run_all_two_pointO.py --csv data/raw/stooq/btc_intraday_1s.csv --live-sweep --run-ca --markets

Flags:
  --markets            Run trading loop across cached markets (like run_all).
  --csv PATH           Price CSV for sweep/CA (default btc_intraday_1s).
  --tau-on FLOAT       tau_on for sweep (default 0.5)
  --tau-off LIST       tau_off values to sweep (default 0.30 0.35 0.40 0.45 0.25 0.20 0.15)
  --live-sweep         Show live PR/PnL Pareto plots while sweeping.
  --run-ca             Run CA tape on the same CSV and stream incremental stats.
  --ca-report-every N  Print CA stats every N steps (default 0 = off).
  --max-steps N        Optional cap for market runs to speed up.
"""

import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from run_trader import run_trading_loop, load_prices, compute_triadic_state
from scripts.run_bars_btc import confidence_from_persistence
import run_all  # for discover_markets
from scripts import sweep_tau_conf
from scripts import ca_epistemic_tape


# --- Trading summaries -----------------------------------------------------

def run_market_summaries(max_steps=None, progress_every=0):
    markets = run_all.discover_markets()
    summaries = []
    for m in markets:
        name = m["name"]
        print(f"\n=== Market: {name} ===")
        summary, _ = run_trading_loop(
            price=m["price"],
            volume=m["volume"],
            source=name,
            sleep_s=0.0,
            max_steps=max_steps,
            log_path=None,
            progress_every=progress_every or (1000 if max_steps is None else max(1, max_steps // 10)),
        )
        summaries.append(summary)
    if summaries:
        df = pd.DataFrame(summaries)
        print("\n=== Market summaries ===")
        print(df[["source", "pnl", "max_drawdown", "trades", "steps", "hold_pct"]])
    return summaries


# --- Sweep with live plots -------------------------------------------------

def run_tau_sweep(csv_path, tau_on, tau_off_list, precision_floor=0.8, live=False, bins=20):
    rows = []
    bin_rows = []
    if live:
        plt.ion()
        fig, (ax_pr, ax_pnl) = plt.subplots(1, 2, figsize=(10, 4))
    for tau_off in tau_off_list:
        metrics, df_run = sweep_tau_conf.run_once(csv_path, tau_on, tau_off)
        rows.append(metrics)
        print(
            f"tau_off={tau_off:.2f}  acceptable={metrics['acceptable_pct']:.3f}  "
            f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
            f"pnl={metrics['pnl_net']:.4f}  max_dd={metrics['max_dd']:.4f}  "
            f"turnover={metrics['turnover']:.4f}  trades={metrics['trades']}  fees={metrics['fees']:.6f}"
        )
        if not np.isnan(metrics["precision"]) and metrics["precision"] < precision_floor:
            print("Precision dropped below floor; stopping sweep.")
            break
        # engagement surface (optional)
        for row in sweep_tau_conf.engagement_bins(df_run, bins=bins):
            row["tau_off"] = tau_off
            bin_rows.append(row)
        # live plots
        if live:
            df_live = pd.DataFrame(rows)
            ax_pr.cla()
            ax_pnl.cla()
            ax_pr.scatter(df_live["recall"], df_live["precision"], c="tab:blue")
            for _, r in df_live.iterrows():
                ax_pr.annotate(
                    f"tau={r['tau_off']:.2f}\nP={r['pnl_net']:.1f}\nDD={r['max_dd']:.1f}",
                    (r["recall"], r["precision"]),
                    fontsize=7,
                )
            ax_pr.set_xlabel("Recall (P(ACT | acceptable))")
            ax_pr.set_ylabel("Precision (P(acceptable | ACT))")
            ax_pr.set_xlim(0, 1)
            ax_pr.set_ylim(0, 1)
            ax_pr.set_title("PR annotated with PnL/DD")
            ax_pnl.scatter(df_live["max_dd"], df_live["pnl_net"], c="tab:green")
            for _, r in df_live.iterrows():
                ax_pnl.annotate(
                    f"tau={r['tau_off']:.2f}\nturn={r['turnover']:.2f}",
                    (r["max_dd"], r["pnl_net"]),
                    fontsize=7,
                )
            ax_pnl.set_xlabel("Max drawdown")
            ax_pnl.set_ylabel("Net PnL")
            ax_pnl.set_title("PnL vs Max DD")
            plt.tight_layout()
            plt.pause(0.01)
    return rows, bin_rows


# --- CA tape preview -------------------------------------------------------

def run_ca_preview(csv_path, width=128, report_every=0):
    price, volume = ca_epistemic_tape.load_price_series(csv_path)
    feats = ca_epistemic_tape.build_features(price, volume)
    snaps, stats, hist = ca_epistemic_tape.run_ca_tape(
        feats, width=width, seed=0, report_every=report_every
    )
    # minimal summary
    print(
        f"CA tape run: steps={len(hist)} "
        f"mean act={np.mean(stats['act']):.3f} "
        f"mean ban={np.mean(stats['ban']):.3f} "
        f"mean tension={np.mean(stats['tension_mean']):.3f}"
    )
    # show plots
    ca_epistemic_tape.plot_snapshots(snaps)
    ca_epistemic_tape.plot_stats(stats)
    ca_epistemic_tape.plot_multiscale(hist)


# --- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", action="store_true", help="Run multi-market trading summaries.")
    ap.add_argument("--csv", type=str, default="data/raw/stooq/btc_intraday_1s.csv", help="Price CSV for sweep/CA.")
    ap.add_argument("--tau-on", type=float, default=0.5, help="tau_on for sweep.")
    ap.add_argument(
        "--tau-off",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.25, 0.20, 0.15],
        help="tau_off values to sweep.",
    )
    ap.add_argument("--precision-floor", type=float, default=0.8, help="Stop sweep if precision below this.")
    ap.add_argument("--live-sweep", action="store_true", help="Show live PR/PnL plots during sweep.")
    ap.add_argument("--run-ca", action="store_true", help="Run CA tape preview.")
    ap.add_argument("--ca-width", type=int, default=128, help="Lag width for CA tape.")
    ap.add_argument(
        "--ca-report-every",
        type=int,
        default=0,
        help="If >0, print incremental CA stats every N steps.",
    )
    ap.add_argument(
        "--market-progress-every",
        type=int,
        default=0,
        help="If >0, print trading loop progress every N steps per market.",
    )
    ap.add_argument("--max-steps", type=int, default=None, help="Optional cap for market runs.")
    args = ap.parse_args()

    if args.markets:
        run_market_summaries(max_steps=args.max_steps, progress_every=args.market_progress_every)

    if args.csv:
        csv_path = pathlib.Path(args.csv)
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
        else:
            print("\n=== Tau sweep ===")
            rows, bin_rows = run_tau_sweep(
                csv_path,
                tau_on=args.tau_on,
                tau_off_list=args.tau_off,
                precision_floor=args.precision_floor,
                live=args.live_sweep,
            )
            if rows:
                df = pd.DataFrame(rows)
                print("\nSweep table (head):")
                print(df.head())

            if args.run_ca:
                print("\n=== CA tape preview ===")
                run_ca_preview(csv_path, width=args.ca_width, report_every=args.ca_report_every)


if __name__ == "__main__":
    main()

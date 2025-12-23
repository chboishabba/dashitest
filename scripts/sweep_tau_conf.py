"""
Sweep hysteresis width (tau_off) while holding tau_on fixed.
Reports precision/recall of ACT vs acceptable; PnL is intentionally ignored.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from runner import run_bars
from run_trader import load_prices, compute_triadic_state
from scripts.run_bars_btc import confidence_from_persistence


def compute_metrics(df: pd.DataFrame):
    acceptable = df["acceptable"].astype(bool)
    act = df["action"] != 0
    acceptable_pct = float(acceptable.mean()) if len(df) else float("nan")
    act_hits = (acceptable & act).sum()
    act_count = act.sum()
    acceptable_count = acceptable.sum()
    precision = float(act_hits) / float(act_count) if act_count > 0 else float("nan")
    recall = float(act_hits) / float(acceptable_count) if acceptable_count > 0 else float("nan")
    hold_pct = float(df["hold"].mean()) if "hold" in df and len(df) else float("nan")
    return {
        "acceptable_pct": acceptable_pct,
        "precision": precision,
        "recall": recall,
        "act_bars": int(act_count),
        "hold_pct": hold_pct,
    }


def compute_pnl_metrics(df: pd.DataFrame):
    """
    Basic PnL audit metrics for a single run.
    Expects df with columns: pnl (net), fee, slippage, fill, exposure, price.
    """
    if df is None or df.empty:
        return {
            "pnl_net": float("nan"),
            "pnl_gross": float("nan"),
            "max_dd": float("nan"),
            "turnover": float("nan"),
            "trades": 0,
            "fees": float("nan"),
            "impact": float("nan"),
            "mean_ret": float("nan"),
            "std_ret": float("nan"),
        }
    equity = df["pnl"] + 1.0
    ret = equity.pct_change().fillna(0.0)
    mean_ret = float(ret.mean())
    std_ret = float(ret.std())
    pnl_net = float(df["pnl"].iloc[-1])
    pnl_gross = pnl_net + float(df["fee"].fillna(0).sum())  # fees already subtract from pnl
    running_max = equity.cummax()
    dd = (equity - running_max).min()
    max_dd = float(dd)
    turnover = float(df["fill"].abs().sum()) if "fill" in df else float("nan")
    trades = int((df["fill"] != 0).sum()) if "fill" in df else 0
    fees = float(df["fee"].fillna(0).sum())
    impact = float(df["slippage"].abs().sum()) if "slippage" in df else float("nan")
    return {
        "pnl_net": pnl_net,
        "pnl_gross": pnl_gross,
        "max_dd": max_dd,
        "turnover": turnover,
        "trades": trades,
        "fees": fees,
        "impact": impact,
        "mean_ret": mean_ret,
        "std_ret": std_ret,
    }


def engagement_bins(df: pd.DataFrame, bins: int = 20):
    """
    Return per-bin engagement rates: list of (bin_center, engagement_rate).
    Uses acceptable=True rows only; engagement=action!=0.
    """
    if df is None or df.empty or "actionability" not in df or "action" not in df or "acceptable" not in df:
        return []
    sub = df[df["acceptable"] == True]  # noqa: E712
    if sub.empty:
        return []
    actionability = pd.to_numeric(sub["actionability"], errors="coerce")
    act = (sub["action"] != 0).astype(int)
    mask = np.isfinite(actionability)
    actionability = actionability[mask]
    act = act.loc[actionability.index]
    if actionability.empty:
        return []
    bins = max(5, bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    cat = pd.cut(actionability, edges, include_lowest=True, labels=labels)
    grouped = act.groupby(cat).mean()
    rows = []
    for center, rate in grouped.items():
        rows.append({"bin_center": float(center), "engagement": float(rate)})
    return rows


def run_once(csv: Path, tau_on: float, tau_off: float):
    price, _ = load_prices(csv)
    ts = np.arange(len(price))
    state = compute_triadic_state(price)
    bars = pd.DataFrame({"ts": ts, "close": price, "state": state})
    rets = np.diff(price, prepend=price[0])
    vol = pd.Series(rets).rolling(50).std().to_numpy()
    conf_seq = confidence_from_persistence(
        state, run_scale=30, ret_vol=vol, vol_thresh=np.nanpercentile(vol, 80)
    )

    def conf_fn(t, s):
        idx = int(t)
        if idx < 0 or idx >= len(conf_seq):
            return 1.0
        return conf_seq[idx]

    df = run_bars(
        bars,
        symbol="BTCUSDT",
        mode="bar",
        log_path=None,  # PnL captured in df; no disk I/O
        confidence_fn=conf_fn,
        tau_conf_enter=tau_on,
        tau_conf_exit=tau_off,
    )
    metrics = compute_metrics(df)
    metrics.update(compute_pnl_metrics(df))
    metrics.update({"tau_on": tau_on, "tau_off": tau_off})
    return metrics, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/stooq/btc_intraday.csv"),
        help="Price CSV (Stooq BTC intraday expected).",
    )
    ap.add_argument("--tau_on", type=float, default=0.5, help="Entry threshold (fixed).")
    ap.add_argument(
        "--tau_off",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.25, 0.20, 0.15],
        help="Exit thresholds to sweep (must be <= tau_on).",
    )
    ap.add_argument(
        "--precision_floor",
        type=float,
        default=0.8,
        help="Stop sweep if precision falls below this (drop in legitimacy).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path for precision/recall curve.",
    )
    ap.add_argument(
        "--out_pnl",
        type=Path,
        default=None,
        help="Optional CSV path for PnL audit metrics per tau_off.",
    )
    ap.add_argument(
        "--out_bins",
        type=Path,
        default=None,
        help="Optional CSV for engagement surface (long form: tau_off, bin_center, engagement).",
    )
    ap.add_argument("--bins", type=int, default=20, help="Bins for actionability engagement surface.")
    ap.add_argument(
        "--live-plot",
        action="store_true",
        help="Show incremental PR and PnL-vs-DD plots as the sweep runs.",
    )
    args = ap.parse_args()

    if args.tau_on < max(args.tau_off):
        raise SystemExit("tau_on must be >= all tau_off values for hysteresis.")
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv} (run data_downloader.py)")

    rows = []
    bin_rows = []
    if args.live_plot:
        plt.ion()
        fig, (ax_pr, ax_pnl) = plt.subplots(1, 2, figsize=(10, 4))

    for tau_off in args.tau_off:
        metrics, df_run = run_once(args.csv, args.tau_on, tau_off)
        rows.append(metrics)
        print(
            f"tau_off={tau_off:.2f}  acceptable={metrics['acceptable_pct']:.3f}  "
            f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
            f"act_bars={metrics['act_bars']}  hold%={metrics['hold_pct']:.3f}  "
            f"pnl={metrics['pnl_net']:.4f}  max_dd={metrics['max_dd']:.4f}  trades={metrics['trades']}  "
            f"turnover={metrics['turnover']:.4f}  fees={metrics['fees']:.6f}"
        )
        if not np.isnan(metrics["precision"]) and metrics["precision"] < args.precision_floor:
            print("Precision dropped below floor; stopping sweep.")
            break
        # compute per-bin engagement for this run if requested
        if args.out_bins:
            for row in engagement_bins(df_run, bins=args.bins):
                row["tau_off"] = tau_off
                bin_rows.append(row)

        # live plots
        if args.live_plot:
            df_live = pd.DataFrame(rows)
            ax_pr.cla()
            ax_pnl.cla()
            # PR plot annotated with net pnl and max dd
            ax_pr.scatter(df_live["recall"], df_live["precision"], c="tab:blue")
            for _, r in df_live.iterrows():
                ax_pr.annotate(
                    f"tau={r['tau_off']:.2f}\nP={r['pnl_net']:.1f}\nDD={r['max_dd']:.1f}",
                    (r["recall"], r["precision"]),
                    fontsize=7,
                )
            ax_pr.set_xlabel("Recall (P(ACT | acceptable))")
            ax_pr.set_ylabel("Precision (P(acceptable | ACT))")
            ax_pr.set_title("PR curve (annotated with PnL, DD)")
            ax_pr.set_xlim(0, 1)
            ax_pr.set_ylim(0, 1)
            # PnL vs Max DD Pareto
            ax_pnl.scatter(df_live["max_dd"], df_live["pnl_net"], c="tab:green", s=40)
            for _, r in df_live.iterrows():
                ax_pnl.annotate(
                    f"tau={r['tau_off']:.2f}\nturn={r['turnover']:.2f}",
                    (r["max_dd"], r["pnl_net"]),
                    fontsize=7,
                )
            ax_pnl.set_xlabel("Max drawdown")
            ax_pnl.set_ylabel("Net PnL")
            ax_pnl.set_title("PnL vs Max DD (size=turnover)")
            plt.tight_layout()
            plt.pause(0.01)
    df = pd.DataFrame(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote sweep metrics to {args.out}")
    if args.out_pnl:
        args.out_pnl.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_pnl, index=False)
        print(f"Wrote PnL audit metrics to {args.out_pnl}")
    if args.out_bins and bin_rows:
        args.out_bins.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bin_rows).to_csv(args.out_bins, index=False)
        print(f"Wrote engagement surface to {args.out_bins}")


if __name__ == "__main__":
    main()

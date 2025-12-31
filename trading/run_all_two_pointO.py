"""
run_all_two_pointO.py
---------------------
One-stop orchestrator to run:
 1) Multi-market trading summaries (reuse run_trader loop; no dashboard).
 2) Tau sweep with PR/PnL metrics (optional live plots).
 3) CA epistemic tape preview (market-driven CA with incremental stats).

Usage (common):
  PYTHONPATH=. python trading/run_all_two_pointO.py --csv data/raw/stooq/btc_intraday_1s.csv --live-sweep --run-ca --markets

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

import requests
import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local imports
try:
    from trading.run_trader import run_trading_loop, load_prices, compute_triadic_state
    from trading.scripts.run_bars_btc import confidence_from_persistence
    from trading import run_all  # for discover_markets
    from trading.scripts import sweep_tau_conf
    from trading.scripts import ca_epistemic_tape
    from trading.scripts.emit_news_windows import contiguous_windows
    from trading.scripts.news_slice import gdelt_fetch
except ModuleNotFoundError:
    from run_trader import run_trading_loop, load_prices, compute_triadic_state
    from scripts.run_bars_btc import confidence_from_persistence
    import run_all  # for discover_markets
    from scripts import sweep_tau_conf
    from scripts import ca_epistemic_tape
    from scripts.emit_news_windows import contiguous_windows
    from scripts.news_slice import gdelt_fetch


# --- Trading summaries -----------------------------------------------------

def run_market_summaries(max_steps=None, progress_every=0, emit_news=False, news_kwargs=None):
    markets = run_all.discover_markets()
    summaries = []
    for m in markets:
        name = m["name"]
        print(f"\n=== Market: {name} ===")
        log_path = pathlib.Path(f"logs/trading_log_{name}.csv")
        summary, _ = run_trading_loop(
            price=m["price"],
            volume=m["volume"],
            source=name,
            time_index=m.get("time"),
            sleep_s=0.0,
            max_steps=max_steps,
            log_path=log_path,
            progress_every=progress_every or (1000 if max_steps is None else max(1, max_steps // 10)),
        )
        summaries.append(summary)
        if emit_news:
            emit_news_windows(log_path, **(news_kwargs or {}))
    if summaries:
        df = pd.DataFrame(summaries)
        print("\n=== Market summaries ===")
        print(df[["source", "pnl", "max_drawdown", "trades", "steps", "hold_pct", "p_bad_mean", "bad_rate"]])
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
                    f"tau={r['tau_off']:.2f}\nturn={r['turnover']:.2f}\nedge/turn={r['edge_per_turnover']:.4f}",
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


# --- News windows from bad flags ------------------------------------------

def emit_news_windows(
    log_path,
    p_bad_hi=0.7,
    max_gap_min=15.0,
    pad_min=60.0,
    query="",
    maxrows=250,
    max_days=5,
    max_failures=3,
    max_windows=20,
):
    """
    Detect contiguous bad windows in the trading log and fetch GDELT events (no API key).
    Prints a short summary to stdout and writes CSVs under logs/news_events/.
    To avoid spamming GDELT, windows are grouped by UTC day; one fetch per day (capped by max_days).
    """
    GDELT_MIN = pd.Timestamp("2015-06-01", tz="UTC")
    log_path = pathlib.Path(log_path)
    if not log_path.exists():
        print(f"[news] log not found: {log_path}")
        return
    df = pd.read_csv(log_path)
    if "ts" not in df.columns:
        print(f"[news] missing ts column in {log_path}; rerun trader with timestamp support.")
        return
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if df["ts"].notna().any():
        ts_min = df["ts"].min()
        ts_max = df["ts"].max()
        if ts_max < GDELT_MIN:
            print(f"[news] {log_path.stem}: all data end before GDELT coverage ({GDELT_MIN.date()}); skipping news fetch.")
            return
    mask = (df["bad_flag"] == 1) & (df["p_bad"] >= p_bad_hi)
    triggers = df.loc[mask, "ts"].dropna()
    if triggers.empty:
        print(f"[news] no bad windows found in {log_path.name}")
        return
    windows = contiguous_windows(triggers, pd.Timedelta(minutes=max_gap_min))
    pad = pd.Timedelta(minutes=pad_min)
    out_dir = pathlib.Path("logs/news_events")
    out_dir.mkdir(parents=True, exist_ok=True)
    # prefilter windows within coverage and cap total windows by trigger count
    filtered = []
    skipped_pre = 0
    for idx, (start, end) in enumerate(windows):
        w_start, w_end = start - pad, end + pad
        if w_end < GDELT_MIN:
            skipped_pre += 1
            continue
        w_start = max(w_start, GDELT_MIN)
        trig_count = ((triggers >= start) & (triggers <= end)).sum()
        window_mask = (df["ts"] >= start) & (df["ts"] <= end)
        p_bad_sum = float(df.loc[window_mask, "p_bad"].sum(skipna=True))
        filtered.append((idx, w_start, w_end, start, end, trig_count, p_bad_sum))

    now = pd.Timestamp.utcnow()

    if not filtered:
        msg = f"[news] no windows within GDELT coverage for {log_path.stem}"
        if skipped_pre:
            msg += f" (skipped {skipped_pre} pre-coverage windows)"
        print(msg)
        return

    # prioritize highest trigger count if too many windows
    filtered.sort(key=lambda x: (x[6], x[5]), reverse=True)
    if len(filtered) > max_windows:
        print(f"[news] capping windows for {log_path.stem}: {len(filtered)} -> {max_windows} (by triggers desc)")
        filtered = filtered[:max_windows]
    # regroup by day for batched fetch; score days by total triggers
    day_windows = {}
    for idx, w_start, w_end, start, end, trig_count, sev in filtered:
        day = w_start.date()
        day_windows.setdefault(day, []).append((idx, w_start, w_end, start, end, trig_count, sev))
    day_scores = {d: sum(t for *_, t, sev in lst) for d, lst in day_windows.items()}

    if not day_windows:
        msg = f"[news] no windows within GDELT coverage for {log_path.stem}"
        if skipped_pre:
            msg += f" (skipped {skipped_pre} pre-coverage windows)"
        print(msg)
        return

    days_sorted = sorted(day_windows.keys(), key=lambda d: day_scores[d], reverse=True)
    if len(days_sorted) > max_days:
        skipped = len(days_sorted) - max_days
        print(f"[news] too many days ({len(days_sorted)}) for {log_path.stem}; fetching first {max_days}, skipping {skipped}.")
        days_sorted = days_sorted[:max_days]

    summary_rows = []
    failures = 0
    for day in days_sorted:
        day_str = day.isoformat()
        day_start = pd.Timestamp(f"{day_str}T00:00:00Z")
        day_end = day_start + pd.Timedelta(days=1)
        if day_start >= now:
            print(f"[news] {log_path.stem} date={day_str} is in the future relative to now={now.date()}; skipping.")
            continue
        if day_end > now:
            print(f"[news] {log_path.stem} date={day_str} clamped end from {day_end.date()} to {now.date()} (no future fetch).")
            day_end = now
        try:
            events_day = gdelt_fetch(day_start, day_end, query or None, maxrows)
            fetch_note = f"events={len(events_day)}"
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 404:
                print(f"[news] {log_path.stem} date={day_str} returned 404; treating as no events.")
                events_day = pd.DataFrame()
                fetch_note = "events=0 (404)"
            else:
                print(f"[news] fetch failed for {day_str} ({log_path.stem}): {e}")
                failures += 1
                if failures >= max_failures:
                    print(f"[news] aborting further fetches for {log_path.stem} after {failures} failures.")
                    break
                events_day = pd.DataFrame()
                fetch_note = "events=0 (fetch failed)"
        except Exception as e:
            print(f"[news] fetch failed for {day_str} ({log_path.stem}): {e}")
            failures += 1
            if failures >= max_failures:
                print(f"[news] aborting further fetches for {log_path.stem} after {failures} failures.")
                break
            events_day = pd.DataFrame()
            fetch_note = "events=0 (fetch failed)"

        print(f"[news] {log_path.stem} date={day_str} {fetch_note}")
        for idx, w_start, w_end, raw_start, raw_end, trig_count, sev in day_windows[day]:
            events_win = events_day[(events_day["ts"] >= w_start) & (events_day["ts"] <= w_end)] if not events_day.empty else pd.DataFrame()
            fname = out_dir / f"{log_path.stem}_events_{idx:03d}_{w_start.strftime('%Y%m%dT%H%M%S')}_{w_end.strftime('%Y%m%dT%H%M%S')}.csv"
            events_win.to_csv(fname, index=False)
            top_codes = ""
            if "EventRootCode" in events_win.columns:
                vc = events_win["EventRootCode"].value_counts().head(3)
                top_codes = ";".join(f"{k}:{v}" for k, v in vc.items())
            # print a tiny preview of up to 2 events for context
            preview_rows = []
            preview_cols = [c for c in ["ts", "EventRootCode", "Goldstein", "AvgTone", "Actor1Name", "Actor2Name", "Actor1CountryCode", "Actor2CountryCode"] if c in events_win.columns]
            if not events_win.empty and preview_cols:
                for _, r in events_win.head(2).iterrows():
                    vals = [str(r.get(c, "")) for c in preview_cols]
                    preview_rows.append("|".join(vals))
            preview_str = "; ".join(preview_rows)
            print(
                f"[news] {log_path.stem} window {idx:03d} {w_start} → {w_end} | "
                f"triggers={trig_count} sev_sum_p_bad={sev:.3f} | "
                f"events={len(events_win)} | top_codes={top_codes} | file={fname.name}"
                + (f" | preview={preview_str}" if preview_str else "")
            )
            summary_rows.append(
                {
                    "market": log_path.stem,
                    "window_id": idx,
                    "start": w_start,
                    "end": w_end,
                    "date": day_str,
                    "triggers": trig_count,
                    "severity_sum_p_bad": sev,
                    "events": len(events_win),
                    "top_codes": top_codes,
                    "file": fname.name,
                }
            )

    summary_path = out_dir / f"{log_path.stem}_events_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[news] wrote summary to {summary_path}")


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
    ap.add_argument(
        "--emit-news-windows",
        action="store_true",
        help="After each market run, detect bad windows and fetch GDELT events (no API key).",
    )
    ap.add_argument("--p-bad-hi", type=float, default=0.7, help="p_bad threshold for news windows.")
    ap.add_argument("--news-max-gap-min", type=float, default=15.0, help="Gap to merge bad triggers into a window.")
    ap.add_argument("--news-pad-min", type=float, default=60.0, help="Pad around each window for news fetch.")
    ap.add_argument("--news-query", type=str, default="", help="Optional keyword filter for news fetch (GDELT query).")
    ap.add_argument("--news-maxrows", type=int, default=250, help="Max rows for news fetch (GDELT).")
    ap.add_argument("--news-max-days", type=int, default=5, help="Max distinct days to fetch per market to avoid spamming.")
    ap.add_argument("--news-max-failures", type=int, default=3, help="Abort news fetches after this many failures per market.")
    ap.add_argument("--news-max-windows", type=int, default=20, help="Max bad windows per market to fetch news for (highest trigger count first).")
    args = ap.parse_args()

    if args.markets:
        run_market_summaries(
            max_steps=args.max_steps,
            progress_every=args.market_progress_every,
            emit_news=args.emit_news_windows,
            news_kwargs={
                "p_bad_hi": args.p_bad_hi,
                "max_gap_min": args.news_max_gap_min,
                "pad_min": args.news_pad_min,
                "query": args.news_query,
                "maxrows": args.news_maxrows,
                "max_days": args.news_max_days,
                "max_failures": args.news_max_failures,
                "max_windows": args.news_max_windows,
            },
        )

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

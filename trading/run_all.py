"""
run_all.py
----------
One-shot runner that evaluates the trader across all cached markets and prints a
comparison table (per-market and overall).
"""

import argparse
import pathlib
import threading
import time

import numpy as np
import pandas as pd

try:
    from trading import run_trader  # provides the trading loop
    from trading import training_dashboard as dash  # for live plotting
except ModuleNotFoundError:
    import run_trader  # provides the trading loop
    import training_dashboard as dash  # for live plotting


def discover_markets():
    """
    Find all usable market CSVs under data/raw/stooq.
    Returns list of dicts: {name, path, price, volume}.
    """
    markets = []
    root = pathlib.Path("data/raw/stooq")
    if root.exists():
        for f in sorted(root.glob("*.csv")):
            try:
                price, volume, ts = run_trader.load_prices(f, return_time=True)
                if len(price) < 10:
                    continue
                markets.append({"name": f.stem, "path": f, "price": price, "volume": volume, "time": ts})
            except Exception:
                continue
    if not markets:
        # fallback synthetic market
        rng = np.random.default_rng(0)
        steps = rng.normal(loc=0.0, scale=0.01, size=1000)
        price = 100 + np.cumsum(steps)
        volume = np.ones_like(price) * 1e6
        markets.append({"name": "synthetic", "path": None, "price": price, "volume": volume, "time": None})
    return markets


def print_scoreboard(results):
    print("\n=== Per-market results ===")
    results = sorted(results, key=lambda r: r["pnl"], reverse=True)
    for r in results:
        hold_pct = f"{r['hold_pct']*100:.1f}%" if r.get("hold_pct") is not None else "n/a"
        print(
            f"{r['source']:<20} steps={r['steps']:>6} trades={r['trades']:>5} "
            f"pnl={r['pnl']:>10.4f} max_dd={r['max_drawdown']:>10.4f} hold={hold_pct}"
        )
    total_pnl = sum(r["pnl"] for r in results)
    winners = sum(1 for r in results if r["pnl"] > 0)
    print("\n=== Overall ===")
    print(f"markets={len(results)}, winners={winners}, losers={len(results)-winners}, total_pnl={total_pnl:.4f}")


def run_market_live(market, log_path, refresh, pr_path=None, sleep_s=0.01, max_steps=None):
    """
    Run a single market in a background thread and live-refresh the dashboard.
    """
    log_path = pathlib.Path(log_path)
    pr_curve = dash.load_log(pathlib.Path(pr_path)) if pr_path else None
    th = threading.Thread(
        target=run_trader.run_trading_loop,
        kwargs={
            "price": market["price"],
            "volume": market["volume"],
            "source": market["name"],
            "sleep_s": sleep_s,
            "max_steps": max_steps,
            "log_path": log_path,
        },
        daemon=True,
    )
    th.start()
    try:
        while th.is_alive():
            log = dash.load_log(log_path)
            if log is None:
                log = dash.synthetic_log()
            dash.draw(log, pr_curve)
            time.sleep(refresh)
        # final draw
        log = dash.load_log(log_path)
        if log is None:
            log = dash.synthetic_log()
        dash.draw(log, pr_curve)
        # brief pause so the final plot is visible between markets
        time.sleep(max(0.5, refresh))
    except KeyboardInterrupt:
        pass
    th.join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Show live dashboard while running each market.")
    ap.add_argument("--refresh", type=float, default=0.5, help="Dashboard refresh seconds when --live.")
    ap.add_argument("--sleep", type=float, default=0.01, help="Per-step sleep inside the trading loop when --live.")
    ap.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps for faster runs.")
    ap.add_argument(
        "--pr",
        type=str,
        default=None,
        help="Optional PR curve CSV (precision/recall sweep) to overlay when --live.",
    )
    args = ap.parse_args()

    markets = discover_markets()
    summaries = []
    for m in markets:
        name = m["name"]
        log_path = pathlib.Path(f"logs/trading_log_{name}.csv")
        print(f"\n=== Running market: {name} ===")
        if args.live:
            run_market_live(
                m,
                log_path=log_path,
                refresh=args.refresh,
                pr_path=args.pr,
                sleep_s=args.sleep,
                max_steps=args.max_steps,
            )
            # run_trading_loop already wrote summary to run_history; recompute summary from log
            log_df = dash.load_log(log_path)
            if log_df is not None and not log_df.empty:
                summary = {
                    "timestamp": pd.Timestamp.utcnow(),
                    "source": name,
                    "steps": int(log_df["t"].max()),
                    "trades": int((log_df["fill"] != 0).sum()) if "fill" in log_df else None,
                    "pnl": float(log_df["pnl"].iloc[-1]),
                    "hold_pct": float((log_df["action"] == 0).mean()) if "action" in log_df else None,
                    "max_drawdown": None,
                }
            else:
                summary, _ = run_trader.run_trading_loop(
                    price=m["price"],
                    volume=m["volume"],
                    source=name,
                    sleep_s=0.0,
                    max_steps=args.max_steps,
                    log_path=log_path,
                )
        else:
            summary, _ = run_trader.run_trading_loop(
                price=m["price"],
                volume=m["volume"],
                source=name,
                sleep_s=0.0,
                max_steps=args.max_steps,
                log_path=log_path,
            )
        summaries.append(summary)
    print_scoreboard(summaries)

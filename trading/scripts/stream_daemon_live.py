#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _list_binance_symbols(quote_asset: str, max_symbols: int | None) -> list[str]:
    """
    Return active spot symbols filtered by quote asset (e.g., USDT).
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    found = []
    for entry in payload.get("symbols", []):
        if entry.get("status") != "TRADING":
            continue
        if entry.get("quoteAsset") != quote_asset:
            continue
        if not entry.get("isSpotTradingAllowed", True):
            continue
        found.append(entry.get("symbol"))
        if max_symbols and len(found) >= max_symbols:
            break
    return [s for s in found if s]


def _wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for daemon at {host}:{port}")


def _fetch_symbols(all_symbols: bool, symbols: list[str], quote_asset: str, max_symbols: int | None) -> list[str]:
    if symbols:
        available = set(_list_binance_symbols(quote_asset, None))
        invalid = [s for s in symbols if s not in available]
        if invalid:
            raise SystemExit(
                f"Invalid symbols: {', '.join(invalid)}. "
                f"Use --list-symbols to inspect available {quote_asset} spot pairs."
            )
        return symbols[: max_symbols] if max_symbols else symbols
    if not all_symbols:
        return []
    return _list_binance_symbols(quote_asset, max_symbols)


def _stream_binance_trades(
    host: str,
    port: int,
    symbols: list[str],
    runtime: float,
    limit: int,
    poll_interval: float,
    log_requests: bool,
) -> int:
    base = "https://api.binance.com/api/v3/aggTrades"
    last_ids: dict[str, int] = {}
    next_allowed: dict[str, float] = {}
    end_time = time.time() + runtime
    sent = 0
    with socket.create_connection((host, port)) as sock:
        while time.time() < end_time:
            now = time.time()
            next_wake = None
            for symbol in symbols:
                ready_at = next_allowed.get(symbol, 0.0)
                if now < ready_at:
                    next_wake = ready_at if next_wake is None else min(next_wake, ready_at)
                    continue
                params = f"symbol={symbol}&limit={limit}"
                last_id = last_ids.get(symbol)
                if last_id is not None:
                    params += f"&fromId={last_id + 1}"
                url = f"{base}?{params}"
                started = time.monotonic()
                status = "200"
                try:
                    with urllib.request.urlopen(url, timeout=10) as resp:
                        status = str(getattr(resp, "status", "200"))
                        payload = json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    status = str(exc.code)
                    payload = []
                except urllib.error.URLError:
                    status = "ERR"
                    payload = []
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if log_requests:
                    print(
                        f"[binance] symbol={symbol} status={status} elapsed_ms={elapsed_ms:.1f} "
                        f"trades={len(payload)}",
                        file=sys.stderr,
                        flush=True,
                    )
                if not payload:
                    next_allowed[symbol] = time.time() + poll_interval
                    continue
                for trade in payload:
                    trade_id = trade.get("a")
                    ts_ms = trade.get("T")
                    price = trade.get("p")
                    qty = trade.get("q")
                    if trade_id is None or ts_ms is None or price is None or qty is None:
                        continue
                    current_id = last_ids.get(symbol)
                    last_ids[symbol] = trade_id if current_id is None else max(current_id, trade_id)
                    msg = {
                        "type": "trade",
                        "symbol": symbol,
                        "ts_ms": int(ts_ms),
                        "price": float(price),
                        "qty": float(qty),
                    }
                    sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
                    sent += 1
                next_allowed[symbol] = time.time() + poll_interval
                now = time.time()
            if next_wake is None:
                time.sleep(0.01)
            else:
                sleep_for = min(0.5, max(0.0, next_wake - time.time()))
                if sleep_for > 0:
                    time.sleep(sleep_for)
    return sent


def main() -> None:
    ap = argparse.ArgumentParser(description="Run stream daemon and feed live Binance trades.")
    ap.add_argument("--host", default="127.0.0.1", help="Daemon host to connect to")
    ap.add_argument("--port", type=int, default=9009, help="Daemon port to bind")
    ap.add_argument("--db", default="logs/research/market_live_binance.duckdb", help="DuckDB path")
    ap.add_argument("--symbol", default=None, help="Single Binance symbol to stream")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbol list to stream")
    ap.add_argument("--all-symbols", action="store_true", help="Stream all spot symbols")
    ap.add_argument("--quote-asset", default="USDT", help="Quote asset filter when streaming all")
    ap.add_argument("--max-symbols", type=int, default=None, help="Cap symbol count when streaming all")
    ap.add_argument("--runtime", type=float, default=20.0, help="Seconds to stream")
    ap.add_argument("--limit", type=int, default=1000, help="Binance aggTrades limit per request")
    ap.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between polls per symbol")
    ap.add_argument("--log-requests", action="store_true", help="Log request timing and HTTP status")
    ap.add_argument("--emit-decisions", action="store_true", help="Enable decision emission")
    ap.add_argument("--decision-sink", action="append", default=[], help="Decision sinks (repeatable)")
    ap.add_argument("--ohlc-sink", action="append", default=[], help="OHLC sinks (repeatable)")
    ap.add_argument("--metrics-host", default=None, help="Metrics bind host")
    ap.add_argument("--metrics-port", type=int, default=None, help="Metrics bind port")
    ap.add_argument("--list-symbols", action="store_true", help="List available Binance spot symbols and exit")
    ap.add_argument("--plot-follow", action="store_true", help="Live plot while streaming")
    ap.add_argument(
        "--plot-backend",
        choices=["mpl", "pyqtgraph"],
        default="mpl",
        help="Plot backend for decision stream",
    )
    ap.add_argument(
        "--plot-mode",
        choices=["decisions", "ohlcv", "dashboard"],
        default="decisions",
        help="Plot decisions (default), OHLCV bars, or dashboard",
    )
    ap.add_argument("--plot-ohlc-symbol", default=None, help="Symbol for OHLCV + PnL plots")
    ap.add_argument("--plot-webm", default=None, help="Write WebM timelapse to this path")
    ap.add_argument("--plot-out", default=None, help="Write a PNG snapshot (non-live)")
    ap.add_argument("--plot-window-seconds", type=float, default=600.0, help="Rolling window for live plot")
    ap.add_argument("--plot-interval-ms", type=int, default=1000, help="Live plot refresh interval")
    ap.add_argument("--plot-show-state", action="store_true", help="Overlay state markers")
    ap.add_argument("--plot-urgency-thickness", action="store_true", help="Scale markers by urgency")
    ap.add_argument("--plot-posture", action="store_true", help="Shade OBSERVE posture periods")
    ap.add_argument("--plot-fps", type=int, default=10, help="FPS for WebM output")
    ap.add_argument("--plot-frame-step", type=int, default=5, help="Frame step for WebM output")
    ap.add_argument("--plot-fee-rate", type=float, default=0.0005, help="Fee rate for PnL plot")
    ap.add_argument("--plot-slippage", type=float, default=0.0003, help="Slippage rate for PnL plot")
    ap.add_argument("--plot-min-trade", type=float, default=0.02, help="Minimum exposure change to trade")
    ap.add_argument("--decision-fee-rate", type=float, default=0.0005, help="Fee rate for cost gate")
    ap.add_argument("--decision-slippage", type=float, default=0.0003, help="Slippage rate for cost gate")
    ap.add_argument("--decision-edge-bps", type=float, default=0.0, help="Edge proxy in bps (0 disables)")
    ap.add_argument("--decision-cost-budget", type=float, default=0.0, help="Per-window cost budget (0 disables)")
    ap.add_argument(
        "--decision-cost-window",
        type=float,
        default=60.0,
        help="Cost budget rolling window in seconds",
    )
    args = ap.parse_args()

    if args.list_symbols:
        available = _list_binance_symbols(args.quote_asset, args.max_symbols)
        for sym in available:
            print(sym)
        return
    if args.plot_backend == "pyqtgraph" and (args.plot_webm or args.plot_out):
        raise SystemExit("PyQtGraph backend does not support --plot-webm or --plot-out.")
    if args.plot_mode == "ohlcv" and (args.plot_webm or args.plot_out):
        raise SystemExit("OHLCV plot does not support --plot-webm or --plot-out.")
    if args.plot_mode == "ohlcv" and args.plot_backend != "pyqtgraph":
        raise SystemExit("OHLCV plot requires --plot-backend pyqtgraph.")
    if args.plot_mode == "dashboard" and not args.emit_decisions:
        raise SystemExit("Dashboard plot requires --emit-decisions.")
    if args.plot_mode == "dashboard" and args.plot_backend != "pyqtgraph":
        raise SystemExit("Dashboard plot requires --plot-backend pyqtgraph.")

    cmd = [
        sys.executable,
        "scripts/stream_daemon.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--db",
        args.db,
    ]
    if args.emit_decisions:
        cmd.append("--emit-decisions")
        cmd.extend(
            [
                "--decision-fee-rate",
                str(args.decision_fee_rate),
                "--decision-slippage",
                str(args.decision_slippage),
                "--decision-edge-bps",
                str(args.decision_edge_bps),
                "--decision-cost-budget",
                str(args.decision_cost_budget),
                "--decision-cost-window",
                str(args.decision_cost_window),
            ]
        )
    decision_sinks = list(args.decision_sink)
    ohlc_sinks = list(args.ohlc_sink)
    plot_path = None
    if args.plot_mode in {"decisions", "dashboard"} and (args.plot_follow or args.plot_webm or args.plot_out):
        for sink in decision_sinks:
            if sink.startswith("file:"):
                plot_path = sink[len("file:") :]
                break
        if plot_path is None:
            plot_path = "logs/decisions.ndjson"
            decision_sinks.append(f"file:{plot_path}")
    ohlc_path = None
    if args.plot_mode in {"ohlcv", "dashboard"} and args.plot_follow:
        for sink in ohlc_sinks:
            if sink.startswith("file:"):
                ohlc_path = sink[len("file:") :]
                break
        if ohlc_path is None:
            ohlc_path = "logs/ohlc_1s.ndjson"
            ohlc_sinks.append(f"file:{ohlc_path}")
    for sink in decision_sinks:
        cmd.extend(["--decision-sink", sink])
    for sink in ohlc_sinks:
        cmd.extend(["--ohlc-sink", sink])
    if args.metrics_host and args.metrics_port:
        cmd.extend(["--metrics-host", args.metrics_host, "--metrics-port", str(args.metrics_port)])

    daemon = subprocess.Popen(cmd)
    plotter = None

    def _shutdown(_sig=None, _frame=None):
        if daemon.poll() is None:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
        if plotter and plotter.poll() is None:
            plotter.terminate()
            try:
                plotter.wait(timeout=5)
            except subprocess.TimeoutExpired:
                plotter.kill()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        _wait_for_port(args.host, args.port, timeout=5.0)

        symbols = []
        if args.symbol:
            symbols.append(args.symbol)
        if args.symbols:
            symbols.extend([s.strip() for s in args.symbols.split(",") if s.strip()])

        symbols = _fetch_symbols(args.all_symbols, symbols, args.quote_asset, args.max_symbols)
        if not symbols:
            raise SystemExit("No symbols selected; use --symbol, --symbols, or --all-symbols.")

        if args.plot_follow or args.plot_webm or args.plot_out:
            if args.plot_mode == "ohlcv":
                plot_cmd = [
                    sys.executable,
                    "scripts/plot_stream_ohlcv_pg.py",
                    "--ndjson",
                    ohlc_path or "",
                    "--symbols",
                    ",".join(symbols),
                    "--interval-ms",
                    str(args.plot_interval_ms),
                    "--window-seconds",
                    str(args.plot_window_seconds),
                ]
                plotter = subprocess.Popen(plot_cmd)
            elif args.plot_mode == "dashboard":
                ohlc_symbol = args.plot_ohlc_symbol or symbols[0]
                plot_cmd = [
                    sys.executable,
                    "scripts/plot_stream_dashboard_pg.py",
                    "--ndjson",
                    plot_path,
                    "--ohlc-ndjson",
                    ohlc_path or "",
                    "--symbols",
                    ",".join(symbols),
                    "--ohlc-symbol",
                    ohlc_symbol,
                    "--interval-ms",
                    str(args.plot_interval_ms),
                    "--window-seconds",
                    str(args.plot_window_seconds),
                    "--fee-rate",
                    str(args.plot_fee_rate),
                    "--slippage",
                    str(args.plot_slippage),
                    "--min-trade",
                    str(args.plot_min_trade),
                ]
                plotter = subprocess.Popen(plot_cmd)
            elif plot_path:
                plot_script = (
                    "scripts/plot_stream_decisions_pg.py"
                    if args.plot_backend == "pyqtgraph"
                    else "scripts/plot_stream_decisions.py"
                )
                plot_cmd = [sys.executable, plot_script, "--ndjson", plot_path]
                if args.plot_follow:
                    plot_cmd.extend(
                        [
                            "--follow",
                            "--interval-ms",
                            str(args.plot_interval_ms),
                            "--window-seconds",
                            str(args.plot_window_seconds),
                        ]
                    )
                if args.plot_show_state:
                    plot_cmd.append("--show-state")
                if args.plot_urgency_thickness:
                    plot_cmd.append("--urgency-thickness")
                if args.plot_posture:
                    plot_cmd.append("--show-posture")
                if args.plot_webm:
                    plot_cmd.extend(
                        [
                            "--webm",
                            args.plot_webm,
                            "--fps",
                            str(args.plot_fps),
                            "--frame-step",
                            str(max(1, args.plot_frame_step)),
                        ]
                    )
                if args.plot_out:
                    plot_cmd.extend(["--out", args.plot_out])
                plotter = subprocess.Popen(plot_cmd)
        sent = _stream_binance_trades(
            args.host,
            args.port,
            symbols,
            args.runtime,
            args.limit,
            args.poll_interval,
            args.log_requests,
        )
        print(f"sent={sent}")
    finally:
        _shutdown()


if __name__ == "__main__":
    main()

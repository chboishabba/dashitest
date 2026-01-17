#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import socket
import time

import duckdb


def _send_messages(host: str, port: int, symbols: list[str], seconds: float, rate: float, mode: str) -> int:
    total = 0
    interval = 1.0 / max(rate, 1.0)
    end_time = time.time() + seconds
    with socket.create_connection((host, port)) as sock:
        while time.time() < end_time:
            symbol = random.choice(symbols)
            ts_ms = int(time.time() * 1000)
            price = 100.0 + random.random()
            qty = random.random() * 2.0
            if mode == "ohlc1s":
                payload = {
                    "type": "ohlc1s",
                    "symbol": symbol,
                    "ts_ms": ts_ms - (ts_ms % 1000),
                    "open": price,
                    "high": price + 0.1,
                    "low": price - 0.1,
                    "close": price,
                    "volume": qty,
                    "trades": 1,
                }
            else:
                payload = {
                    "type": "trade",
                    "symbol": symbol,
                    "ts_ms": ts_ms,
                    "price": price,
                    "qty": qty,
                }
            line = json.dumps(payload) + "\n"
            sock.sendall(line.encode("utf-8"))
            total += 1
            time.sleep(interval)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream daemon load test")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--rate", type=float, default=200.0, help="Messages per second")
    ap.add_argument("--symbols", default="BTCUSDT,ES,NQ")
    ap.add_argument("--mode", choices=["trade", "ohlc1s"], default="trade")
    ap.add_argument("--db", default="logs/research/market.duckdb")
    ap.add_argument(
        "--read-only",
        action="store_true",
        help="Open DuckDB in read-only mode to avoid writer locks",
    )
    ap.add_argument(
        "--check-decisions",
        action="store_true",
        help="Probe decision tables when decision mode is enabled",
    )
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    sent = _send_messages(args.host, args.port, symbols, args.seconds, args.rate, args.mode)

    con = duckdb.connect(args.db, read_only=args.read_only)
    rows = con.execute("SELECT COUNT(*) FROM ohlc_1s").fetchone()[0]
    summary = [f"sent={sent}", f"mode={args.mode}", f"ohlc_rows={rows}"]
    if args.check_decisions:
        tables = {
            "stream_state": "state_rows",
            "stream_actions": "action_rows",
        }
        for table, label in tables.items():
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table],
            ).fetchone()[0]
            if exists:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                summary.append(f"{label}={count}")
            else:
                summary.append(f"{label}=missing")
    print(" ".join(summary))


if __name__ == "__main__":
    main()

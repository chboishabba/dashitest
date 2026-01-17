#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import selectors
import signal
import socket
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import duckdb
import pandas as pd

try:
    from tools.ingest_archives_to_duckdb import ingest_dataframe
    from trading.summarisation.summariser import Summariser, SummarySpec
    from phase6.gate import Phase6ExposureGate
    from strategy.triadic_strategy import TriadicStrategy
    from signals.triadic import compute_triadic_state
    from posture import Posture
    from signals.asymmetry_sensor import InfluenceTensorMonitor
except ModuleNotFoundError:  # pragma: no cover - support running from scripts/
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from tools.ingest_archives_to_duckdb import ingest_dataframe
    from trading.summarisation.summariser import Summariser, SummarySpec
    from phase6.gate import Phase6ExposureGate
    from strategy.triadic_strategy import TriadicStrategy
    from signals.triadic import compute_triadic_state
    from posture import Posture
    from signals.asymmetry_sensor import InfluenceTensorMonitor


@dataclass
class OhlcBucket:
    ts_sec: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int


class TradeAggregator:
    def __init__(self) -> None:
        self._buckets: dict[str, OhlcBucket] = {}

    def add_trade(self, symbol: str, ts_ms: int, price: float, qty: float) -> dict[str, Any] | None:
        ts_sec = int(ts_ms // 1000)
        bucket = self._buckets.get(symbol)
        if bucket is None:
            self._buckets[symbol] = OhlcBucket(ts_sec, price, price, price, price, qty, 1)
            return None
        if ts_sec == bucket.ts_sec:
            bucket.high = max(bucket.high, price)
            bucket.low = min(bucket.low, price)
            bucket.close = price
            bucket.volume += qty
            bucket.trades += 1
            return None
        if ts_sec < bucket.ts_sec:
            return None
        row = self._bucket_to_row(symbol, bucket)
        self._buckets[symbol] = OhlcBucket(ts_sec, price, price, price, price, qty, 1)
        return row

    def flush_all(self) -> list[dict[str, Any]]:
        rows = []
        for symbol, bucket in list(self._buckets.items()):
            rows.append(self._bucket_to_row(symbol, bucket))
            del self._buckets[symbol]
        return rows

    @staticmethod
    def _bucket_to_row(symbol: str, bucket: OhlcBucket) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "timestamp": bucket.ts_sec * 1000,
            "open": bucket.open,
            "high": bucket.high,
            "low": bucket.low,
            "close": bucket.close,
            "volume": bucket.volume,
            "trades": bucket.trades,
        }


class DecisionEngine:
    def __init__(
        self,
        *,
        state_window: int,
        tau_on: float,
        tau_off: float,
        base_size: float,
        phase6_log_dir: str | None,
        influence_log_dir: str | None,
    ) -> None:
        self.state_window = max(2, state_window)
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.base_size = base_size
        self.phase6_gate = Phase6ExposureGate(phase6_log_dir) if phase6_log_dir else None
        self.influence_monitor = (
            InfluenceTensorMonitor(influence_log_dir) if influence_log_dir else None
        )
        self.state_buffers: dict[str, deque[float]] = {}
        self.strategies: dict[str, TriadicStrategy] = {}

    def _strategy_for(self, symbol: str) -> TriadicStrategy:
        strategy = self.strategies.get(symbol)
        if strategy is None:
            strategy = TriadicStrategy(
                symbol=symbol,
                base_size=self.base_size,
                tau_on=self.tau_on,
                tau_off=self.tau_off,
                influence_monitor=self.influence_monitor,
            )
            self.strategies[symbol] = strategy
        return strategy

    def update(self, row: dict[str, Any]) -> tuple[int, Any, bool, Posture] | None:
        try:
            symbol = str(row["symbol"])
            ts_ms = int(row["timestamp"])
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            return None
        closes = self.state_buffers.setdefault(symbol, deque(maxlen=self.state_window))
        closes.append(close)
        state = int(compute_triadic_state(list(closes), window=self.state_window)[-1])
        gate_open = self.phase6_gate.is_allowed() if self.phase6_gate else True
        posture = Posture.TRADE_NORMAL if gate_open else Posture.OBSERVE
        intent = self._strategy_for(symbol).step(ts=ts_ms, state=state, posture=posture)
        return state, intent, gate_open, posture


class DecisionStorage:
    def __init__(self, db_path: str, source_label: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.source_label = source_label
        self.con = duckdb.connect(str(self.db_path))
        self.state_columns = [
            "timestamp",
            "symbol",
            "close",
            "state",
            "gate_open",
            "posture",
            "source_file",
        ]
        self.action_columns = [
            "timestamp",
            "symbol",
            "state",
            "direction",
            "target_exposure",
            "urgency",
            "hold",
            "actionability",
            "reason",
            "gate_open",
            "posture",
            "source_file",
        ]
        self.pending_states: list[dict[str, Any]] = []
        self.pending_actions: list[dict[str, Any]] = []
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS stream_state (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                close DOUBLE,
                state INTEGER,
                gate_open BOOLEAN,
                posture INTEGER,
                source_file VARCHAR
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS stream_actions (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                state INTEGER,
                direction INTEGER,
                target_exposure DOUBLE,
                urgency DOUBLE,
                hold BOOLEAN,
                actionability DOUBLE,
                reason VARCHAR,
                gate_open BOOLEAN,
                posture INTEGER,
                source_file VARCHAR
            );
            """
        )
        action_columns = ", ".join(self.action_columns)
        self.con.execute(
            f"""
            CREATE VIEW IF NOT EXISTS stream_actions_latest AS
            SELECT {action_columns}
            FROM (
                SELECT {action_columns},
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM stream_actions
            )
            WHERE rn = 1
            """
        )

    def append_state(self, row: dict[str, Any]) -> None:
        row["source_file"] = self.source_label
        self.pending_states.append(row)

    def append_action(self, row: dict[str, Any]) -> None:
        row["source_file"] = self.source_label
        self.pending_actions.append(row)

    def flush(self) -> None:
        self._flush_table("stream_state", self.state_columns, self.pending_states)
        self._flush_table("stream_actions", self.action_columns, self.pending_actions)

    def _flush_table(self, table: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "symbol"])
        if df.empty:
            rows.clear()
            return
        df = df[columns]
        self.con.register("df_live_decisions", df)
        column_list = ", ".join(columns)
        self.con.execute(
            f"INSERT INTO {table} ({column_list}) SELECT {column_list} FROM df_live_decisions"
        )
        rows.clear()


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - external interface
        if self.path not in ("/", "/metrics"):
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(self.server.daemon_ref.metrics_snapshot()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class NdjsonSink:
    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.kind = None
        self.path = None
        self.host = None
        self.port = None
        self.socket = None
        self.handle = None
        self._parse()

    def _parse(self) -> None:
        if self.spec.startswith("file:"):
            self.kind = "file"
            self.path = self.spec[len("file:") :]
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a", encoding="utf-8")
            return
        if self.spec.startswith("tcp://"):
            self.kind = "tcp"
            parsed = urlparse(self.spec)
            self.host = parsed.hostname
            self.port = parsed.port
            return
        raise ValueError(f"Unknown NDJSON sink: {self.spec}")

    def send(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload) + "\n"
        if self.kind == "file":
            if self.handle:
                self.handle.write(line)
                self.handle.flush()
            return
        if self.kind == "tcp":
            if not self.host or not self.port:
                return
            if self.socket is None:
                try:
                    self.socket = socket.create_connection((self.host, self.port), timeout=2)
                except OSError:
                    self.socket = None
                    return
            try:
                self.socket.sendall(line.encode("utf-8"))
            except OSError:
                try:
                    self.socket.close()
                except OSError:
                    pass
                self.socket = None

    def close(self) -> None:
        if self.handle:
            try:
                self.handle.close()
            except OSError:
                pass
            self.handle = None
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None


class DecisionSink(NdjsonSink):
    pass


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


class DecisionCostGate:
    def __init__(
        self,
        *,
        fee_rate: float,
        slippage: float,
        edge_bps: float,
        cost_budget: float,
        cost_window_s: float,
    ) -> None:
        self.fee_rate = max(0.0, fee_rate)
        self.slippage = max(0.0, slippage)
        self.edge_bps = max(0.0, edge_bps)
        self.cost_budget = max(0.0, cost_budget)
        self.cost_window_ms = int(max(1.0, cost_window_s) * 1000.0)
        self.last_targets: dict[str, float] = {}
        self.cost_history: dict[str, deque[tuple[int, float]]] = {}

    def _prune_costs(self, symbol: str, ts_ms: int) -> None:
        history = self.cost_history.get(symbol)
        if not history:
            return
        cutoff = ts_ms - self.cost_window_ms
        while history and history[0][0] < cutoff:
            history.popleft()

    def _window_cost(self, symbol: str, ts_ms: int) -> float:
        self._prune_costs(symbol, ts_ms)
        history = self.cost_history.get(symbol)
        if not history:
            return 0.0
        return sum(cost for _ts, cost in history)

    def _record_cost(self, symbol: str, ts_ms: int, cost: float) -> None:
        if self.cost_budget <= 0.0:
            return
        history = self.cost_history.setdefault(symbol, deque())
        history.append((ts_ms, cost))

    def apply(self, ts_ms: int, symbol: str, price: float, intent: Any) -> dict[str, Any]:
        last_target = self.last_targets.get(symbol, 0.0)
        proposed_target = float(intent.target_exposure) * float(intent.direction)
        if getattr(intent, "hold", False):
            proposed_target = last_target

        delta = proposed_target - last_target
        if price <= 0.0:
            delta = 0.0
        cost = abs(delta) * price * (self.fee_rate + self.slippage)

        allow_reduction = abs(proposed_target) <= abs(last_target)
        blocked = False
        reasons: list[str] = []

        if not allow_reduction and self.cost_budget > 0.0:
            budget_remaining = self.cost_budget - self._window_cost(symbol, ts_ms)
            if cost > max(0.0, budget_remaining):
                blocked = True
                reasons.append("budget")

        if not allow_reduction and self.edge_bps > 0.0:
            edge_rate = self.edge_bps * 1e-4
            actionability = max(0.0, min(1.0, float(getattr(intent, "actionability", 0.0))))
            urgency = max(0.0, min(1.0, float(getattr(intent, "urgency", 0.0))))
            expected_benefit = abs(delta) * price * edge_rate * actionability * urgency
            if expected_benefit < cost:
                blocked = True
                reasons.append("edge")

        if blocked:
            direction = _sign(last_target)
            target_exposure = abs(last_target)
            reason = str(getattr(intent, "reason", ""))
            suffix = "+".join(reasons)
            if reason:
                reason = f"{reason} + cost_gate_{suffix}"
            else:
                reason = f"cost_gate_{suffix}"
            return {
                "direction": int(direction),
                "target_exposure": float(target_exposure),
                "urgency": 0.0,
                "hold": True,
                "actionability": 0.0,
                "reason": reason,
                "blocked": True,
                "cost": 0.0,
            }

        if abs(delta) > 0.0:
            self._record_cost(symbol, ts_ms, cost)
        self.last_targets[symbol] = proposed_target
        return {
            "direction": int(intent.direction),
            "target_exposure": float(intent.target_exposure),
            "urgency": float(intent.urgency),
            "hold": bool(intent.hold),
            "actionability": float(intent.actionability),
            "reason": str(intent.reason),
            "blocked": False,
            "cost": cost,
        }


class StreamDaemon:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        db_path: str,
        batch_size: int,
        flush_interval: float,
        summarise_window: int | None,
        emit_decisions: bool,
        decision_stdout: bool,
        state_window: int,
        tau_on: float,
        tau_off: float,
        decision_base_size: float,
        decision_fee_rate: float,
        decision_slippage: float,
        decision_edge_bps: float,
        decision_cost_budget: float,
        decision_cost_window: float,
        phase6_log_dir: str | None,
        influence_log_dir: str | None,
        decision_sinks: list[str] | None,
        ohlc_sinks: list[str] | None,
        tail_path: str | None,
        tail_follow: bool,
        metrics_host: str | None,
        metrics_port: int | None,
    ) -> None:
        self.host = host
        self.port = port
        self.db_path = db_path
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.1, flush_interval)
        self.selector = selectors.DefaultSelector()
        self.server = None
        self.buffers: dict[socket.socket, bytearray] = {}
        self.running = True
        self.aggregator = TradeAggregator()
        self.pending_rows: list[dict[str, Any]] = []
        self.last_flush = time.monotonic()
        self.summarise_window = summarise_window
        self.summariser = None
        self.close_buffers: dict[str, deque[float]] = {}
        self.time_buffers: dict[str, deque[int]] = {}
        self.last_summary_sec: dict[str, int] = {}
        self.emit_decisions = emit_decisions
        self.decision_stdout = decision_stdout
        self.decision_engine = None
        self.decision_storage = None
        self.decision_cost_gate = None
        self.run_id = uuid.uuid4().hex
        self.decision_sinks = [DecisionSink(spec) for spec in decision_sinks or []]
        self.ohlc_sinks = [NdjsonSink(spec) for spec in ohlc_sinks or []]
        self.last_bar_ts_ms: int | None = None
        self.last_action_ts_ms: int | None = None
        self.last_state_ts_ms: int | None = None
        self.metrics_host = metrics_host
        self.metrics_port = metrics_port
        self.metrics_server = None
        self.metrics_thread = None
        self.tail_path = Path(tail_path) if tail_path else None
        self.tail_follow = tail_follow
        self.tail_queue: deque[bytes] = deque()
        self.tail_thread = None
        self.stats = {
            "connections": 0,
            "lines": 0,
            "invalid_lines": 0,
            "trades": 0,
            "ohlc": 0,
            "decisions": 0,
            "decisions_blocked": 0,
            "decision_cost_estimate": 0.0,
        }
        if summarise_window:
            spec = SummarySpec(window_seconds=summarise_window)
            self.summariser = Summariser(spec=spec)
        if emit_decisions:
            self.decision_engine = DecisionEngine(
                state_window=state_window,
                tau_on=tau_on,
                tau_off=tau_off,
                base_size=decision_base_size,
                phase6_log_dir=phase6_log_dir,
                influence_log_dir=influence_log_dir,
            )
            self.decision_storage = DecisionStorage(db_path, source_label="live:decision")
            if decision_cost_budget > 0.0 or decision_edge_bps > 0.0:
                self.decision_cost_gate = DecisionCostGate(
                    fee_rate=decision_fee_rate,
                    slippage=decision_slippage,
                    edge_bps=decision_edge_bps,
                    cost_budget=decision_cost_budget,
                    cost_window_s=decision_cost_window,
                )

    def start(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen()
        self.server.setblocking(False)
        self.selector.register(self.server, selectors.EVENT_READ, self._accept)
        self._start_tail()
        self._start_metrics()

        while self.running:
            events = self.selector.select(timeout=0.5)
            for key, _mask in events:
                callback = key.data
                callback(key.fileobj)
            self._drain_tail()
            self._periodic_flush()

        self._shutdown()

    def _accept(self, sock: socket.socket) -> None:
        conn, _addr = sock.accept()
        conn.setblocking(False)
        self.buffers[conn] = bytearray()
        self.selector.register(conn, selectors.EVENT_READ, self._read)
        self.stats["connections"] += 1

    def _read(self, conn: socket.socket) -> None:
        try:
            data = conn.recv(8192)
        except ConnectionResetError:
            data = b""
        if not data:
            self._close_conn(conn)
            return
        buffer = self.buffers[conn]
        buffer.extend(data)
        while True:
            newline = buffer.find(b"\n")
            if newline == -1:
                break
            line = buffer[:newline]
            del buffer[: newline + 1]
            if not line:
                continue
            self._handle_line(line)

    def _handle_line(self, line: bytes) -> None:
        try:
            payload = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            self.stats["invalid_lines"] += 1
            return
        msg_type = payload.get("type")
        self.stats["lines"] += 1
        if msg_type == "trade":
            self._handle_trade(payload)
        elif msg_type == "ohlc1s":
            self._handle_ohlc(payload)
        else:
            self.stats["invalid_lines"] += 1

    def _handle_trade(self, payload: dict[str, Any]) -> None:
        try:
            symbol = str(payload["symbol"])
            ts_ms = int(payload["ts_ms"])
            price = float(payload["price"])
            qty = float(payload["qty"])
        except (KeyError, TypeError, ValueError):
            return
        self.stats["trades"] += 1
        row = self.aggregator.add_trade(symbol, ts_ms, price, qty)
        if row:
            self._enqueue_row(row)

    def _handle_ohlc(self, payload: dict[str, Any]) -> None:
        try:
            row = {
                "symbol": str(payload["symbol"]),
                "timestamp": int(payload["ts_ms"]),
                "open": float(payload["open"]),
                "high": float(payload["high"]),
                "low": float(payload["low"]),
                "close": float(payload["close"]),
                "volume": float(payload["volume"]),
                "trades": int(payload.get("trades", 0)),
            }
        except (KeyError, TypeError, ValueError):
            return
        self.stats["ohlc"] += 1
        self._enqueue_row(row)

    def _enqueue_row(self, row: dict[str, Any]) -> None:
        self.pending_rows.append(row)
        self.last_bar_ts_ms = int(row.get("timestamp", 0))
        self._update_summary_buffers(row)
        self._emit_decision(row)
        self._emit_ohlc(row)
        if len(self.pending_rows) >= self.batch_size:
            self._flush_rows()

    def _update_summary_buffers(self, row: dict[str, Any]) -> None:
        if not self.summariser or self.summarise_window is None:
            return
        symbol = row["symbol"]
        ts_ms = int(row["timestamp"])
        close = float(row["close"])
        closes = self.close_buffers.setdefault(symbol, deque(maxlen=self.summarise_window))
        times = self.time_buffers.setdefault(symbol, deque(maxlen=self.summarise_window))
        closes.append(close)
        times.append(ts_ms)
        if len(closes) < self.summarise_window:
            return
        last_sec = ts_ms // 1000
        last_summary = self.last_summary_sec.get(symbol)
        if last_summary is None or last_sec - last_summary >= self.summarise_window:
            self.summariser.summarize(
                list(closes),
                symbol=symbol,
                timestamps=list(times),
            )
            self.last_summary_sec[symbol] = last_sec

    def _flush_rows(self) -> None:
        if not self.pending_rows:
            return
        df = pd.DataFrame(self.pending_rows)
        ingest_dataframe(frame=df, db=self.db_path, source_file="live")
        self.pending_rows.clear()
        self.last_flush = time.monotonic()
        if self.decision_storage:
            self.decision_storage.flush()

    def _periodic_flush(self) -> None:
        if time.monotonic() - self.last_flush < self.flush_interval:
            return
        for row in self.aggregator.flush_all():
            self.pending_rows.append(row)
        self._flush_rows()

    def _emit_decision(self, row: dict[str, Any]) -> None:
        if not self.decision_engine or not self.decision_storage:
            return
        decision = self.decision_engine.update(row)
        if decision is None:
            return
        state, intent, gate_open, posture = decision
        try:
            ts_ms = int(row["timestamp"])
            symbol = str(row["symbol"])
            close = float(row["close"])
        except (KeyError, TypeError, ValueError):
            return
        state_row = {
            "timestamp": ts_ms,
            "symbol": symbol,
            "close": close,
            "state": state,
            "gate_open": gate_open,
            "posture": int(posture),
        }
        gate_payload = None
        if self.decision_cost_gate:
            gate_payload = self.decision_cost_gate.apply(ts_ms, symbol, close, intent)
            if gate_payload.get("blocked"):
                self.stats["decisions_blocked"] += 1
            self.stats["decision_cost_estimate"] += float(gate_payload.get("cost", 0.0))
        action_row = {
            "timestamp": ts_ms,
            "symbol": symbol,
            "state": state,
            "direction": int(gate_payload["direction"]) if gate_payload else int(intent.direction),
            "target_exposure": float(gate_payload["target_exposure"])
            if gate_payload
            else float(intent.target_exposure),
            "urgency": float(gate_payload["urgency"]) if gate_payload else float(intent.urgency),
            "hold": bool(gate_payload["hold"]) if gate_payload else bool(intent.hold),
            "actionability": float(gate_payload["actionability"])
            if gate_payload
            else float(intent.actionability),
            "reason": str(gate_payload["reason"]) if gate_payload else str(intent.reason),
            "gate_open": gate_open,
            "posture": int(posture),
        }
        self.decision_storage.append_state(state_row)
        self.decision_storage.append_action(action_row)
        self.stats["decisions"] += 1
        self.last_action_ts_ms = ts_ms
        self.last_state_ts_ms = ts_ms
        payload = {
            **action_row,
            "run_id": self.run_id,
            "source": "stream_daemon",
        }
        if self.decision_stdout:
            print(json.dumps(payload), flush=True)
        for sink in self.decision_sinks:
            sink.send(payload)

    def _emit_ohlc(self, row: dict[str, Any]) -> None:
        if not self.ohlc_sinks:
            return
        payload = dict(row)
        payload["type"] = "ohlc1s"
        payload["ts_ms"] = payload.get("timestamp")
        for sink in self.ohlc_sinks:
            sink.send(payload)

    def _start_tail(self) -> None:
        if not self.tail_path:
            return
        self.tail_thread = threading.Thread(target=self._tail_worker, daemon=True)
        self.tail_thread.start()

    def _tail_worker(self) -> None:
        if not self.tail_path:
            return
        if not self.tail_path.exists():
            return
        with self.tail_path.open("rb") as fh:
            while self.running:
                line = fh.readline()
                if not line:
                    if not self.tail_follow:
                        break
                    time.sleep(0.1)
                    continue
                stripped = line.strip()
                if stripped:
                    self.tail_queue.append(stripped)

    def _drain_tail(self) -> None:
        while self.tail_queue:
            line = self.tail_queue.popleft()
            if line:
                self._handle_line(line)

    def _start_metrics(self) -> None:
        if not self.metrics_host or not self.metrics_port:
            return
        self.metrics_server = ThreadingHTTPServer(
            (self.metrics_host, self.metrics_port), _MetricsHandler
        )
        self.metrics_server.daemon_ref = self
        self.metrics_thread = threading.Thread(
            target=self.metrics_server.serve_forever, daemon=True
        )
        self.metrics_thread.start()

    def metrics_snapshot(self) -> dict[str, Any]:
        now = time.time()
        flush_age = max(0.0, time.monotonic() - self.last_flush)
        return {
            "ts": now,
            "host": self.host,
            "port": self.port,
            "connections": self.stats["connections"],
            "lines": self.stats["lines"],
            "invalid_lines": self.stats["invalid_lines"],
            "trades": self.stats["trades"],
            "ohlc": self.stats["ohlc"],
            "decisions": self.stats["decisions"],
            "decisions_blocked": self.stats["decisions_blocked"],
            "decision_cost_estimate": self.stats["decision_cost_estimate"],
            "pending_rows": len(self.pending_rows),
            "last_flush_age_s": flush_age,
            "last_bar_ts_ms": self.last_bar_ts_ms,
            "last_action_ts_ms": self.last_action_ts_ms,
            "last_state_ts_ms": self.last_state_ts_ms,
        }

    def _close_conn(self, conn: socket.socket) -> None:
        try:
            self.selector.unregister(conn)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        self.buffers.pop(conn, None)

    def _shutdown(self) -> None:
        for row in self.aggregator.flush_all():
            self.pending_rows.append(row)
        self._flush_rows()
        if self.metrics_server:
            self.metrics_server.shutdown()
            self.metrics_server.server_close()
        for sink in self.decision_sinks:
            sink.close()
        for sink in self.ohlc_sinks:
            sink.close()
        for conn in list(self.buffers.keys()):
            self._close_conn(conn)
        if self.server:
            try:
                self.selector.unregister(self.server)
            except Exception:
                pass
            self.server.close()

    def stop(self) -> None:
        self.running = False


def main() -> None:
    ap = argparse.ArgumentParser(description="TCP stream daemon (NDJSON)")
    ap.add_argument("--host", default="0.0.0.0", help="Bind host")
    ap.add_argument("--port", type=int, default=9009, help="Bind port")
    ap.add_argument("--db", default="logs/research/market.duckdb", help="DuckDB path")
    ap.add_argument("--batch-size", type=int, default=200, help="Rows per ingest batch")
    ap.add_argument("--flush-interval", type=float, default=1.0, help="Seconds between forced flushes")
    ap.add_argument("--summarise-window", type=int, default=None, help="Window size in seconds for summariser")
    ap.add_argument("--emit-decisions", action="store_true", help="Emit decision actions/state per bar")
    ap.add_argument(
        "--no-decision-stdout",
        dest="decision_stdout",
        action="store_false",
        help="Disable decision NDJSON stdout output",
    )
    ap.set_defaults(decision_stdout=True)
    ap.add_argument("--state-window", type=int, default=200, help="Bar window for triadic state")
    ap.add_argument("--tau-on", type=float, default=0.0, help="Triadic strategy tau_on threshold")
    ap.add_argument("--tau-off", type=float, default=0.0, help="Triadic strategy tau_off threshold")
    ap.add_argument("--decision-base-size", type=float, default=0.05, help="Triadic base size")
    ap.add_argument("--decision-fee-rate", type=float, default=0.0005, help="Fee rate for cost gate")
    ap.add_argument("--decision-slippage", type=float, default=0.0003, help="Slippage rate for cost gate")
    ap.add_argument(
        "--decision-edge-bps",
        type=float,
        default=0.0,
        help="Edge proxy in bps for cost gate (0 disables)",
    )
    ap.add_argument(
        "--decision-cost-budget",
        type=float,
        default=0.0,
        help="Per-window cost budget for cost gate (0 disables)",
    )
    ap.add_argument(
        "--decision-cost-window",
        type=float,
        default=60.0,
        help="Cost budget rolling window in seconds",
    )
    ap.add_argument("--phase6-log-dir", default="logs/phase6", help="Phase 6 gate log dir")
    ap.add_argument("--influence-log-dir", default="logs/asymmetry", help="Influence log dir")
    ap.add_argument(
        "--decision-sink",
        action="append",
        default=[],
        help="Decision sinks: file:/path.ndjson or tcp://host:port (repeatable)",
    )
    ap.add_argument(
        "--ohlc-sink",
        action="append",
        default=[],
        help="OHLC sinks: file:/path.ndjson or tcp://host:port (repeatable)",
    )
    ap.add_argument("--tail", dest="tail_path", default=None, help="Replay NDJSON file into daemon")
    ap.add_argument("--tail-follow", action="store_true", help="Follow tail file for appended data")
    ap.add_argument("--metrics-host", default=None, help="Metrics HTTP bind host")
    ap.add_argument("--metrics-port", type=int, default=None, help="Metrics HTTP bind port")
    args = ap.parse_args()

    daemon = StreamDaemon(
        host=args.host,
        port=args.port,
        db_path=args.db,
        batch_size=args.batch_size,
        flush_interval=args.flush_interval,
        summarise_window=args.summarise_window,
        emit_decisions=args.emit_decisions,
        decision_stdout=args.decision_stdout,
        state_window=args.state_window,
        tau_on=args.tau_on,
        tau_off=args.tau_off,
        decision_base_size=args.decision_base_size,
        decision_fee_rate=args.decision_fee_rate,
        decision_slippage=args.decision_slippage,
        decision_edge_bps=args.decision_edge_bps,
        decision_cost_budget=args.decision_cost_budget,
        decision_cost_window=args.decision_cost_window,
        phase6_log_dir=args.phase6_log_dir if args.emit_decisions else None,
        influence_log_dir=args.influence_log_dir if args.emit_decisions else None,
        decision_sinks=args.decision_sink if args.emit_decisions else [],
        ohlc_sinks=args.ohlc_sink,
        tail_path=args.tail_path,
        tail_follow=args.tail_follow,
        metrics_host=args.metrics_host,
        metrics_port=args.metrics_port,
    )

    def _handle_signal(_sig, _frame):
        daemon.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    daemon.start()


if __name__ == "__main__":
    main()

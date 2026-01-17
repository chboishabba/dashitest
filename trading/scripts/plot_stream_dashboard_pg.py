#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def _load_ndjson(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _prepare_decisions(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if symbols:
        df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "direction", "target_exposure"])
    df["signed_exposure"] = df["direction"].astype(float) * df["target_exposure"].astype(float)
    df.sort_values("timestamp", inplace=True)
    return df


def _prepare_ohlc(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "timestamp" not in df.columns and "ts_ms" in df.columns:
        df["timestamp"] = df["ts_ms"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    if symbols:
        df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return df
    return df.sort_values(["timestamp", "symbol"])


def _to_epoch_seconds(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, utc=True).astype(np.int64).to_numpy() / 1e9


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass
class PnLState:
    position: float = 0.0
    avg_price: float | None = None
    realized: float = 0.0
    fee_rate: float = 0.0005
    slippage: float = 0.0003
    min_trade: float = 0.02

    def apply_target(self, target: float, price: float) -> None:
        if np.isclose(target, self.position):
            return
        delta = target - self.position
        if abs(delta) < self.min_trade:
            return
        slip = self.slippage * (1 if delta > 0 else -1)
        fill_price = price * (1 + slip)
        fee = abs(delta) * price * self.fee_rate
        self.realized -= fee
        if self.position == 0.0:
            self.position = target
            self.avg_price = fill_price if target != 0.0 else None
            return

        pos_sign = _sign(self.position)
        tgt_sign = _sign(target)
        if pos_sign == tgt_sign and tgt_sign != 0:
            if abs(target) > abs(self.position):
                add_size = abs(target) - abs(self.position)
                total = abs(self.position) + add_size
                if total > 0:
                    self.avg_price = (
                        (self.avg_price or fill_price) * abs(self.position) + fill_price * add_size
                    ) / total
                self.position = target
                return
            if abs(target) < abs(self.position):
                closed = abs(self.position) - abs(target)
                self.realized += (fill_price - (self.avg_price or fill_price)) * closed * pos_sign
                self.position = target
                if self.position == 0.0:
                    self.avg_price = None
                return

        closed = abs(self.position)
        self.realized += (fill_price - (self.avg_price or fill_price)) * closed * pos_sign
        self.position = target
        self.avg_price = fill_price if target != 0.0 else None

    def unrealized(self, price: float) -> float:
        if self.position == 0.0 or self.avg_price is None:
            return 0.0
        return (price - self.avg_price) * self.position


class OHLCItem(pg.GraphicsObject):
    def __init__(self, bar_width: float = 0.35) -> None:
        super().__init__()
        self.bar_width = bar_width
        self.data = np.empty((0, 5), dtype=float)
        self.picture: QtGui.QPicture | None = None
        self.up_pen = pg.mkPen((0, 170, 0), width=1)
        self.down_pen = pg.mkPen((200, 0, 0), width=1)
        self.flat_pen = pg.mkPen((150, 150, 150), width=1)

    def setData(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=float)
        self._generate_picture()
        self.update()

    def _generate_picture(self) -> None:
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        for ts, open_, high, low, close in self.data:
            if close > open_:
                painter.setPen(self.up_pen)
            elif close < open_:
                painter.setPen(self.down_pen)
            else:
                painter.setPen(self.flat_pen)
            painter.drawLine(QtCore.QPointF(ts, low), QtCore.QPointF(ts, high))
            painter.drawLine(QtCore.QPointF(ts - self.bar_width, open_), QtCore.QPointF(ts, open_))
            painter.drawLine(QtCore.QPointF(ts, close), QtCore.QPointF(ts + self.bar_width, close))
        painter.end()
        self.picture = picture

    def paint(self, painter: QtGui.QPainter, _opt, _widget=None) -> None:
        if self.picture is None:
            return
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self) -> QtCore.QRectF:
        if self.picture is None:
            return QtCore.QRectF()
        return QtCore.QRectF(self.picture.boundingRect())


class LiveDashboard(QtWidgets.QMainWindow):
    def __init__(
        self,
        decisions_path: Path,
        db_path: Path | None,
        ohlc_path: Path | None,
        symbols: list[str],
        ohlc_symbol: str,
        interval_ms: int,
        window_seconds: float | None,
        fee_rate: float,
        slippage: float,
        min_trade: float,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Decision + OHLCV + PnL (PyQtGraph)")
        self.decisions_path = decisions_path
        self.db_path = db_path
        self.ohlc_path = ohlc_path
        self.interval_ms = max(200, int(interval_ms))
        self.window_seconds = window_seconds
        self.symbols = symbols
        self.ohlc_symbol = ohlc_symbol
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.min_trade = min_trade

        self.decisions_df = _prepare_decisions(_load_ndjson(decisions_path), symbols)
        self._open_tail()

        self.con = None
        if self.ohlc_path is None:
            if self.db_path is None:
                raise SystemExit("OHLC source required: provide --ohlc-ndjson or --db.")
            try:
                self.con = duckdb.connect(str(self.db_path), read_only=True)
            except duckdb.IOException as exc:
                raise SystemExit(
                    "DuckDB is locked by another process. Provide --ohlc-ndjson "
                    "or add --ohlc-sink file:PATH when running the daemon."
                ) from exc
        self.ohlc_fh = None
        if self.ohlc_path is not None:
            self._open_ohlc_tail()
        self.ohlc_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self.last_ohlc_ts: pd.Timestamp | None = None
        self.pnl_symbols = self._resolve_pnl_symbols(symbols)
        self.pnl_states = {
            symbol: PnLState(fee_rate=fee_rate, slippage=slippage, min_trade=min_trade)
            for symbol in self.pnl_symbols
        }
        self.pnl_last_bar_ts: dict[str, pd.Timestamp | None] = {symbol: None for symbol in self.pnl_symbols}
        self.pnl_last_close: dict[str, float] = {}
        self.pnl_decision_queue: dict[str, list[tuple[int, float]]] = {symbol: [] for symbol in self.pnl_symbols}
        self.pnl_decision_cursor: dict[str, int] = {symbol: 0 for symbol in self.pnl_symbols}
        self.pnl_last_decision_ts: dict[str, int] = {symbol: -1 for symbol in self.pnl_symbols}
        self.pnl_df = pd.DataFrame(columns=["timestamp", "total", "unrealized"])

        layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(layout)
        pg.setConfigOptions(antialias=True)

        self.exposure_plot = layout.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem(utcOffset=0)})
        self.exposure_plot.showGrid(x=True, y=True, alpha=0.25)
        self.exposure_plot.setLabel("left", "signed exposure")
        self.exposure_plot.setLabel("bottom", "time (UTC)")
        self.exposure_plot.addLegend(offset=(10, 10))
        self.exposure_plot.setMinimumHeight(240)

        self.delta_plot = layout.addPlot(row=1, col=0, axisItems={"bottom": pg.DateAxisItem(utcOffset=0)})
        self.delta_plot.showGrid(x=True, y=True, alpha=0.2)
        self.delta_plot.setLabel("left", "delta exposure")
        self.delta_plot.setMaximumHeight(140)
        self.delta_plot.setXLink(self.exposure_plot)
        self.delta_plot.addLine(y=0.0, pen=pg.mkPen((120, 120, 120), width=1))
        self.delta_plot.addLegend(offset=(10, 10))

        self.price_plot = layout.addPlot(row=2, col=0, axisItems={"bottom": pg.DateAxisItem(utcOffset=0)})
        self.price_plot.showGrid(x=True, y=True, alpha=0.25)
        self.price_plot.setLabel("left", f"{ohlc_symbol} price")
        self.price_plot.setMinimumHeight(200)
        self.price_plot.setXLink(self.exposure_plot)

        self.volume_plot = layout.addPlot(row=3, col=0, axisItems={"bottom": pg.DateAxisItem(utcOffset=0)})
        self.volume_plot.showGrid(x=True, y=True, alpha=0.2)
        self.volume_plot.setLabel("left", "volume")
        self.volume_plot.setMaximumHeight(120)
        self.volume_plot.setXLink(self.exposure_plot)

        self.pnl_plot = layout.addPlot(row=4, col=0, axisItems={"bottom": pg.DateAxisItem(utcOffset=0)})
        self.pnl_plot.showGrid(x=True, y=True, alpha=0.2)
        self.pnl_plot.setLabel("left", "PnL (aggregate)")
        self.pnl_plot.setLabel("bottom", "time (UTC)")
        self.pnl_plot.setMaximumHeight(160)
        self.pnl_plot.setXLink(self.exposure_plot)
        self.pnl_plot.addLine(y=0.0, pen=pg.mkPen((120, 120, 120), width=1))

        self.lines: dict[str, pg.PlotDataItem] = {}
        self.delta_lines: dict[str, pg.PlotDataItem] = {}
        symbols_for_lines = sorted(self.decisions_df["symbol"].unique())
        if not symbols_for_lines and symbols:
            symbols_for_lines = sorted(set(symbols))
        for symbol in symbols_for_lines:
            self._ensure_symbol(symbol)

        self.candle_item = OHLCItem()
        self.price_plot.addItem(self.candle_item)
        self.volume_item = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush(80, 120, 200, 160))
        self.volume_plot.addItem(self.volume_item)

        self.pnl_total_line = self.pnl_plot.plot([], [], pen=pg.mkPen((40, 200, 80), width=2), name="total")
        self.pnl_unreal_line = self.pnl_plot.plot(
            [],
            [],
            pen=pg.mkPen((200, 200, 200), width=1, style=QtCore.Qt.PenStyle.DashLine),
            name="unrealized",
        )

        self._prime_ohlc()
        self._prime_decisions()
        self._prime_pnl_history()
        self._render_all()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(self.interval_ms)

    def _open_tail(self) -> None:
        self.fh = self.decisions_path.open("r", encoding="utf-8")
        self.fh.seek(0, 2)

    def _open_ohlc_tail(self) -> None:
        if self.ohlc_path is None:
            return
        self.ohlc_fh = self.ohlc_path.open("r", encoding="utf-8")
        self.ohlc_fh.seek(0, 2)

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol in self.lines:
            return
        color = pg.intColor(hash(symbol) % 256, hues=256, values=200, minValue=120)
        line = self.exposure_plot.plot([], [], pen=pg.mkPen(color, width=2), name=symbol)
        self.lines[symbol] = line
        delta_line = self.delta_plot.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(color),
            symbolPen=None,
            name=symbol,
        )
        self.delta_lines[symbol] = delta_line

    def _resolve_pnl_symbols(self, symbols: list[str]) -> list[str]:
        if symbols:
            return symbols
        if not self.decisions_df.empty:
            return sorted(self.decisions_df["symbol"].unique())
        return [self.ohlc_symbol]

    def _prime_decisions(self) -> None:
        if self.decisions_df.empty:
            return
        self._enqueue_decisions(self.decisions_df)

    def _enqueue_decisions(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        for symbol, sym_df in df.groupby("symbol"):
            if symbol not in self.pnl_decision_queue:
                self.pnl_decision_queue[symbol] = []
                self.pnl_decision_cursor[symbol] = 0
                self.pnl_last_decision_ts[symbol] = -1
                self.pnl_states[symbol] = PnLState(
                    fee_rate=self.fee_rate,
                    slippage=self.slippage,
                    min_trade=self.min_trade,
                )
                self.pnl_last_bar_ts[symbol] = None
                self.pnl_symbols.append(symbol)
            queue = self.pnl_decision_queue[symbol]
            for ts, exposure in zip(sym_df["timestamp"], sym_df["signed_exposure"]):
                ts_ns = int(pd.Timestamp(ts).value)
                queue.append((ts_ns, float(exposure)))
            if len(queue) > 1 and queue[-1][0] < queue[-2][0]:
                queue.sort(key=lambda item: item[0])
                timestamps = [item[0] for item in queue]
                self.pnl_decision_cursor[symbol] = bisect.bisect_right(
                    timestamps,
                    self.pnl_last_decision_ts.get(symbol, -1),
                )

    def _read_new_decisions(self) -> pd.DataFrame:
        if self.fh is None:
            return pd.DataFrame()
        rows = []
        while True:
            line = self.fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            return pd.DataFrame()
        return _prepare_decisions(pd.DataFrame(rows), self.symbols)

    def _read_new_ohlc_rows(self) -> pd.DataFrame:
        if self.ohlc_fh is None:
            return pd.DataFrame()
        rows = []
        while True:
            line = self.ohlc_fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            return pd.DataFrame()
        symbols = sorted(set(self.pnl_symbols + [self.ohlc_symbol]))
        return _prepare_ohlc(pd.DataFrame(rows), symbols)

    def _prime_ohlc(self) -> None:
        if self.ohlc_path is not None:
            return
        if self.con is None:
            return
        max_ts = self.con.execute(
            "SELECT max(timestamp) FROM ohlc_1s WHERE symbol = ?",
            [self.ohlc_symbol],
        ).fetchone()[0]
        if max_ts is None:
            return
        max_ts = pd.to_datetime(max_ts, utc=True)
        cutoff = None
        if self.window_seconds is not None:
            cutoff = max_ts - pd.Timedelta(seconds=self.window_seconds)
        if cutoff is None:
            df = self.con.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlc_1s
                WHERE symbol = ?
                ORDER BY timestamp
                """,
                [self.ohlc_symbol],
            ).df()
        else:
            df = self.con.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlc_1s
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp
                """,
                [self.ohlc_symbol, cutoff],
            ).df()
        if df.empty:
            return
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        self.ohlc_df = df
        self.last_ohlc_ts = pd.to_datetime(df["timestamp"].max(), utc=True)
        if self.ohlc_symbol not in self.pnl_last_bar_ts:
            self.pnl_last_bar_ts[self.ohlc_symbol] = self.last_ohlc_ts

    def _prime_pnl_history(self) -> None:
        if not self.pnl_symbols:
            return
        if self.ohlc_path is not None:
            symbols = sorted(set(self.pnl_symbols + [self.ohlc_symbol]))
            raw = _prepare_ohlc(_load_ndjson(self.ohlc_path), symbols)
            self.pnl_states = {symbol: PnLState() for symbol in self.pnl_symbols}
            self.pnl_df = pd.DataFrame(columns=["timestamp", "total", "unrealized"])
            self.pnl_last_close = {}
            for symbol in self.pnl_symbols:
                self.pnl_last_decision_ts[symbol] = -1
                self.pnl_decision_cursor[symbol] = 0
            self._enqueue_decisions(self.decisions_df)
            if not raw.empty:
                self.ohlc_df = raw[raw["symbol"] == self.ohlc_symbol].copy()
                if not self.ohlc_df.empty:
                    self.last_ohlc_ts = pd.to_datetime(self.ohlc_df["timestamp"].max(), utc=True)
                for symbol in self.pnl_symbols:
                    sym_rows = raw[raw["symbol"] == symbol]
                    if sym_rows.empty:
                        continue
                    self.pnl_last_bar_ts[symbol] = pd.to_datetime(sym_rows["timestamp"].max(), utc=True)
                for _, row in raw.iterrows():
                    self._apply_pnl_tick(row["symbol"], pd.to_datetime(row["timestamp"], utc=True), float(row["close"]))
            return
        if self.con is None:
            return
        max_rows = self.con.execute(
            "SELECT symbol, max(timestamp) FROM ohlc_1s WHERE symbol IN (SELECT * FROM UNNEST(?)) GROUP BY symbol",
            [self.pnl_symbols],
        ).fetchall()
        if not max_rows:
            return
        max_map = {row[0]: pd.to_datetime(row[1], utc=True) for row in max_rows if row[1] is not None}
        if not max_map:
            return
        cutoff_map: dict[str, pd.Timestamp | None] = {}
        for symbol, max_ts in max_map.items():
            if self.window_seconds is None:
                cutoff_map[symbol] = None
            else:
                cutoff_map[symbol] = max_ts - pd.Timedelta(seconds=self.window_seconds)
        self.pnl_states = {
            symbol: PnLState(fee_rate=self.fee_rate, slippage=self.slippage, min_trade=self.min_trade)
            for symbol in self.pnl_symbols
        }
        self.pnl_df = pd.DataFrame(columns=["timestamp", "total", "unrealized"])
        self.pnl_last_close = {}
        for symbol in self.pnl_symbols:
            self.pnl_last_decision_ts[symbol] = -1
            self.pnl_decision_cursor[symbol] = 0
        self._enqueue_decisions(self.decisions_df)

        rows: list[tuple[pd.Timestamp, str, float]] = []
        for symbol in self.pnl_symbols:
            max_ts = max_map.get(symbol)
            if max_ts is None:
                continue
            cutoff = cutoff_map.get(symbol)
            if cutoff is None:
                df = self.con.execute(
                    "SELECT timestamp, close FROM ohlc_1s WHERE symbol = ? ORDER BY timestamp",
                    [symbol],
                ).df()
            else:
                df = self.con.execute(
                    "SELECT timestamp, close FROM ohlc_1s WHERE symbol = ? AND timestamp >= ? ORDER BY timestamp",
                    [symbol, cutoff],
                ).df()
            if df.empty:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            self.pnl_last_bar_ts[symbol] = pd.to_datetime(df["timestamp"].max(), utc=True)
            for ts, close in zip(df["timestamp"], df["close"]):
                rows.append((ts, symbol, float(close)))

        rows.sort(key=lambda item: (item[0], item[1]))
        for ts, symbol, close in rows:
            self._apply_pnl_tick(symbol, ts, close)

    def _apply_pnl_tick(self, symbol: str, ts: pd.Timestamp, close: float) -> None:
        state = self.pnl_states.setdefault(
            symbol,
            PnLState(fee_rate=self.fee_rate, slippage=self.slippage, min_trade=self.min_trade),
        )
        queue = self.pnl_decision_queue.setdefault(symbol, [])
        cursor = self.pnl_decision_cursor.setdefault(symbol, 0)
        ts_ns = int(ts.value)
        while cursor < len(queue):
            decision_ts, exposure = queue[cursor]
            if decision_ts > ts_ns:
                break
            state.apply_target(exposure, close)
            cursor += 1
            self.pnl_last_decision_ts[symbol] = decision_ts
        self.pnl_decision_cursor[symbol] = cursor
        self.pnl_last_close[symbol] = close

        unrealized = 0.0
        total = 0.0
        for sym, sym_state in self.pnl_states.items():
            last_close = self.pnl_last_close.get(sym)
            if last_close is None:
                continue
            unreal = sym_state.unrealized(last_close)
            unrealized += unreal
            total += sym_state.realized + unreal

        if not self.pnl_df.empty and self.pnl_df.iloc[-1]["timestamp"] == ts:
            self.pnl_df.iloc[-1, self.pnl_df.columns.get_loc("total")] = total
            self.pnl_df.iloc[-1, self.pnl_df.columns.get_loc("unrealized")] = unrealized
        else:
            self.pnl_df = pd.concat(
                [
                    self.pnl_df,
                    pd.DataFrame({"timestamp": [ts], "total": [total], "unrealized": [unrealized]}),
                ],
                ignore_index=True,
            )

    def _fetch_new_ohlc(self) -> pd.DataFrame:
        if self.con is None or self.last_ohlc_ts is None:
            return pd.DataFrame()
        return self.con.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlc_1s
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp
            """,
            [self.ohlc_symbol, self.last_ohlc_ts],
        ).df()

    def _fetch_new_pnl_rows(self) -> list[tuple[pd.Timestamp, str, float]]:
        if self.con is None:
            return []
        rows: list[tuple[pd.Timestamp, str, float]] = []
        for symbol in self.pnl_symbols:
            last_ts = self.pnl_last_bar_ts.get(symbol)
            if last_ts is None:
                continue
            df = self.con.execute(
                """
                SELECT timestamp, close
                FROM ohlc_1s
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp
                """,
                [symbol, last_ts],
            ).df()
            if df.empty:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            self.pnl_last_bar_ts[symbol] = pd.to_datetime(df["timestamp"].max(), utc=True)
            for ts, close in zip(df["timestamp"], df["close"]):
                rows.append((ts, symbol, float(close)))
        rows.sort(key=lambda item: (item[0], item[1]))
        return rows

    def _apply_window(self) -> None:
        if self.window_seconds is None:
            return
        if not self.decisions_df.empty:
            cutoff = self.decisions_df["timestamp"].max() - pd.Timedelta(seconds=self.window_seconds)
            self.decisions_df = self.decisions_df[self.decisions_df["timestamp"] >= cutoff]
        if not self.ohlc_df.empty:
            cutoff = self.ohlc_df["timestamp"].max() - pd.Timedelta(seconds=self.window_seconds)
            self.ohlc_df = self.ohlc_df[self.ohlc_df["timestamp"] >= cutoff]
        if not self.pnl_df.empty:
            cutoff = self.pnl_df["timestamp"].max() - pd.Timedelta(seconds=self.window_seconds)
            self.pnl_df = self.pnl_df[self.pnl_df["timestamp"] >= cutoff]

    def _render_exposures(self) -> None:
        if self.decisions_df.empty:
            return
        frame = self.decisions_df
        for symbol in sorted(frame["symbol"].unique()):
            self._ensure_symbol(symbol)
        for symbol, line in self.lines.items():
            sym_df = frame[frame["symbol"] == symbol]
            if sym_df.empty:
                line.setData([], [])
                delta_line = self.delta_lines.get(symbol)
                if delta_line is not None:
                    delta_line.setData([], [])
                continue
            x = _to_epoch_seconds(sym_df["timestamp"])
            y = sym_df["signed_exposure"].to_numpy()
            line.setData(x, y, stepMode="right")
            delta_line = self.delta_lines.get(symbol)
            if delta_line is not None:
                delta = np.diff(y, prepend=y[0])
                mask = ~np.isclose(delta, 0.0)
                delta_line.setData(x[mask], delta[mask])

    def _render_ohlc(self) -> None:
        if self.ohlc_df.empty:
            return
        times = _to_epoch_seconds(self.ohlc_df["timestamp"])
        ohlc = np.column_stack(
            (
                times,
                self.ohlc_df["open"].to_numpy(dtype=float),
                self.ohlc_df["high"].to_numpy(dtype=float),
                self.ohlc_df["low"].to_numpy(dtype=float),
                self.ohlc_df["close"].to_numpy(dtype=float),
            )
        )
        self.candle_item.setData(ohlc)
        self.volume_item.setOpts(
            x=times,
            height=self.ohlc_df["volume"].to_numpy(dtype=float),
            width=0.8,
        )
        self.price_plot.enableAutoRange()
        self.volume_plot.enableAutoRange()

    def _render_pnl(self) -> None:
        if self.pnl_df.empty:
            self.pnl_total_line.setData([], [])
            self.pnl_unreal_line.setData([], [])
            return
        times = _to_epoch_seconds(self.pnl_df["timestamp"])
        total = self.pnl_df["total"].to_numpy(dtype=float)
        unreal = self.pnl_df["unrealized"].to_numpy(dtype=float)
        self.pnl_total_line.setData(times, total)
        self.pnl_unreal_line.setData(times, unreal)
        self.pnl_plot.enableAutoRange()

    def _render_all(self) -> None:
        self._apply_window()
        self._render_exposures()
        self._render_ohlc()
        self._render_pnl()

    def refresh(self) -> None:
        new_decisions = self._read_new_decisions()
        if not new_decisions.empty:
            if self.decisions_df.empty:
                self.decisions_df = new_decisions
            else:
                self.decisions_df = pd.concat([self.decisions_df, new_decisions], ignore_index=True)
            self._enqueue_decisions(new_decisions)

        new_ohlc = pd.DataFrame()
        new_pnl_rows: list[tuple[pd.Timestamp, str, float]] = []
        if self.ohlc_path is not None:
            if any(ts is None for ts in self.pnl_last_bar_ts.values()):
                self._prime_pnl_history()
            new_ohlc = self._read_new_ohlc_rows()
            if not new_ohlc.empty:
                ohlc_plot = new_ohlc[new_ohlc["symbol"] == self.ohlc_symbol]
                if not ohlc_plot.empty:
                    if self.ohlc_df.empty:
                        self.ohlc_df = ohlc_plot
                    else:
                        self.ohlc_df = pd.concat([self.ohlc_df, ohlc_plot], ignore_index=True)
                    self.last_ohlc_ts = pd.to_datetime(self.ohlc_df["timestamp"].max(), utc=True)
                for _, row in new_ohlc.iterrows():
                    self._apply_pnl_tick(row["symbol"], pd.to_datetime(row["timestamp"], utc=True), float(row["close"]))
        else:
            if self.last_ohlc_ts is None:
                self._prime_ohlc()
            if any(ts is None for ts in self.pnl_last_bar_ts.values()):
                self._prime_pnl_history()
            new_ohlc = self._fetch_new_ohlc()
            if not new_ohlc.empty:
                new_ohlc["timestamp"] = pd.to_datetime(new_ohlc["timestamp"], utc=True)
                if self.ohlc_df.empty:
                    self.ohlc_df = new_ohlc
                else:
                    self.ohlc_df = pd.concat([self.ohlc_df, new_ohlc], ignore_index=True)
                self.last_ohlc_ts = pd.to_datetime(self.ohlc_df["timestamp"].max(), utc=True)
            new_pnl_rows = self._fetch_new_pnl_rows()
            for ts, symbol, close in new_pnl_rows:
                self._apply_pnl_tick(symbol, ts, close)

        if not new_decisions.empty or not new_ohlc.empty or new_pnl_rows:
            self._render_all()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live dashboard: decisions + OHLCV + PnL (PyQtGraph).")
    ap.add_argument("--ndjson", required=True, help="Decision NDJSON path")
    ap.add_argument("--db", default=None, help="DuckDB path with ohlc_1s table")
    ap.add_argument("--ohlc-ndjson", default=None, help="OHLC NDJSON path")
    ap.add_argument("--fee-rate", type=float, default=0.0005, help="Fee rate per exposure change")
    ap.add_argument("--slippage", type=float, default=0.0003, help="Slippage rate (fraction of price)")
    ap.add_argument("--min-trade", type=float, default=0.02, help="Minimum exposure change to trade")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbol filter for exposure")
    ap.add_argument("--ohlc-symbol", required=True, help="Symbol to plot OHLCV + PnL")
    ap.add_argument("--interval-ms", type=int, default=1000, help="Update interval in ms")
    ap.add_argument("--window-seconds", type=float, default=600.0, help="Rolling window in seconds")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else []

    app = QtWidgets.QApplication([])
    ohlc_path = Path(args.ohlc_ndjson) if args.ohlc_ndjson else None
    db_path = Path(args.db) if args.db else None
    win = LiveDashboard(
        Path(args.ndjson),
        db_path,
        ohlc_path,
        symbols,
        args.ohlc_symbol,
        args.interval_ms,
        args.window_seconds,
        args.fee_rate,
        args.slippage,
        args.min_trade,
    )
    win.show()
    app.exec()


if __name__ == "__main__":
    main()

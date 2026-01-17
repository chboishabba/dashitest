#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


@dataclass
class SymbolPanel:
    symbol: str
    price_plot: pg.PlotItem
    volume_plot: pg.PlotItem
    candle_item: OHLCItem
    volume_item: pg.BarGraphItem
    data: pd.DataFrame
    last_ts: int | None = None


class LiveOhlcvPlot(QtWidgets.QMainWindow):
    def __init__(
        self,
        db_path: Path | None,
        ndjson_path: Path | None,
        symbols: list[str],
        interval_ms: int,
        window_seconds: float | None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("OHLCV stream (PyQtGraph)")
        self.db_path = db_path
        self.ndjson_path = ndjson_path
        self.interval_ms = max(200, int(interval_ms))
        self.window_seconds = window_seconds
        self.con = None
        self.ndjson_fh = None
        if self.ndjson_path is None:
            if self.db_path is None:
                raise SystemExit("OHLC source required: provide --ndjson or --db.")
            try:
                self.con = duckdb.connect(str(self.db_path), read_only=True)
            except duckdb.IOException as exc:
                raise SystemExit(
                    "DuckDB is locked by another process. Provide --ndjson "
                    "or add --ohlc-sink file:PATH when running the daemon."
                ) from exc
        else:
            self._open_tail()

        if not symbols:
            symbols = self._fetch_symbols()
        self.symbols = symbols
        if not self.symbols:
            raise SystemExit("No symbols found to plot.")

        layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(layout)
        pg.setConfigOptions(antialias=True)

        self.panels: dict[str, SymbolPanel] = {}
        row = 0
        for symbol in self.symbols:
            axis = pg.DateAxisItem(utcOffset=0)
            price_plot = layout.addPlot(row=row, col=0, axisItems={"bottom": axis})
            price_plot.showGrid(x=True, y=True, alpha=0.25)
            price_plot.setLabel("left", symbol)
            price_plot.setLabel("bottom", "time (UTC)")
            candle_item = OHLCItem()
            price_plot.addItem(candle_item)

            row += 1
            volume_axis = pg.DateAxisItem(utcOffset=0)
            volume_plot = layout.addPlot(row=row, col=0, axisItems={"bottom": volume_axis})
            volume_plot.showGrid(x=True, y=True, alpha=0.2)
            volume_plot.setLabel("left", "volume")
            volume_plot.setMaximumHeight(120)
            volume_plot.setXLink(price_plot)
            volume_item = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush(80, 120, 200, 160))
            volume_plot.addItem(volume_item)

            panel = SymbolPanel(
                symbol=symbol,
                price_plot=price_plot,
                volume_plot=volume_plot,
                candle_item=candle_item,
                volume_item=volume_item,
                data=pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
                last_ts=None,
            )
            self.panels[symbol] = panel
            row += 1

        self._prime()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(self.interval_ms)

    def _fetch_symbols(self) -> list[str]:
        if self.con is None:
            if self.ndjson_path is None:
                return []
            df = _prepare_ohlc(_load_ndjson(self.ndjson_path), [])
            return sorted(df["symbol"].unique()) if not df.empty else []
        rows = self.con.execute("SELECT DISTINCT symbol FROM ohlc_1s ORDER BY symbol").fetchall()
        return [row[0] for row in rows if row and row[0]]

    def _open_tail(self) -> None:
        if self.ndjson_path is None:
            return
        self.ndjson_fh = self.ndjson_path.open("r", encoding="utf-8")
        self.ndjson_fh.seek(0, 2)

    def _read_new_ndjson(self) -> pd.DataFrame:
        if self.ndjson_fh is None:
            return pd.DataFrame()
        rows = []
        while True:
            line = self.ndjson_fh.readline()
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
        return _prepare_ohlc(pd.DataFrame(rows), self.symbols)

    def _prime(self) -> None:
        if self.con is None:
            if self.ndjson_path is None:
                return
            df = _prepare_ohlc(_load_ndjson(self.ndjson_path), self.symbols)
            if df.empty:
                return
            for panel in self.panels.values():
                sym_df = df[df["symbol"] == panel.symbol]
                if sym_df.empty:
                    continue
                panel.data = sym_df
                panel.last_ts = int(sym_df["timestamp"].astype("int64").max())
                self._render(panel)
            return
        for panel in self.panels.values():
            max_ts = self.con.execute(
                "SELECT max(timestamp) FROM ohlc_1s WHERE symbol = ?",
                [panel.symbol],
            ).fetchone()[0]
            if max_ts is None:
                continue
            cutoff = None
            if self.window_seconds is not None:
                cutoff = int(max_ts - self.window_seconds * 1000)
            if cutoff is None:
                df = self.con.execute(
                    """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlc_1s
                    WHERE symbol = ?
                    ORDER BY timestamp
                    """,
                    [panel.symbol],
                ).df()
            else:
                df = self.con.execute(
                    """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlc_1s
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp
                    """,
                    [panel.symbol, cutoff],
                ).df()
            if df.empty:
                continue
            panel.data = df
            panel.last_ts = int(df["timestamp"].max())
            self._render(panel)

    def _fetch_new(self, symbol: str, last_ts: int | None) -> pd.DataFrame:
        if self.con is None:
            return pd.DataFrame()
        if last_ts is None:
            last_ts = -1
        return self.con.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlc_1s
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp
            """,
            [symbol, last_ts],
        ).df()

    def _render(self, panel: SymbolPanel) -> None:
        if panel.data.empty:
            return
        if np.issubdtype(panel.data["timestamp"].dtype, np.datetime64):
            times = panel.data["timestamp"].astype("int64").to_numpy() / 1e9
        else:
            times = panel.data["timestamp"].to_numpy(dtype=np.int64) / 1000.0
        ohlc = np.column_stack(
            (
                times,
                panel.data["open"].to_numpy(dtype=float),
                panel.data["high"].to_numpy(dtype=float),
                panel.data["low"].to_numpy(dtype=float),
                panel.data["close"].to_numpy(dtype=float),
            )
        )
        panel.candle_item.setData(ohlc)
        panel.volume_item.setOpts(
            x=times,
            height=panel.data["volume"].to_numpy(dtype=float),
            width=0.8,
        )
        panel.price_plot.enableAutoRange()
        panel.volume_plot.enableAutoRange()

    def refresh(self) -> None:
        if self.con is None:
            new_df = self._read_new_ndjson()
            if new_df.empty:
                return
            for panel in self.panels.values():
                sym_df = new_df[new_df["symbol"] == panel.symbol]
                if sym_df.empty:
                    continue
                if panel.data.empty:
                    panel.data = sym_df
                else:
                    panel.data = pd.concat([panel.data, sym_df], ignore_index=True)
                panel.last_ts = int(panel.data["timestamp"].astype("int64").max())
                if self.window_seconds is not None:
                    cutoff = panel.data["timestamp"].max() - pd.Timedelta(seconds=self.window_seconds)
                    panel.data = panel.data[panel.data["timestamp"] >= cutoff]
                self._render(panel)
            return
        for panel in self.panels.values():
            new_df = self._fetch_new(panel.symbol, panel.last_ts)
            if new_df.empty:
                continue
            panel.data = pd.concat([panel.data, new_df], ignore_index=True)
            panel.last_ts = int(panel.data["timestamp"].max())
            if self.window_seconds is not None:
                cutoff = panel.last_ts - int(self.window_seconds * 1000)
                panel.data = panel.data[panel.data["timestamp"] >= cutoff]
            self._render(panel)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live OHLCV plotter (PyQtGraph).")
    ap.add_argument("--db", default=None, help="DuckDB path with ohlc_1s table")
    ap.add_argument("--ndjson", default=None, help="OHLC NDJSON path")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbol list")
    ap.add_argument("--interval-ms", type=int, default=1000, help="Refresh interval in ms")
    ap.add_argument("--window-seconds", type=float, default=600.0, help="Rolling window in seconds")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else []
    db_path = Path(args.db) if args.db else None
    ndjson_path = Path(args.ndjson) if args.ndjson else None

    app = QtWidgets.QApplication([])
    win = LiveOhlcvPlot(db_path, ndjson_path, symbols, args.interval_ms, args.window_seconds)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()

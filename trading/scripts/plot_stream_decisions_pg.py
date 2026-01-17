#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


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


def _prepare_frame(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
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


class LivePlot(QtWidgets.QMainWindow):
    def __init__(
        self,
        df: pd.DataFrame,
        path: Path,
        interval_ms: int,
        window_seconds: float | None,
        show_state: bool,
        urgency_thickness: bool,
        show_posture: bool,
        symbols: list[str],
    ) -> None:
        super().__init__()
        self.setWindowTitle("Decision stream (PyQtGraph)")
        self.path = path
        self.interval_ms = max(50, int(interval_ms))
        self.window_seconds = window_seconds
        self.show_state = show_state
        self.urgency_thickness = urgency_thickness
        self.show_posture = show_posture
        self.symbols = symbols
        self.df = df
        self.fh = None
        self._open_tail()

        self.plot = pg.PlotWidget()
        self.setCentralWidget(self.plot)
        pg.setConfigOptions(antialias=True)
        self.plot.addLegend(offset=(10, 10))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "time (UTC)")
        self.plot.setLabel("left", "signed exposure")

        self.lines = {}
        self.state_pos = {}
        self.state_neg = {}
        self.urgency = {}
        self.posture_spans = []
        for symbol in sorted(df["symbol"].unique()):
            self._ensure_symbol(symbol)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(self.interval_ms)
        self.refresh(initial=True)

    def _open_tail(self) -> None:
        self.fh = self.path.open("r", encoding="utf-8")
        self.fh.seek(0, 2)

    def _read_new_rows(self) -> pd.DataFrame:
        if self.fh is None:
            return pd.DataFrame()
        lines = []
        while True:
            line = self.fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not lines:
            return pd.DataFrame()
        df_new = pd.DataFrame(lines)
        return _prepare_frame(df_new, self.symbols)

    def _apply_window(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.window_seconds is None or frame.empty:
            return frame
        cutoff = frame["timestamp"].max() - pd.Timedelta(seconds=self.window_seconds)
        return frame[frame["timestamp"] >= cutoff]

    def _clear_posture(self) -> None:
        for item in self.posture_spans:
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self.posture_spans = []

    def _add_posture(self, frame: pd.DataFrame) -> None:
        if not self.show_posture or frame.empty or "posture" not in frame.columns:
            return
        self._clear_posture()
        times = frame["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9
        posture = frame["posture"].fillna(0).to_numpy(dtype=int)
        if len(times) < 2 or len(posture) != len(times):
            return
        start_idx = 0
        step = float(np.median(np.diff(times)))
        colors = {
            -1: (200, 0, 0, 50),
            0: (120, 120, 120, 50),
            1: (0, 200, 0, 50),
        }
        for i in range(1, len(posture)):
            if posture[i] != posture[start_idx]:
                self._span(times[start_idx], times[i - 1], posture[start_idx], step, colors)
                start_idx = i
        self._span(times[start_idx], times[-1], posture[start_idx], step, colors)

    def _span(self, x_start: float, x_end: float, val: int, step: float, colors: dict[int, tuple[int, ...]]) -> None:
        rect = QtWidgets.QGraphicsRectItem(x_start, -1e9, (x_end - x_start) + step, 2e9)
        rect.setBrush(pg.mkBrush(*colors.get(val, (80, 80, 80, 40))))
        rect.setPen(pg.mkPen(None))
        rect.setZValue(-10)
        self.plot.addItem(rect, ignoreBounds=True)
        self.posture_spans.append(rect)

    def _ensure_symbol(self, symbol: str) -> None:
        if symbol in self.lines:
            return
        color = pg.intColor(hash(symbol) % 256, hues=256, values=200, minValue=120)
        line = self.plot.plot([], [], pen=pg.mkPen(color, width=2), name=symbol)
        self.lines[symbol] = line
        if self.show_state:
            self.state_pos[symbol] = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(0, 200, 0, 160), size=7)
            self.state_neg[symbol] = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(200, 0, 0, 160), size=7)
            self.plot.addItem(self.state_pos[symbol])
            self.plot.addItem(self.state_neg[symbol])
        if self.urgency_thickness:
            self.urgency[symbol] = pg.ScatterPlotItem(pen=None, brush=color, size=5, pxMode=True)
            self.plot.addItem(self.urgency[symbol])

    def refresh(self, initial: bool = False) -> None:
        new_df = self._read_new_rows()
        if initial:
            pass
        if not new_df.empty:
            if self.df.empty:
                self.df = new_df
            else:
                self.df = pd.concat([self.df, new_df], ignore_index=True)
        frame = self._apply_window(self.df)
        if frame.empty:
            return

        self._add_posture(frame)
        for symbol in sorted(frame["symbol"].unique()):
            self._ensure_symbol(symbol)
        for symbol, line in self.lines.items():
            sym_df = frame[frame["symbol"] == symbol]
            if sym_df.empty:
                line.setData([], [])
                if symbol in self.state_pos:
                    self.state_pos[symbol].setData([], [])
                    self.state_neg[symbol].setData([], [])
                if symbol in self.urgency:
                    self.urgency[symbol].setData([], [])
                continue
            x = sym_df["timestamp"].astype(np.int64).to_numpy() / 1e9
            y = sym_df["signed_exposure"].to_numpy()
            line.setData(x, y)
            if symbol in self.urgency and "urgency" in sym_df.columns:
                urg = sym_df["urgency"].fillna(0.0).clip(0.0, 1.0).to_numpy()
                self.urgency[symbol].setData(x, y, size=6 + 14 * urg)
            if symbol in self.state_pos and "state" in sym_df.columns:
                pos = sym_df[sym_df["state"] == 1]
                neg = sym_df[sym_df["state"] == -1]
                pos_x = pos["timestamp"].astype(np.int64).to_numpy() / 1e9
                neg_x = neg["timestamp"].astype(np.int64).to_numpy() / 1e9
                self.state_pos[symbol].setData(pos_x, pos["signed_exposure"].to_numpy())
                self.state_neg[symbol].setData(neg_x, neg["signed_exposure"].to_numpy())
        self.plot.enableAutoRange()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live decision plotter (PyQtGraph).")
    ap.add_argument("--ndjson", required=True, help="Decision NDJSON path")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbol filter")
    ap.add_argument("--interval-ms", type=int, default=1000, help="Update interval in ms")
    ap.add_argument("--window-seconds", type=float, default=None, help="Keep last N seconds visible")
    ap.add_argument("--show-state", action="store_true", help="Overlay state markers")
    ap.add_argument("--urgency-thickness", action="store_true", help="Scale marker size by urgency")
    ap.add_argument("--show-posture", action="store_true", help="Shade posture periods")
    ap.add_argument("--follow", action="store_true", help="Ignored (always tails file)")
    args = ap.parse_args()

    path = Path(args.ndjson)
    if not path.exists():
        raise SystemExit(f"NDJSON not found: {path}")
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else []
    df = _prepare_frame(_load_ndjson(path), symbols)

    app = QtWidgets.QApplication([])
    win = LivePlot(
        df,
        path,
        args.interval_ms,
        args.window_seconds,
        args.show_state,
        args.urgency_thickness,
        args.show_posture,
        symbols,
    )
    win.show()
    app.exec()


if __name__ == "__main__":
    main()

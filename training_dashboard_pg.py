"""
training_dashboard_pg.py
------------------------
PyQtGraph-based live dashboard for triadic trading logs (fast, multi-pane).

Panes:
 1) Price with buy/sell markers and bad_flag shading.
 2) PnL with rolling HOLD%.
 3) p_bad + bad_flag (step fill).
 4) Volume (if present).

Usage:
  PYTHONPATH=. python training_dashboard_pg.py --log logs/trading_log.csv --refresh 1.0
  # Progressive day-by-day reveal (each refresh adds the next day if ts present)
  PYTHONPATH=. python training_dashboard_pg.py --log logs/trading_log.csv --refresh 1.0 --progressive-days
"""

import argparse
import pathlib
import pandas as pd
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui


def load_log(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def synthetic_log(n=1000):
    t = np.arange(n)
    price = 100 + np.cumsum(np.random.normal(0, 0.2, size=n))
    pnl = 100000 + np.cumsum(np.random.normal(0, 1, size=n))
    p_bad = np.clip(np.random.beta(2, 5, size=n), 0, 1)
    bad_flag = (p_bad > 0.7).astype(int)
    action = np.random.choice([-1, 0, 1], size=n, p=[0.1, 0.7, 0.2])
    hold = (action == 0).astype(int)
    volume = np.random.randint(1e5, 2e5, size=n)
    return pd.DataFrame(
        {
            "t": t,
            "price": price,
            "pnl": pnl,
            "p_bad": p_bad,
            "bad_flag": bad_flag,
            "action": action,
            "hold": hold,
            "volume": volume,
        }
    )


def rolling_hold(log, window=100):
    if "hold" not in log:
        return None
    return log["hold"].rolling(window, min_periods=1).mean()


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, log_path: pathlib.Path, refresh_s: float, progressive_days: bool):
        super().__init__()
        self.log_path = log_path
        self.refresh_s = refresh_s
        self.progressive_days = progressive_days
        self.day_idx = 0
        self.day_keys = None
        self.posture_spans = []
        self.init_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        if refresh_s > 0:
        self.timer.start(int(refresh_s * 1000))
        self.update_data()

    def _parse_ts(self, log):
        if "ts" not in log.columns:
            return None
        ts = log["ts"]
        if pd.api.types.is_numeric_dtype(ts):
            max_abs = ts.abs().max()
            if pd.isna(max_abs):
                return pd.to_datetime(ts, errors="coerce")
            if max_abs >= 1e18:
                unit = "ns"
            elif max_abs >= 1e15:
                unit = "us"
            elif max_abs >= 1e12:
                unit = "ms"
            else:
                unit = "s"
            return pd.to_datetime(ts, unit=unit, errors="coerce")
        return pd.to_datetime(ts, errors="coerce")

    def init_ui(self):
        self.win = pg.GraphicsLayoutWidget(show=True, title="Triadic Trading Dashboard (PyQtGraph)")
        self.setCentralWidget(self.win)
        pg.setConfigOptions(antialias=True)

        # Price pane
        self.p_price = self.win.addPlot(row=0, col=0, title="Price + actions + bad_flag")
        self.price_curve = self.p_price.plot(pen=pg.mkPen("w", width=1))
        self.buy_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(0, 200, 0, 160), size=5)
        self.sell_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(200, 0, 0, 160), size=5)
        self.p_price.addItem(self.buy_scatter)
        self.p_price.addItem(self.sell_scatter)
        self.bad_fill = self.p_price.plot(pen=None, fillLevel=0, brush=pg.mkBrush(255, 0, 0, 40), stepMode=True)

        # PnL pane
        self.p_pnl = self.win.addPlot(row=1, col=0, title="PnL + HOLD%")
        self.pnl_curve = self.p_pnl.plot(pen=pg.mkPen("y", width=1))
        dash_style = getattr(QtCore.Qt, "DashLine", QtCore.Qt.PenStyle.DashLine)
        self.hold_curve = self.p_pnl.plot(pen=pg.mkPen("c", width=1, style=dash_style))

        # p_bad pane
        self.p_bad_plot = self.win.addPlot(row=2, col=0, title="p_bad + bad_flag")
        self.p_bad_curve = self.p_bad_plot.plot(pen=pg.mkPen("m", width=1))
        self.bad_flag_curve = self.p_bad_plot.plot(
            pen=pg.mkPen((255, 80, 80), width=1),
            fillLevel=0,
            brush=pg.mkBrush(255, 0, 0, 50),
            stepMode=True,
        )

        # Volume pane
        self.p_vol = self.win.addPlot(row=3, col=0, title="Volume")
        self.vol_curve = self.p_vol.plot(pen=pg.mkPen("g", width=1))

        # Posture pane (ACT/HOLD/BAN shading)
        self.p_posture = self.win.addPlot(row=4, col=0, title="Posture (BAN/HOLD/ACT)")
        self.p_posture.setYRange(-1.5, 1.5)
        self.p_posture.setMouseEnabled(y=False)
        self.p_posture.hideAxis("left")
        self.p_posture.showGrid(x=True, y=False, alpha=0.3)
        self.p_posture.setXLink(self.p_price)

    def _clear_posture_spans(self):
        for item in self.posture_spans:
            try:
                self.p_posture.removeItem(item)
            except Exception:
                pass
        self.posture_spans = []

    def _add_posture_spans(self, x_arr, g_arr):
        """
        Shade posture runs as translucent rectangles for readability.
        g_arr: {-1,0,1} → BAN/HOLD/ACT
        """
        if len(x_arr) == 0 or len(g_arr) == 0:
            return

        # Use a representative step size to extend rectangles.
        if len(x_arr) > 1:
            step = float(np.median(np.diff(x_arr)))
            step = step if step > 0 else 1.0
        else:
            step = 1.0

        colors = {
            -1: (200, 0, 0, 60),    # BAN
            0: (120, 120, 120, 60), # HOLD
            1: (0, 200, 0, 60),     # ACT
        }

        runs = []
        start_idx = 0
        for i in range(1, len(g_arr)):
            if g_arr[i] != g_arr[start_idx]:
                runs.append((start_idx, i - 1, g_arr[start_idx]))
                start_idx = i
        runs.append((start_idx, len(g_arr) - 1, g_arr[start_idx]))

        for s, e, val in runs:
            x_start = float(x_arr[s])
            x_end = float(x_arr[e])
            rect = QtWidgets.QGraphicsRectItem(
                x_start, -1.5, (x_end - x_start) + step, 3.0
            )
            rect.setBrush(pg.mkBrush(*colors.get(val, (80, 80, 80, 40))))
            rect.setPen(pg.mkPen(None))
            rect.setZValue(-10)
            self.p_posture.addItem(rect)
            self.posture_spans.append(rect)

    def update_data(self):
        log = load_log(self.log_path)
        if log is None or log.empty:
            log = synthetic_log()

        # Progressive reveal by day (ts required)
        ts_dt = self._parse_ts(log)

        if self.progressive_days and ts_dt is not None and ts_dt.notna().any():
            ts_days = ts_dt.dt.normalize()
            day_keys = ts_days.dropna().unique()
            day_keys.sort()
            if self.day_keys is None or len(self.day_keys) != len(day_keys) or not np.array_equal(
                self.day_keys, day_keys
            ):
                self.day_keys = day_keys
            if self.day_keys is not None and len(self.day_keys) > 0:
                # Clamp day index
                self.day_idx = min(self.day_idx, len(self.day_keys) - 1)
                visible_days = set(self.day_keys[: self.day_idx + 1])
                mask = ts_days.isin(visible_days)
                log = log.loc[mask]
                # Advance for next frame if possible
                if self.refresh_s > 0 and self.day_idx < len(self.day_keys) - 1:
                    self.day_idx += 1

        if ts_dt is not None and ts_dt.notna().any():
            x = ts_dt.loc[log.index]
        else:
            x = log["ts"] if "ts" in log.columns and log["ts"].notna().any() else log["t"]
        price = log["price"]
        pnl = log["pnl"] if "pnl" in log else None
        p_bad = log["p_bad"] if "p_bad" in log else None
        bad_flag = log["bad_flag"] if "bad_flag" in log else None
        action = log["action"] if "action" in log else None
        volume = log["volume"] if "volume" in log else None
        hold_roll = rolling_hold(log, window=200)
        posture = None
        if "ban" in log or "hold" in log or "action" in log:
            g = np.ones(len(log), dtype=int)  # default ACT
            if "ban" in log:
                g[np.array(log["ban"]) > 0] = -1
            if "hold" in log:
                mask = (g != -1) & (np.array(log["hold"]) > 0)
                g[mask] = 0
            elif "action" in log:
                mask = (g != -1) & (np.array(log["action"]) == 0)
                g[mask] = 0
            posture = g

        self.price_curve.setData(x, price)
        step_x = None
        if bad_flag is not None:
            x_arr = np.array(x)
            y_arr = np.array(bad_flag)

            # If timestamps are non-numeric, fall back to evenly spaced indices.
            if not np.issubdtype(x_arr.dtype, np.number):
                x_arr = np.arange(len(y_arr))

            # Build X for stepMode=True: len(x) = len(y) + 1.
            if len(x_arr) == len(y_arr):
                # Use the last step size if available; otherwise default to 1.
                step = x_arr[-1] - x_arr[-2] if len(x_arr) > 1 else 1
                x_arr = np.append(x_arr, x_arr[-1] + step)
            elif len(x_arr) > len(y_arr) + 1:
                # Trim any excess so we don't trip the length check.
                x_arr = x_arr[: len(y_arr) + 1]
            elif len(x_arr) < len(y_arr) + 1:
                # Pad forward if timestamps are missing.
                step = x_arr[-1] - x_arr[-2] if len(x_arr) > 1 else 1
                pad_count = len(y_arr) + 1 - len(x_arr)
                pad = x_arr[-1] + step * np.arange(1, pad_count + 1)
                x_arr = np.append(x_arr, pad)
            step_x = x_arr
            self.bad_fill.setData(step_x, y_arr)
        if action is not None:
            buys = action == 1
            sells = action == -1
            self.buy_scatter.setData(x[buys], price[buys])
            self.sell_scatter.setData(x[sells], price[sells])

        if pnl is not None:
            self.pnl_curve.setData(x, pnl)
        if hold_roll is not None:
            self.hold_curve.setData(x, hold_roll)

        if p_bad is not None:
            self.p_bad_curve.setData(x, p_bad)
        if bad_flag is not None:
            # Reuse the step-aligned x for the bad_flag panel if available.
            if step_x is None:
                x_arr = np.array(x)
                y_arr = np.array(bad_flag)
                if not np.issubdtype(x_arr.dtype, np.number):
                    x_arr = np.arange(len(y_arr))
                if len(x_arr) == len(y_arr):
                    step = x_arr[-1] - x_arr[-2] if len(x_arr) > 1 else 1
                    x_arr = np.append(x_arr, x_arr[-1] + step)
                elif len(x_arr) > len(y_arr) + 1:
                    x_arr = x_arr[: len(y_arr) + 1]
                elif len(x_arr) < len(y_arr) + 1:
                    step = x_arr[-1] - x_arr[-2] if len(x_arr) > 1 else 1
                    pad_count = len(y_arr) + 1 - len(x_arr)
                    pad = x_arr[-1] + step * np.arange(1, pad_count + 1)
                    x_arr = np.append(x_arr, pad)
                step_x = x_arr
            self.bad_flag_curve.setData(step_x, bad_flag)

        if volume is not None:
            self.vol_curve.setData(x, volume)

        # Posture shading pane
        self._clear_posture_spans()
        if posture is not None:
            x_arr = np.array(x)
            if not np.issubdtype(x_arr.dtype, np.number):
                x_arr = np.arange(len(posture))
            self._add_posture_spans(x_arr, posture)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="logs/trading_log.csv", help="CSV log file path")
    ap.add_argument("--refresh", type=float, default=1.0, help="Refresh seconds; set 0 for one-shot")
    ap.add_argument(
        "--progressive-days",
        action="store_true",
        help="Progressively reveal data day-by-day on each refresh (requires ts column).",
    )
    args = ap.parse_args()

    log_path = pathlib.Path(args.log)
    app = QtWidgets.QApplication([])
    dash = Dashboard(log_path=log_path, refresh_s=args.refresh, progressive_days=args.progressive_days)
    dash.show()
    if args.refresh <= 0:
        dash.update_data()
    QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    main()

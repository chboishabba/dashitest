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
"""

import argparse
import pathlib
import pandas as pd
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


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
    def __init__(self, log_path: pathlib.Path, refresh_s: float):
        super().__init__()
        self.log_path = log_path
        self.refresh_s = refresh_s
        self.init_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        if refresh_s > 0:
            self.timer.start(int(refresh_s * 1000))
        self.update_data()

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

    def update_data(self):
        log = load_log(self.log_path)
        if log is None or log.empty:
            log = synthetic_log()

        x = log["ts"] if "ts" in log.columns and log["ts"].notna().any() else log["t"]
        price = log["price"]
        pnl = log["pnl"] if "pnl" in log else None
        p_bad = log["p_bad"] if "p_bad" in log else None
        bad_flag = log["bad_flag"] if "bad_flag" in log else None
        action = log["action"] if "action" in log else None
        volume = log["volume"] if "volume" in log else None
        hold_roll = rolling_hold(log, window=200)

        self.price_curve.setData(x, price)
        if bad_flag is not None:
            # For stepMode=True, x must be len(y)+1; pad last point.
            x_bad = np.append(np.array(x), x.iloc[-1] if hasattr(x, "iloc") else x[-1] + 1)
            y_bad = np.append(np.array(bad_flag), bad_flag.iloc[-1] if hasattr(bad_flag, "iloc") else bad_flag[-1])
            self.bad_fill.setData(x_bad, y_bad)
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
            self.bad_flag_curve.setData(x, bad_flag)

        if volume is not None:
            self.vol_curve.setData(x, volume)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="logs/trading_log.csv", help="CSV log file path")
    ap.add_argument("--refresh", type=float, default=1.0, help="Refresh seconds; set 0 for one-shot")
    args = ap.parse_args()

    log_path = pathlib.Path(args.log)
    app = QtWidgets.QApplication([])
    dash = Dashboard(log_path=log_path, refresh_s=args.refresh)
    dash.show()
    if args.refresh <= 0:
        dash.update_data()
    QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    main()

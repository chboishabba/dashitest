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
  PYTHONPATH=. python trading/training_dashboard_pg.py --log logs/trading_log.csv --refresh 1.0
  # Progressive day-by-day reveal (each refresh adds the next day if ts present)
  PYTHONPATH=. python trading/training_dashboard_pg.py --log logs/trading_log.csv --refresh 1.0 --progressive-days
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
    ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(t, unit="h")
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
            "ts": ts,
        }
    )


def rolling_hold(log, window=100):
    if "hold" not in log:
        return None
    return log["hold"].rolling(window, min_periods=1).mean()


def coerce_plot_x(x_vals, length):
    """
    Convert timestamps or mixed-type x values into a numeric array for pyqtgraph.
    Falls back to a simple index if values are not numeric or parseable.
    """
    x_arr = np.array(x_vals)
    if np.issubdtype(x_arr.dtype, np.datetime64):
        return x_arr.astype("datetime64[ns]").astype("int64") / 1e9
    if x_arr.dtype == object:
        dt = pd.to_datetime(x_arr, errors="coerce")
        if dt.notna().any():
            return dt.astype("int64") / 1e9
    if not np.issubdtype(x_arr.dtype, np.number):
        return np.arange(length)
    return x_arr


def list_csv_logs(log_dir: pathlib.Path):
    if not log_dir.exists():
        return []
    return sorted(p for p in log_dir.glob("*.csv") if p.is_file())


def select_log_path(cli_log: str | None, logs_dir: pathlib.Path, choice_idx: int | None = None):
    """
    Resolve the log path. If cli_log is provided, use it. Otherwise, list CSVs under logs_dir
    and let the user pick (defaulting to the newest).
    """
    if cli_log:
        return pathlib.Path(cli_log)

    candidates = list_csv_logs(logs_dir)
    if not candidates:
        default_path = logs_dir / "trading_log.csv"
        print(f"No CSV logs found under {logs_dir}. Falling back to {default_path} (may be missing).")
        return default_path

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    if choice_idx is not None:
        if 1 <= choice_idx <= len(candidates):
            return candidates[choice_idx - 1]
        print(f"[log-index] {choice_idx} out of range 1..{len(candidates)}; falling back to prompt.")

    print("Select a log CSV (Enter = newest):")
    for idx, path in enumerate(candidates, 1):
        print(f"  [{idx}] {path.name}")

    try:
        choice = input(f"Choice [Enter for {newest.name}]: ").strip()
    except EOFError:
        choice = ""

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]

    return newest


def prepare_progressive_view(log, day_idx, day_keys, refresh_s, progressive_days):
    """
    Apply progressive-day filtering and return the filtered log, parsed timestamps,
    updated day_keys, and next day_idx.
    """
    ts_dt = None
    if "ts" in log.columns:
        ts_dt = pd.to_datetime(log["ts"], errors="coerce")

    if progressive_days and ts_dt is not None and ts_dt.notna().any():
        ts_days = ts_dt.dt.normalize()
        candidate_keys = np.sort(ts_days.dropna().unique())
        if day_keys is None or len(day_keys) != len(candidate_keys) or not np.array_equal(day_keys, candidate_keys):
            day_keys = candidate_keys
        if day_keys is not None and len(day_keys) > 0:
            day_idx = min(day_idx, len(day_keys) - 1)
            visible_days = set(day_keys[: day_idx + 1])
            log = log.loc[ts_days.isin(visible_days)]
            if refresh_s > 0 and day_idx < len(day_keys) - 1:
                day_idx += 1

    if ts_dt is not None:
        ts_dt = ts_dt.loc[log.index]

    return log, ts_dt, day_keys, day_idx


class Dashboard(QtWidgets.QMainWindow):
    def __init__(
        self,
        log_path: pathlib.Path,
        refresh_s: float,
        progressive_days: bool,
        debug: bool = False,
        debug_every: int = 1000,
        hist: bool = False,
        hist_window: int = 2000,
        hist_bins: int = 40,
        plane_scale: str = "linear",
        plane_norm_window: int = 0,
    ):
        super().__init__()
        self.log_path = log_path
        self.refresh_s = refresh_s
        self.progressive_days = progressive_days
        self.debug = debug
        self.debug_every = debug_every
        self.last_debug_rows = 0
        self.hist = hist
        self.hist_window = hist_window
        self.hist_bins = hist_bins
        self.plane_scale = plane_scale
        self.plane_norm_window = plane_norm_window
        self.day_idx = 0
        self.day_keys = None
        self.progressive_warned = False
        self.cached_log = pd.DataFrame()
        self.posture_spans = []
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

        # Posture pane (ACT/HOLD/BAN shading)
        self.p_eff = self.win.addPlot(row=4, col=0, title="Efficiency (cash_eff bps / exec_eff drag bps)")
        self.cash_eff_curve = self.p_eff.plot(pen=pg.mkPen("w", width=1))
        self.exec_eff_curve = self.p_eff.plot(pen=pg.mkPen("c", width=1))

        # Posture pane (ACT/HOLD/BAN shading)
        self.p_goal = self.win.addPlot(row=5, col=0, title="MDL/Stress/GoalProb")
        self.mdl_curve = self.p_goal.plot(pen=pg.mkPen("y", width=1))
        self.stress_curve = self.p_goal.plot(pen=pg.mkPen("r", width=1))
        self.goal_prob_curve = self.p_goal.plot(pen=pg.mkPen("g", width=1))

        # Plane rates pane
        plane_title = "Plane rates (0..3)"
        if self.plane_scale == "log10":
            plane_title = "Plane rates log10 (0..3)"
        self.p_planes = self.win.addPlot(row=6, col=0, title=plane_title)
        self.plane0_curve = self.p_planes.plot(pen=pg.mkPen("w", width=1))
        self.plane1_curve = self.p_planes.plot(pen=pg.mkPen((255, 180, 0), width=1))
        self.plane2_curve = self.p_planes.plot(pen=pg.mkPen((0, 200, 255), width=1))
        self.plane3_curve = self.p_planes.plot(pen=pg.mkPen((200, 80, 255), width=1))

        # Posture pane (ACT/HOLD/BAN shading)
        posture_row = 8 if self.hist else 7
        self.p_posture = self.win.addPlot(row=posture_row, col=0, title="Posture (BAN/HOLD/ACT)")
        self.p_posture.setYRange(-1.5, 1.5)
        self.p_posture.setMouseEnabled(y=False)
        self.p_posture.hideAxis("left")
        self.p_posture.showGrid(x=True, y=False, alpha=0.3)
        self.p_posture.setXLink(self.p_price)

        self.p_hist = None
        self.hist_cash = None
        self.hist_exec = None
        if self.hist:
            self.p_hist = self.win.addPlot(row=7, col=0, title="Histograms (cash_eff bps / exec drag bps)")
            self.hist_cash = pg.BarGraphItem(x=[], height=[], width=0.9, brush=pg.mkBrush(200, 200, 200, 120))
            self.hist_exec = pg.BarGraphItem(x=[], height=[], width=0.9, brush=pg.mkBrush(80, 200, 255, 120))
            self.p_hist.addItem(self.hist_cash)
            self.p_hist.addItem(self.hist_exec)

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
            if self.progressive_days and not self.progressive_warned:
                print("[progressive-days] Log missing/empty; using synthetic data with hourly timestamps.")
                self.progressive_warned = True

        ts_dt_full = pd.to_datetime(log["ts"], errors="coerce") if "ts" in log.columns else None
        allowed_mask = None
        if self.progressive_days and ts_dt_full is not None and ts_dt_full.notna().any():
            ts_days = ts_dt_full.dt.normalize()
            candidate_keys = np.sort(ts_days.dropna().unique())
            if self.day_keys is None or len(self.day_keys) != len(candidate_keys) or not np.array_equal(
                self.day_keys, candidate_keys
            ):
                self.day_keys = candidate_keys
            if self.day_keys is not None and len(self.day_keys) > 0:
                self.day_idx = min(self.day_idx, len(self.day_keys) - 1)
                if self.refresh_s > 0 and self.day_idx < len(self.day_keys) - 1:
                    self.day_idx += 1
                visible_days = set(self.day_keys[: self.day_idx + 1])
                allowed_mask = ts_days.isin(visible_days)
        elif self.progressive_days and not self.progressive_warned:
            print("[progressive-days] ts column missing or unparseable; showing full data.")
            self.progressive_warned = True

        allowed_log = log if allowed_mask is None else log.loc[allowed_mask]

        if self.cached_log.empty:
            self.cached_log = allowed_log.copy()
        elif len(allowed_log) < len(self.cached_log):
            self.cached_log = allowed_log.copy()
        else:
            new_rows = allowed_log.iloc[len(self.cached_log) :]
            if not new_rows.empty:
                self.cached_log = pd.concat([self.cached_log, new_rows])

        log = self.cached_log
        ts_dt = pd.to_datetime(log["ts"], errors="coerce") if "ts" in log.columns else None

        if ts_dt is not None and ts_dt.notna().any():
            x = ts_dt
        else:
            x = log["ts"] if "ts" in log.columns and log["ts"].notna().any() else log["t"]
        x_plot = coerce_plot_x(x, len(log))
        if len(x_plot) > 1:
            diffs = np.diff(x_plot)
            if np.any(diffs < 0):
                order = np.argsort(x_plot, kind="mergesort")
                log = log.iloc[order]
                x_plot = x_plot[order]
                if self.debug:
                    print("[debug] x not monotonic; sorted by x for plotting.")
        if len(x_plot) > 0:
            x_span = float(np.nanmax(x_plot) - np.nanmin(x_plot))
            x_mag = float(np.nanmax(np.abs(x_plot)))
            if x_mag > 1e6 and x_span < 1e6:
                x0 = float(x_plot[0])
                x_plot = x_plot - x0
                if self.debug:
                    print("[debug] x shifted to relative seconds; offset:", x0)
        should_debug = (
            self.debug
            and (self.last_debug_rows == 0 or len(log) - self.last_debug_rows >= self.debug_every)
        )
        if should_debug:
            x_name = "ts" if isinstance(x, pd.Series) and x.name == "ts" else "t"
            print(
                "[debug] rows:", len(log),
                "x_src:", x_name,
                "x_dtype:", getattr(np.array(x).dtype, "name", str(np.array(x).dtype)),
                "x_plot_dtype:", x_plot.dtype,
                "x_plot_minmax:", (float(np.nanmin(x_plot)), float(np.nanmax(x_plot))) if len(x_plot) else None,
            )
        price = log["price"]
        pnl = log["pnl"] if "pnl" in log else None
        p_bad = log["p_bad"] if "p_bad" in log else None
        bad_flag = log["bad_flag"] if "bad_flag" in log else None
        action = log["action"] if "action" in log else None
        volume = log["volume"] if "volume" in log else None
        cash_eff = log["cash_eff"] if "cash_eff" in log else None
        exec_eff = log["exec_eff"] if "exec_eff" in log else None
        mdl_rate = log["mdl_rate"] if "mdl_rate" in log else None
        stress = log["stress"] if "stress" in log else None
        goal_prob = log["goal_prob"] if "goal_prob" in log else None
        plane_rate0 = log["plane_rate0"] if "plane_rate0" in log else None
        plane_rate1 = log["plane_rate1"] if "plane_rate1" in log else None
        plane_rate2 = log["plane_rate2"] if "plane_rate2" in log else None
        plane_rate3 = log["plane_rate3"] if "plane_rate3" in log else None
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

        self.price_curve.setData(x_plot, price)
        if price is not None and len(price) > 0:
            pmin = float(np.nanmin(price))
            pmax = float(np.nanmax(price))
            if np.isfinite(pmin) and np.isfinite(pmax) and pmin != pmax:
                self.p_price.setYRange(pmin, pmax, padding=0.05)
            if should_debug:
                print("[debug] price_range:", (pmin, pmax))
        step_x = None
        if bad_flag is not None:
            x_arr = np.array(x_plot)
            y_arr = np.array(bad_flag)

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
            if price is not None and len(price) > 0:
                pmin = float(np.nanmin(price))
                pmax = float(np.nanmax(price))
                if np.isfinite(pmin) and np.isfinite(pmax) and pmin != pmax:
                    y_arr = pmin + y_arr * (pmax - pmin)
                    self.bad_fill.setData(step_x, y_arr, fillLevel=pmin)
                else:
                    self.bad_fill.setData(step_x, y_arr)
            else:
                self.bad_fill.setData(step_x, y_arr)
        if action is not None:
            buys = action == 1
            sells = action == -1
            self.buy_scatter.setData(x_plot[np.asarray(buys)], price[buys])
            self.sell_scatter.setData(x_plot[np.asarray(sells)], price[sells])

        if pnl is not None:
            self.pnl_curve.setData(x_plot, pnl)
            pnl_min = float(np.nanmin(pnl))
            pnl_max = float(np.nanmax(pnl))
            if np.isfinite(pnl_min) and np.isfinite(pnl_max) and pnl_min != pnl_max:
                self.p_pnl.setYRange(pnl_min, pnl_max, padding=0.05)
            if should_debug:
                print("[debug] pnl_range:", (pnl_min, pnl_max))
        if hold_roll is not None:
            self.hold_curve.setData(x_plot, hold_roll)

        if p_bad is not None:
            self.p_bad_curve.setData(x_plot, p_bad)
        if bad_flag is not None:
            # Reuse the step-aligned x for the bad_flag panel if available.
            if step_x is None:
                x_arr = np.array(x_plot)
                y_arr = np.array(bad_flag)
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
            self.vol_curve.setData(x_plot, volume)
            vmin = float(np.nanmin(volume))
            vmax = float(np.nanmax(volume))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                self.p_vol.setYRange(vmin, vmax, padding=0.05)
            if should_debug:
                print("[debug] volume_range:", (vmin, vmax))
        elif should_debug:
            print("[debug] volume column missing; volume pane empty.")

        if cash_eff is not None or exec_eff is not None:
            if cash_eff is not None:
                cash_eff_bps = np.array(cash_eff, dtype=float) * 1e4
                self.cash_eff_curve.setData(x_plot, cash_eff_bps)
            if exec_eff is not None:
                exec_eff_drag_bps = (1.0 - np.array(exec_eff, dtype=float)) * 1e4
                self.exec_eff_curve.setData(x_plot, exec_eff_drag_bps)
            eff_series = []
            if cash_eff is not None:
                eff_series.append(cash_eff_bps)
            if exec_eff is not None:
                eff_series.append(exec_eff_drag_bps)
            if eff_series:
                eff_all = pd.concat([pd.Series(s) for s in eff_series]).to_numpy()
                emin = float(np.nanmin(eff_all))
                emax = float(np.nanmax(eff_all))
                if np.isfinite(emin) and np.isfinite(emax) and emin != emax:
                    self.p_eff.setYRange(emin, emax, padding=0.05)
        elif should_debug:
            print("[debug] efficiency columns missing; efficiency pane empty.")

        if mdl_rate is not None or stress is not None or goal_prob is not None:
            if mdl_rate is not None:
                self.mdl_curve.setData(x_plot, mdl_rate)
            if stress is not None:
                self.stress_curve.setData(x_plot, stress)
            if goal_prob is not None:
                self.goal_prob_curve.setData(x_plot, goal_prob)
        elif should_debug:
            print("[debug] mdl/goal columns missing; goal pane empty.")

        if plane_rate0 is not None or plane_rate1 is not None or plane_rate2 is not None or plane_rate3 is not None:
            eps = 1e-9
            if self.plane_norm_window and self.plane_norm_window > 1:
                plane_series = []
                for series in (plane_rate0, plane_rate1, plane_rate2, plane_rate3):
                    if series is None:
                        continue
                    arr = np.array(series, dtype=float)
                    tail = arr[-self.plane_norm_window :]
                    tail = tail[np.isfinite(tail)]
                    if tail.size:
                        plane_series.append(float(np.nanmax(tail)))
                max_norm = max(plane_series) if plane_series else 0.0
                if max_norm > 0:
                    if plane_rate0 is not None:
                        plane_rate0 = np.array(plane_rate0, dtype=float) / max_norm
                    if plane_rate1 is not None:
                        plane_rate1 = np.array(plane_rate1, dtype=float) / max_norm
                    if plane_rate2 is not None:
                        plane_rate2 = np.array(plane_rate2, dtype=float) / max_norm
                    if plane_rate3 is not None:
                        plane_rate3 = np.array(plane_rate3, dtype=float) / max_norm
            if self.plane_scale == "log10":
                if plane_rate0 is not None:
                    plane_rate0 = np.log10(np.array(plane_rate0, dtype=float) + eps)
                if plane_rate1 is not None:
                    plane_rate1 = np.log10(np.array(plane_rate1, dtype=float) + eps)
                if plane_rate2 is not None:
                    plane_rate2 = np.log10(np.array(plane_rate2, dtype=float) + eps)
                if plane_rate3 is not None:
                    plane_rate3 = np.log10(np.array(plane_rate3, dtype=float) + eps)
            if plane_rate0 is not None:
                self.plane0_curve.setData(x_plot, plane_rate0)
            if plane_rate1 is not None:
                self.plane1_curve.setData(x_plot, plane_rate1)
            if plane_rate2 is not None:
                self.plane2_curve.setData(x_plot, plane_rate2)
            if plane_rate3 is not None:
                self.plane3_curve.setData(x_plot, plane_rate3)
            plane_series = [s for s in (plane_rate0, plane_rate1, plane_rate2, plane_rate3) if s is not None]
            if plane_series:
                plane_all = pd.concat([pd.Series(s) for s in plane_series]).to_numpy()
                pmin = float(np.nanmin(plane_all))
                pmax = float(np.nanmax(plane_all))
                if np.isfinite(pmin) and np.isfinite(pmax):
                    if pmin == pmax:
                        self.p_planes.setYRange(0.0, max(0.01, pmax), padding=0.1)
                    else:
                        self.p_planes.setYRange(pmin, pmax, padding=0.1)
        elif should_debug:
            print("[debug] plane_rate columns missing; plane pane empty.")

        if self.hist and self.p_hist is not None and (cash_eff is not None or exec_eff is not None):
            window = max(10, self.hist_window)
            if cash_eff is not None:
                cash_eff_bps = np.array(cash_eff, dtype=float) * 1e4
                cash_tail = cash_eff_bps[-window:]
                cash_tail = cash_tail[np.isfinite(cash_tail)]
                if cash_tail.size:
                    counts, edges = np.histogram(cash_tail, bins=self.hist_bins)
                    centers = (edges[:-1] + edges[1:]) / 2.0
                    self.hist_cash.setOpts(x=centers, height=counts, width=centers[1] - centers[0])
            if exec_eff is not None:
                exec_eff_drag_bps = (1.0 - np.array(exec_eff, dtype=float)) * 1e4
                exec_tail = exec_eff_drag_bps[-window:]
                exec_tail = exec_tail[np.isfinite(exec_tail)]
                if exec_tail.size:
                    counts, edges = np.histogram(exec_tail, bins=self.hist_bins)
                    centers = (edges[:-1] + edges[1:]) / 2.0
                    self.hist_exec.setOpts(x=centers, height=counts, width=centers[1] - centers[0])
        if should_debug:
            self.last_debug_rows = len(log)

        # Posture shading pane
        self._clear_posture_spans()
        if posture is not None:
            x_arr = np.array(x_plot)
            self._add_posture_spans(x_arr, posture)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=None, help="CSV log file path")
    ap.add_argument("--refresh", type=float, default=1.0, help="Refresh seconds; set 0 for one-shot")
    ap.add_argument(
        "--progressive-days",
        action="store_true",
        help="Progressively reveal data day-by-day on each refresh (requires ts column).",
    )
    ap.add_argument("--debug", action="store_true", help="Print debug info about plot data coercion.")
    ap.add_argument("--debug-every", type=int, default=1000, help="Print debug info every N rows.")
    ap.add_argument("--hist", action="store_true", help="Show rolling histograms for efficiency.")
    ap.add_argument("--hist-window", type=int, default=2000, help="Histogram window size.")
    ap.add_argument("--hist-bins", type=int, default=40, help="Histogram bin count.")
    ap.add_argument(
        "--plane-scale",
        type=str,
        default="linear",
        choices=["linear", "log10"],
        help="Plane-rate y scaling.",
    )
    ap.add_argument(
        "--plane-norm-window",
        type=int,
        default=0,
        help="Normalize plane rates by rolling max over this window (0 disables).",
    )
    ap.add_argument("--logs-dir", type=str, default="logs", help="Directory to search for CSV logs when --log not set.")
    ap.add_argument("--log-index", type=int, default=None, help="Select log by index from listed logs (1-based).")
    args = ap.parse_args()

    log_path = select_log_path(args.log, pathlib.Path(args.logs_dir), choice_idx=args.log_index)
    print(f"Using log: {log_path}")
    app = QtWidgets.QApplication([])
    dash = Dashboard(
        log_path=log_path,
        refresh_s=args.refresh,
        progressive_days=args.progressive_days,
        debug=args.debug,
        debug_every=args.debug_every,
        hist=args.hist,
        hist_window=args.hist_window,
        hist_bins=args.hist_bins,
        plane_scale=args.plane_scale,
        plane_norm_window=args.plane_norm_window,
    )
    dash.show()
    if args.refresh <= 0:
        dash.update_data()
    QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    main()

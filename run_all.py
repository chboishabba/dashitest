"""
run_all.py
----------
One-shot runner that starts the ternary trader logger and a live dashboard in the
same process. It will use cached Stooq data if present, otherwise a synthetic
price series. No extra setup needed beyond `python run_all.py`.
"""

import os
import threading
import time
import pathlib
import matplotlib

# Prefer an interactive backend; fall back if unavailable.
if matplotlib.get_backend().lower().startswith("agg"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt

import run_trader  # provides the logging loop
import training_dashboard as dash  # provides load_log, synthetic_log, draw


def main():
    log_path = pathlib.Path("logs/trading_log.csv")
    # Start trader in a background thread
    # max_steps=None -> run entire dataset. Adjust sleep_s for refresh speed.
    t = threading.Thread(target=run_trader.main, kwargs={"max_steps": None, "sleep_s": 0.01}, daemon=True)
    t.start()

    plt.ion()
    refresh = 0.5
    try:
        while t.is_alive():
            log = dash.load_log(log_path)
            if log is None:
                log = dash.synthetic_log()
            dash.draw(log)
            time.sleep(refresh)
        # one final draw after trader exits
        log = dash.load_log(log_path)
        if log is None:
            log = dash.synthetic_log()
        dash.draw(log)
        plt.show(block=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

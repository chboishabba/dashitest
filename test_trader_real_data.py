"""
Simple sanity check that run_trader uses real Stooq data when available.
Run with: python test_trader_real_data.py
Skips if no valid Stooq CSVs are present.
"""

import pathlib
import pandas as pd

import run_trader


def main():
    # Prefer BTC intraday if present, otherwise any Stooq CSV.
    try:
        csv_path = run_trader.find_btc_csv() or run_trader.find_stooq_csv()
    except Exception as e:
        print(f"SKIP: {e}")
        return

    # Clean log and run a short pass on real data
    log_path = pathlib.Path("logs/trading_log.csv")
    log_path.unlink(missing_ok=True)
    run_trader.main(max_steps=10, sleep_s=0)

    # Verify log exists and used real data (not synthetic)
    if not log_path.exists():
        raise RuntimeError("Log file was not created.")
    df = pd.read_csv(log_path)
    if df.empty:
        raise RuntimeError("Log file is empty.")
    if "source" not in df.columns:
        raise RuntimeError("Log missing 'source' column.")
    if (df["source"] == "synthetic").any():
        raise RuntimeError("Trader fell back to synthetic despite cached data.")

    print("PASS: run_trader used cached real data and produced a non-empty log.")


if __name__ == "__main__":
    main()

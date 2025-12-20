# data_downloader.py
import time
import random
import pathlib
import os
import requests
import socket
import pandas as pd
import numpy as np

# Force yfinance to use a writable cache under our data tree (or disable it)
# before import to avoid readonly sqlite issues on some systems.
_yf_cache_dir = pathlib.Path("data/cache/yfinance")
_yf_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YF_CACHE_DIR", str(_yf_cache_dir.resolve()))
os.environ.setdefault("YFINANCE_CACHE_DIR", str(_yf_cache_dir.resolve()))
os.environ.setdefault("YF_NO_CACHE", "1")

# Optional yfinance dependency
try:
    import yfinance as yf
except ImportError:
    yf = None

# Per-source polite rate limits
MIN_DELAY_STOOQ = 3.0  # seconds between Stooq requests
MIN_DELAY_YAHOO = 3.0  # seconds between Yahoo requests

_last_request_stooq = 0.0
_last_request_yahoo = 0.0


def polite_sleep(source: str):
    """Sleep to enforce per-source minimum delay."""
    global _last_request_stooq, _last_request_yahoo
    now = time.time()
    if source == "stooq":
        elapsed = now - _last_request_stooq
        if elapsed < MIN_DELAY_STOOQ:
            time.sleep(MIN_DELAY_STOOQ - elapsed + random.uniform(0.1, 0.3))
        _last_request_stooq = time.time()
    else:
        elapsed = now - _last_request_yahoo
        if elapsed < MIN_DELAY_YAHOO:
            time.sleep(MIN_DELAY_YAHOO - elapsed + random.uniform(0.1, 0.3))
        _last_request_yahoo = time.time()


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def download_stooq(symbol: str, out_dir="data/raw/stooq", overwrite=False):
    """
    Download full daily history from Stooq.
    """
    ensure_dir(out_dir)
    out_path = pathlib.Path(out_dir) / f"{symbol}.csv"

    if out_path.exists() and not overwrite:
        print(f"[stooq] cache hit: {symbol}")
        return out_path

    url = f"https://stooq.pl/q/d/l/?s={symbol}&i=d"
    print(f"[stooq] downloading {symbol}")
    polite_sleep("stooq")

    # retry with simple backoff and DNS guard
    base = 0.5
    for i in range(5):
        try:
            # quick DNS resolve to fail fast if offline
            socket.gethostbyname("stooq.pl")
            r = requests.get(url, timeout=30)
            # basic content check: ensure CSV-like response
            if r.status_code != 200 or b"<html" in r.content[:200].lower():
                raise RuntimeError(f"Bad response for {symbol}: status={r.status_code}")
            out_path.write_bytes(r.content)
            return out_path
        except Exception:
            if i == 4:
                raise
            time.sleep(min(base * (2 ** i) + random.uniform(0, 0.25), 10.0))
    return out_path


def download_btc_coingecko(out_path="data/raw/stooq/btc.us.csv", overwrite=False, min_rows=1000):
    """
    Download daily BTC/USD from CoinGecko (no API key) and save as CSV with
    columns resembling Stooq: Date,Open,High,Low,Close,Volume.
    """
    out_path = pathlib.Path(out_path)
    ensure_dir(out_path.parent)
    if out_path.exists() and not overwrite:
        try:
            existing = sum(1 for _ in open(out_path))
            if existing >= min_rows:
                print(f"[coingecko] cache hit: {out_path.name} ({existing} rows)")
                return out_path
            else:
                print(f"[coingecko] existing {out_path.name} too short ({existing} rows); refreshing")
        except Exception:
            print(f"[coingecko] refreshing {out_path.name} due to read error")

    polite_sleep("stooq")
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "max"}
    base = 0.5
    for i in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"Bad status {r.status_code}")
            data = r.json()
            prices = data.get("prices", [])
            vols = data.get("total_volumes", [])
            if not prices:
                raise RuntimeError("No prices from CoinGecko")
            # align volumes by timestamp if possible
            ts_to_vol = {int(v[0]): v[1] for v in vols}
            rows = []
            for ts_ms, px in prices:
                dt = pd.to_datetime(ts_ms, unit="ms").date().isoformat()
                vol = ts_to_vol.get(int(ts_ms), np.nan)
                rows.append(
                    {
                        "Date": dt,
                        "Open": px,
                        "High": px,
                        "Low": px,
                        "Close": px,
                        "Volume": vol,
                    }
                )
            df = pd.DataFrame(rows).sort_values("Date")
            df.to_csv(out_path, index=False)
            print(f"[coingecko] saved BTC to {out_path} ({len(df)} rows)")
            return out_path
        except Exception as e:
            if i == 4:
                raise
            time.sleep(min(base * (2 ** i) + random.uniform(0, 0.25), 10.0))
    return out_path


def download_btc_yahoo(out_path="data/raw/stooq/btc_yf.csv", overwrite=False, min_rows=1000):
    """
    Download daily BTC-USD via yfinance (no API key). Requires yfinance installed.
    """
    out_path = pathlib.Path(out_path)
    ensure_dir(out_path.parent)
    if yf is None:
        print("[yahoo] yfinance not installed; skipping BTC-USD download.")
        return None

    if out_path.exists() and not overwrite:
        try:
            existing = sum(1 for _ in open(out_path))
            if existing >= min_rows:
                print(f"[yahoo] cache hit: {out_path.name} ({existing} rows)")
                return out_path
            else:
                print(f"[yahoo] existing {out_path.name} too short ({existing} rows); refreshing")
        except Exception:
            print(f"[yahoo] refreshing {out_path.name} due to read error")

    polite_sleep("yahoo")
    base = 0.5
    df = None
    for i in range(5):
        try:
            df = yf.download(
                "BTC-USD",
                interval="1d",
                period="max",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception:
            time.sleep(min(base * (2 ** i) + random.uniform(0, 0.25), 10.0))

    if df is None or df.empty:
        print("[yahoo] BTC-USD download failed or empty.")
        return None

    df.to_csv(out_path)
    print(f"[yahoo] saved BTC-USD to {out_path} ({len(df)} rows)")
    return out_path


def download_btc_yahoo_intraday(
    out_path="data/raw/stooq/btc_intraday.csv",
    interval="1m",
    period="7d",
    overwrite=False,
    max_bytes=10 * 1024 * 1024,  # 10 MB cap
    min_rows=1000,
):
    """
    Download intraday BTC-USD via yfinance (1m up to 7d). Enforces a max file size.
    """
    out_path = pathlib.Path(out_path)
    ensure_dir(out_path.parent)
    if yf is None:
        print("[yahoo] yfinance not installed; skipping BTC-USD intraday download.")
        return None

    if out_path.exists() and not overwrite:
        try:
            existing_rows = sum(1 for _ in open(out_path))
            size = out_path.stat().st_size
            if existing_rows >= min_rows and size <= max_bytes:
                print(f"[yahoo] cache hit: {out_path.name} ({existing_rows} rows, {size} bytes)")
                return out_path
            else:
                print(f"[yahoo] refreshing {out_path.name} (rows={existing_rows}, bytes={size})")
        except Exception:
            print(f"[yahoo] refreshing {out_path.name} due to read error")

    # ensure yfinance cache dir exists (even if cache disabled)
    _yf_cache_dir.mkdir(parents=True, exist_ok=True)

    polite_sleep("yahoo")
    base = 0.5
    df = None
    for i in range(5):
        try:
            df = yf.download(
                "BTC-USD",
                interval=interval,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception:
            time.sleep(min(base * (2 ** i) + random.uniform(0, 0.25), 10.0))

    if df is None or df.empty:
        print("[yahoo] BTC-USD intraday download failed or empty.")
        return None

    df.to_csv(out_path)
    size = out_path.stat().st_size
    if size > max_bytes:
        # Trim to fit the size cap by keeping the most recent rows.
        ratio = max_bytes / size
        keep = max(min_rows, int(len(df) * ratio))
        if keep < len(df):
            df.tail(keep).to_csv(out_path)
            size = out_path.stat().st_size
            print(f"[yahoo] trimmed intraday data to {keep} rows to fit {max_bytes} bytes (now {size} bytes)")
        else:
            print(f"[yahoo] intraday file {size} bytes exceeds cap {max_bytes}; delete or reduce period.")
            return None
    print(f"[yahoo] saved BTC-USD intraday to {out_path} ({len(df)} rows, {size} bytes)")
    return out_path


def download_yahoo(
    symbol: str,
    interval="1h",
    period="max",
    out_dir="data/raw/yahoo",
    overwrite=False,
):
    """
    interval: 1d, 1h, 15m, 5m, 1m
    """
    ensure_dir(out_dir)
    safe_sym = symbol.replace("-", "_")
    out_path = pathlib.Path(out_dir) / f"{safe_sym}_{interval}.csv"

    if out_path.exists() and not overwrite:
        print(f"[yahoo] cache hit: {symbol} {interval}")
        return out_path

    if yf is None:
        raise RuntimeError("yfinance is not installed; cannot download Yahoo data.")

    print(f"[yahoo] downloading {symbol} {interval}")
    polite_sleep("yahoo")

    base = 0.5
    df = None
    for i in range(5):
        try:
            df = yf.download(
                symbol,
                interval=interval,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception:
            time.sleep(min(base * (2 ** i) + random.uniform(0, 0.25), 10.0))

    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol}")

    df.to_csv(out_path)
    return out_path


def csv_to_parquet(csv_path, out_dir="data/parquet"):
    ensure_dir(out_dir)
    csv_path = pathlib.Path(csv_path)
    df = pd.read_csv(csv_path)

    out_path = pathlib.Path(out_dir) / (csv_path.stem + ".parquet")
    df.to_parquet(out_path, index=False)
    return out_path


def bulk_stooq(symbols, **kwargs):
    paths = []
    for s in symbols:
        try:
            p = download_stooq(s, **kwargs)
            paths.append(p)
        except Exception as e:
            print(f"[stooq] failed {s}: {e}")
    return paths


def bulk_yahoo(symbols, **kwargs):
    paths = []
    for s in symbols:
        try:
            p = download_yahoo(s, **kwargs)
            paths.append(p)
        except Exception as e:
            print(f"[yahoo] failed {s}: {e}")
    return paths


if __name__ == "__main__":
    # Simple CLI demo: download a few symbols from Stooq (including btc) and Yahoo
    symbols = ["spy.us", "msft.us", "aapl.us", "btc.us"]
    print("Downloading sample symbols from Stooq...")
    try:
        bulk_stooq(symbols)
    except Exception as e:
        print(f"Stooq download failed (possibly offline): {e}")
    print("Downloading BTC from CoinGecko...")
    try:
        download_btc_coingecko()
    except Exception as e:
        print(f"CoinGecko BTC download failed: {e}")
    if yf is not None:
        print("Downloading sample symbols from Yahoo (1d, 1y)...")
        bulk_yahoo(["SPY", "MSFT", "AAPL", "BTC-USD"], interval="1d", period="1y")
        print("Downloading BTC-USD daily (full history) via yfinance...")
        download_btc_yahoo()
        print("Downloading BTC-USD intraday (1m, ~7d) via yfinance with size cap...")
        download_btc_yahoo_intraday()
    else:
        print("yfinance not installed; skipping Yahoo download.")

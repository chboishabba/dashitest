# data_downloader.py
import time
import random
import pathlib
import os
import json
try:
    import requests
except ImportError:
    requests = None
import socket
import pandas as pd
import numpy as np
import urllib.parse
import urllib.request

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

BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = f"{BINANCE_BASE}/api/v3/klines"
BINANCE_AGG_TRADES = f"{BINANCE_BASE}/api/v3/aggTrades"
BINANCE_LIMIT = 1000  # max rows per request
BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_PREMIUM = f"{BINANCE_FAPI}/fapi/v1/premiumIndex"
BINANCE_OPEN_INTEREST = f"{BINANCE_FAPI}/futures/data/openInterestHist"
DERIBIT_BASE = "https://www.deribit.com"
DERIBIT_INSTRUMENTS = f"{DERIBIT_BASE}/api/v2/public/get_instruments"
DERIBIT_SUMMARY = f"{DERIBIT_BASE}/api/v2/public/get_book_summary_by_currency"


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


def _interval_to_ms(interval: str) -> int:
    lookup = {
        "1s": 1_000,
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
    }
    if interval not in lookup:
        raise ValueError(f"Unsupported Binance interval: {interval}")
    return lookup[interval]


def _binance_get(url, params):
    """
    Simple retry wrapper for Binance public REST calls.
    """
    base = 0.5
    for i in range(5):
        try:
            payload = _get_json(url, params=params)
            if payload is None:
                raise RuntimeError("Binance request returned no data")
            return payload
        except Exception:
            if i == 4:
                raise
            time.sleep(min(base * (2 ** i) + random.uniform(0, 0.25), 10.0))
    return []


def _http_get(url, params=None, timeout=30):
    if params:
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"
    else:
        full_url = url
    if requests is not None:
        resp = requests.get(url, params=params, timeout=timeout)
        return resp.status_code, resp.text, resp.content
    with urllib.request.urlopen(full_url, timeout=timeout) as resp:
        content = resp.read()
        text = content.decode("utf-8", errors="replace")
        return resp.status, text, content


def _get_json(url, params=None):
    base = 0.5
    for i in range(5):
        try:
            status, text, content = _http_get(url, params=params, timeout=30)
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {text[:200]}")
            return json.loads(text)
        except Exception:
            if i == 4:
                raise
            time.sleep(min(base * (2 ** i) + random.uniform(0, 0.25), 10.0))
    return None


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
            status, text, content = _http_get(url, timeout=30)
            # basic content check: ensure CSV-like response
            if status != 200 or b"<html" in content[:200].lower():
                raise RuntimeError(f"Bad response for {symbol}: status={status}")
            out_path.write_bytes(content)
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


def download_btc_binance_intraday(
    out_path="data/raw/stooq/btc_intraday.csv",
    interval="1m",
    lookback_days=70,
    overwrite=False,
    max_rows=250_000,
):
    """
    Download extended intraday BTCUSDT klines from Binance (no API key).
    Defaults to ~70 days of 1m bars (roughly 10x the old 7d window).
    """
    out_path = pathlib.Path(out_path)
    ensure_dir(out_path.parent)

    interval_ms = _interval_to_ms(interval)
    target_rows = int((lookback_days * 86_400_000) / interval_ms)
    if out_path.exists() and not overwrite:
        try:
            existing_rows = sum(1 for _ in open(out_path))
            if existing_rows >= 0.9 * target_rows:
                print(f"[binance] cache hit: {out_path.name} ({existing_rows} rows)")
                return out_path
            else:
                print(f"[binance] refreshing {out_path.name} (rows={existing_rows}, target≈{target_rows})")
        except Exception:
            print(f"[binance] refreshing {out_path.name} due to read error")

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(lookback_days * 86_400_000)
    rows = []
    cursor = start_ms
    requests_made = 0
    while cursor < end_ms and len(rows) < max_rows:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": BINANCE_LIMIT,
        }
        data = _binance_get(BINANCE_KLINES, params)
        requests_made += 1
        if not data:
            break
        for k in data:
            rows.append(
                {
                    "Datetime": pd.to_datetime(k[0], unit="ms", utc=True),
                    "Open": float(k[1]),
                    "High": float(k[2]),
                    "Low": float(k[3]),
                    "Close": float(k[4]),
                    "Volume": float(k[5]),
                }
            )
        cursor = data[-1][6] + 1  # advance just past the last close time
        if len(data) < BINANCE_LIMIT:
            break
        time.sleep(0.25)
        if len(rows) >= max_rows:
            print(f"[binance] reached max_rows={max_rows}; data may be truncated.")
            break

    if not rows:
        print("[binance] BTCUSDT intraday download failed or empty.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(
        f"[binance] saved BTCUSDT intraday to {out_path} "
        f"({len(df)} rows, interval={interval}, lookback_days={lookback_days}, requests={requests_made})"
    )
    return out_path


def download_btc_binance_seconds(
    out_path="data/raw/stooq/btc_intraday_1s.csv",
    hours=10,
    overwrite=False,
    max_trades=1_000_000,
    sleep_s=0.25,
):
    """
    Download aggregated BTCUSDT trades from Binance and resample to 1-second bars.
    This is heavier than klines; default is ~10 hours of per-second bars.
    """
    out_path = pathlib.Path(out_path)
    ensure_dir(out_path.parent)

    target_rows = hours * 3600
    if out_path.exists() and not overwrite:
        try:
            existing_rows = sum(1 for _ in open(out_path))
            if existing_rows >= 0.8 * target_rows:
                print(f"[binance] cache hit: {out_path.name} ({existing_rows} rows)")
                return out_path
            else:
                print(f"[binance] refreshing {out_path.name} (rows={existing_rows}, target≈{target_rows})")
        except Exception:
            print(f"[binance] refreshing {out_path.name} due to read error")

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(hours * 3_600_000)
    trades = []
    cursor = start_ms
    requests_made = 0

    while cursor < end_ms and len(trades) < max_trades:
        params = {
            "symbol": "BTCUSDT",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": BINANCE_LIMIT,
        }
        batch = _binance_get(BINANCE_AGG_TRADES, params)
        requests_made += 1
        if not batch:
            break
        trades.extend(batch)
        cursor = batch[-1]["T"] + 1  # move just past the last trade timestamp
        if len(batch) < BINANCE_LIMIT:
            break
        time.sleep(sleep_s)
        if len(trades) >= max_trades:
            print(f"[binance] reached max_trades={max_trades}; data may be truncated.")
            break

    if not trades:
        print("[binance] BTCUSDT 1s download failed or empty.")
        return None

    df_trades = pd.DataFrame(trades)
    df_trades["timestamp"] = pd.to_datetime(df_trades["T"], unit="ms", utc=True)
    df_trades["price"] = pd.to_numeric(df_trades["p"], errors="coerce")
    df_trades["qty"] = pd.to_numeric(df_trades["q"], errors="coerce")
    df_trades = df_trades.sort_values("timestamp")

    bars = (
        df_trades.resample("1S", on="timestamp")
        .agg(
            Open=("price", "first"),
            High=("price", "max"),
            Low=("price", "min"),
            Close=("price", "last"),
            Volume=("qty", "sum"),
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
        .reset_index()
        .rename(columns={"timestamp": "Datetime"})
    )

    if bars.empty:
        print("[binance] No per-second bars assembled from trades.")
        return None

    bars.to_csv(out_path, index=False)
    print(
        f"[binance] saved BTCUSDT 1s bars to {out_path} "
        f"({len(bars)} rows, trades={len(trades)}, requests={requests_made})"
    )
    return out_path


def download_binance_premium(symbol="BTCUSDT", out_path="data/market_meta/binance_premium_BTCUSDT.json", overwrite=False):
    ensure_dir(pathlib.Path(out_path).parent)
    out_path = pathlib.Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"[binance] premium cache hit: {out_path.name}")
        return out_path
    print(f"[binance] downloading premium index for {symbol}")
    payload = _get_json(BINANCE_PREMIUM, params={"symbol": symbol})
    if payload is None:
        raise RuntimeError("Binance premium download failed.")
    out_path.write_text(json.dumps(payload))
    return out_path


def download_binance_open_interest(
    symbol="BTCUSDT",
    period="5m",
    limit=500,
    out_path="data/market_meta/binance_open_interest_BTCUSDT_5m.json",
    overwrite=False,
):
    ensure_dir(pathlib.Path(out_path).parent)
    out_path = pathlib.Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"[binance] open interest cache hit: {out_path.name}")
        return out_path
    print(f"[binance] downloading open interest for {symbol} period={period}")
    payload = _get_json(
        BINANCE_OPEN_INTEREST,
        params={"symbol": symbol, "period": period, "limit": int(limit)},
    )
    if payload is None:
        raise RuntimeError("Binance open interest download failed.")
    out_path.write_text(json.dumps(payload))
    return out_path


def download_deribit_options_instruments(
    currency="BTC",
    out_path="data/market_meta/deribit_options_instruments_BTC.json",
    overwrite=False,
):
    ensure_dir(pathlib.Path(out_path).parent)
    out_path = pathlib.Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"[deribit] instruments cache hit: {out_path.name}")
        return out_path
    print(f"[deribit] downloading instruments for {currency}")
    payload = _get_json(
        DERIBIT_INSTRUMENTS,
        params={"currency": currency, "kind": "option", "expired": "false"},
    )
    if payload is None:
        raise RuntimeError("Deribit instruments download failed.")
    out_path.write_text(json.dumps(payload))
    return out_path


def download_deribit_options_summary(
    currency="BTC",
    out_path="data/market_meta/deribit_options_summary_BTC.json",
    overwrite=False,
):
    ensure_dir(pathlib.Path(out_path).parent)
    out_path = pathlib.Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"[deribit] summary cache hit: {out_path.name}")
        return out_path
    print(f"[deribit] downloading summary for {currency}")
    payload = _get_json(
        DERIBIT_SUMMARY,
        params={"currency": currency, "kind": "option"},
    )
    if payload is None:
        raise RuntimeError("Deribit summary download failed.")
    out_path.write_text(json.dumps(payload))
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
    print("Downloading BTC-USD intraday (1m, ~70d) via Binance...")
    try:
        download_btc_binance_intraday()
    except Exception as e:
        print(f"Binance intraday download failed: {e}")
    print("Downloading BTC-USD per-second (~10h) via Binance aggregated trades...")
    try:
        download_btc_binance_seconds()
    except Exception as e:
        print(f"Binance per-second download failed: {e}")
    if yf is not None:
        print("Downloading sample symbols from Yahoo (1d, 1y)...")
        bulk_yahoo(["SPY", "MSFT", "AAPL", "BTC-USD"], interval="1d", period="1y")
        print("Downloading BTC-USD daily (full history) via yfinance...")
        download_btc_yahoo()
        print("Downloading BTC-USD intraday (1m, ~7d) via yfinance with size cap...")
        download_btc_yahoo_intraday()
    else:
        print("yfinance not installed; skipping Yahoo download.")

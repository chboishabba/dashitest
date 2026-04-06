import pathlib

import numpy as np
import pandas as pd


def find_btc_csv():
    raw = pathlib.Path("data/raw/stooq")
    if not raw.exists():
        return None
    # prefer intraday then daily
    files = (
        sorted(raw.glob("btc_intraday_1s*.csv"))
        + sorted(raw.glob("btc_intraday*.csv"))
        + sorted(raw.glob("btc*.csv"))
    )
    for f in files:
        try:
            df_head = pd.read_csv(f, nrows=5)
            cols = [c.lower() for c in df_head.columns]
            if any(k in cols for k in ("close", "zamkniecie")) and not df_head.empty:
                df_full = pd.read_csv(f)
                if len(df_full) >= 1000:
                    return f
        except Exception:
            continue
    return None


def find_stooq_csv():
    raw = pathlib.Path("data/raw/stooq")
    if not raw.exists():
        raise FileNotFoundError("data/raw/stooq not found; run trading/data_downloader.py first.")
    files = sorted(raw.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No Stooq CSVs found; run trading/data_downloader.py.")
    # pick first valid CSV with Close column and data
    for f in files:
        try:
            df_head = pd.read_csv(f, nrows=5)
            cols = [c.lower() for c in df_head.columns]
            if any(k in cols for k in ("close", "zamkniecie")) and not df_head.empty:
                return f
        except Exception:
            continue
    raise FileNotFoundError(
        "No valid Stooq CSV with Close/Zamkniecie column; re-run trading/data_downloader.py with .us symbols."
    )


def list_price_csvs(raw_root: pathlib.Path) -> list[pathlib.Path]:
    if not raw_root.exists():
        return []
    return sorted(p for p in raw_root.rglob("*.csv") if p.is_file())


def load_price_frame(path: pathlib.Path) -> pd.DataFrame:
    def read_basic(p):
        return pd.read_csv(p)

    def read_skip(p):
        return pd.read_csv(p, skiprows=2)

    def pick(col_map, *keys):
        for key in keys:
            if key in col_map:
                return col_map[key]
        return None

    def parse_df(df):
        col_map = {c.lower(): c for c in df.columns}
        close_key = pick(col_map, "close", "zamkniecie")
        if close_key is None:
            raise ValueError(f"Close/Zamkniecie column not found in {path}")

        date_key = pick(col_map, "date", "data", "datetime")
        if date_key is None and "price" in col_map:
            maybe_dates = pd.to_datetime(df[col_map["price"]], errors="coerce")
            if maybe_dates.notna().mean() > 0.8:
                date_key = col_map["price"]

        out = pd.DataFrame()
        out["close"] = pd.to_numeric(df[close_key], errors="coerce")

        open_key = pick(col_map, "open", "otwarcie")
        high_key = pick(col_map, "high", "max", "najwyzszy")
        low_key = pick(col_map, "low", "min", "najnizszy")
        volume_key = pick(col_map, "volume", "wolumen")
        bid_key = pick(col_map, "bid", "best_bid")
        ask_key = pick(col_map, "ask", "best_ask")
        spread_key = pick(col_map, "spread")

        out["open"] = pd.to_numeric(df[open_key], errors="coerce") if open_key else out["close"]
        out["high"] = pd.to_numeric(df[high_key], errors="coerce") if high_key else out["close"]
        out["low"] = pd.to_numeric(df[low_key], errors="coerce") if low_key else out["close"]
        if volume_key:
            out["volume"] = pd.to_numeric(df[volume_key], errors="coerce")
        else:
            out["volume"] = pd.Series(np.ones(len(out)) * 1e6)
        out["bid"] = pd.to_numeric(df[bid_key], errors="coerce") if bid_key else np.nan
        out["ask"] = pd.to_numeric(df[ask_key], errors="coerce") if ask_key else np.nan
        out["spread"] = pd.to_numeric(df[spread_key], errors="coerce") if spread_key else np.nan

        if date_key is not None:
            out["timestamp"] = pd.to_datetime(df[date_key], errors="coerce")
            valid = out["timestamp"].notna() & out["close"].notna()
            out = out.loc[valid].copy()
            out = out.sort_values("timestamp").reset_index(drop=True)
        else:
            out = out.loc[out["close"].notna()].reset_index(drop=True)
            out["timestamp"] = pd.NaT

        pos_vol = out["volume"][np.isfinite(out["volume"]) & (out["volume"] > 0)]
        fallback_vol = np.median(pos_vol) if pos_vol.size else 1e6
        out["volume"] = np.where((~np.isfinite(out["volume"])) | (out["volume"] <= 0), fallback_vol, out["volume"])

        return out

    for reader in (read_basic, read_skip):
        try:
            df = reader(path)
            cols_lower = [c.lower() for c in df.columns]
            if "price" in cols_lower:
                price_col = cols_lower.index("price")
                head_vals = df.iloc[:2, price_col].astype(str).str.lower()
                if head_vals.str.contains("ticker").any():
                    df = df.iloc[2:].reset_index(drop=True)
            out = parse_df(df)
            if len(out) >= 10 and np.isfinite(out["close"]).any():
                return out
        except Exception:
            continue
    raise ValueError(f"Could not parse prices from {path}")


def load_prices(path: pathlib.Path, return_time: bool = False):
    frame = load_price_frame(path)
    close = frame["close"].to_numpy()
    vol = frame["volume"].to_numpy()
    ts = frame["timestamp"].to_numpy()
    if return_time:
        return close, vol, ts
    return close, vol

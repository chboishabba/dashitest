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


def load_prices(path: pathlib.Path, return_time: bool = False):
    def read_basic(p):
        return pd.read_csv(p)

    def read_skip(p):
        # for yfinance multi-index CSV: skip ticker header rows
        return pd.read_csv(p, skiprows=2)

    def parse_df(df):
        col_map = {c.lower(): c for c in df.columns}
        close_key = None
        for key in ("close", "zamkniecie"):
            if key in col_map:
                close_key = col_map[key]
                break
        if close_key is None:
            raise ValueError(f"Close/Zamkniecie column not found in {path}")

        date_key = None
        for dk in ("date", "data", "datetime"):
            if dk in col_map:
                date_key = col_map[dk]
                break
        # yfinance exports sometimes put dates under "Price" after skipping header rows
        if date_key is None and "price" in col_map:
            maybe_dates = pd.to_datetime(df[col_map["price"]], errors="coerce")
            if maybe_dates.notna().mean() > 0.8:
                date_key = col_map["price"]

        close_series = pd.to_numeric(df[close_key], errors="coerce")

        vol_key = None
        for vk in ("volume", "wolumen"):
            if vk in col_map:
                vol_key = col_map[vk]
                break
        if vol_key is not None:
            vol_series = pd.to_numeric(df[vol_key], errors="coerce")
        else:
            vol_series = pd.Series(np.ones(len(close_series)) * 1e6)

        if date_key is not None:
            dates = pd.to_datetime(df[date_key], errors="coerce")
            valid = (~close_series.isna()) & (~dates.isna())
            close_series = close_series[valid]
            vol_series = vol_series[valid]
            dates = dates[valid]
            order = np.argsort(dates.to_numpy())
            close_series = close_series.iloc[order]
            vol_series = vol_series.iloc[order]
        else:
            close_series = close_series.dropna()
            vol_series = vol_series.loc[close_series.index]

        time_index = dates.to_numpy() if date_key is not None else None
        return close_series.to_numpy(), vol_series.to_numpy(), time_index

    # try basic read, then fallback to skiprows if too many non-numeric
    for reader in (read_basic, read_skip):
        try:
            df = reader(path)
            # handle yfinance multi-index export (ticker row, then date row)
            cols_lower = [c.lower() for c in df.columns]
            if "price" in cols_lower:
                price_col = cols_lower.index("price")
                head_vals = df.iloc[:2, price_col].astype(str).str.lower()
                if head_vals.str.contains("ticker").any():
                    df = df.iloc[2:].reset_index(drop=True)
            close, vol, ts = parse_df(df)
            # replace nonpositive/NaN volume
            pos_vol = vol[np.isfinite(vol) & (vol > 0)]
            fallback_vol = np.median(pos_vol) if pos_vol.size else 1e6
            vol = np.where((~np.isfinite(vol)) | (vol <= 0), fallback_vol, vol)
            # require reasonable length
            if len(close) >= 10 and np.isfinite(close).any():
                if return_time:
                    return close, vol, ts
                return close, vol
        except Exception:
            continue
    raise ValueError(f"Could not parse prices from {path}")

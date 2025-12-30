"""
Contextual news/stress fetchers for regime windows.

Provides:
- Reuters market wraps via RSS (no key).
- Macro calendar (TradingEconomics) if TE_API_KEY is set; empty otherwise.
- Stress proxies via yfinance (VIX, USDCNH, Copper) if available; empty on import failure.
- A unified `contextual_events(date, keywords=None)` helper returning a dict of signals.

Usage (example):
  from trading.scripts.contextual_news import contextual_events
  ctx = contextual_events("2015-08-24", keywords=["china", "greece", "yuan"])
  print(ctx)
"""

import os
import datetime as dt
from typing import List, Dict, Any

import requests

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    import yfinance as yf
except ImportError:
    yf = None


def parse_date(date_str: str) -> dt.date:
    return dt.datetime.fromisoformat(date_str).date()


def fetch_reuters_wrap(keywords: List[str], feed_url: str = "https://www.reuters.com/markets/rss") -> List[Dict[str, Any]]:
    """Fetch Reuters markets RSS and return entries matching keywords (case-insensitive)."""
    if feedparser is None:
        return []
    feed = feedparser.parse(feed_url)
    out = []
    for e in feed.entries:
        text = (e.title or "") + " " + (e.summary or "")
        if any(k.lower() in text.lower() for k in keywords):
            out.append(
                {
                    "published": e.get("published"),
                    "title": e.get("title"),
                    "link": e.get("link"),
                }
            )
    return out


def fetch_tradingeconomics_calendar(d1: str, d2: str) -> List[Dict[str, Any]]:
    """Fetch macro calendar from TradingEconomics if TE_API_KEY is set; otherwise return []."""
    api_key = os.environ.get("TE_API_KEY")
    if not api_key:
        return []
    url = "https://api.tradingeconomics.com/calendar"
    params = {"c": api_key, "d1": d1, "d2": d2}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def fetch_yf_close(ticker: str, date: dt.date) -> float:
    if yf is None:
        return float("nan")
    end = date + dt.timedelta(days=1)
    try:
        data = yf.download(ticker, start=date.isoformat(), end=end.isoformat(), progress=False)
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        pass
    return float("nan")


def stress_proxies(date: dt.date) -> Dict[str, float]:
    """Return simple stress proxies for a given date (may contain NaNs if yfinance missing)."""
    return {
        "vix_close": fetch_yf_close("^VIX", date),
        "usdcnh_close": fetch_yf_close("USDCNH=X", date),
        "copper_close": fetch_yf_close("HG=F", date),
    }


def contextual_events(date_str: str, keywords: List[str] | None = None) -> Dict[str, Any]:
    """
    Aggregate contextual signals for a given date:
    - market wrap headlines (Reuters RSS filtered by keywords)
    - macro calendar entries (TradingEconomics if key present)
    - stress proxies (VIX, USDCNH, Copper)
    """
    keywords = keywords or ["china", "greece", "yuan", "debt", "stocks fell", "liquidity", "fx"]
    d = parse_date(date_str)
    wrap = fetch_reuters_wrap(keywords=keywords)
    macro = fetch_tradingeconomics_calendar(date_str, date_str)
    stress = stress_proxies(d)
    return {
        "date": d.isoformat(),
        "market_wrap": wrap,
        "macro_events": macro,
        "stress": stress,
    }


if __name__ == "__main__":
    import pprint

    today = dt.date.today().isoformat()
    pprint.pp(contextual_events(today))

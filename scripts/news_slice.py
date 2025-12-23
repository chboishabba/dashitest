"""
Fetch headlines or event summaries for a given time slice to align with bad_day spikes.

Supports:
- NewsAPI (headlines): requires NEWSAPI_KEY env var.
- GDELT (event counts + tone): no key required.
- RSS (headlines): no key; provide one or more feed URLs.

Examples:
  PYTHONPATH=. python scripts/news_slice.py --provider newsapi --start 2025-03-01T09:00:00Z --end 2025-03-01T12:00:00Z --out logs/news_slice.csv
  PYTHONPATH=. python scripts/news_slice.py --provider gdelt --start 2025-03-01T00:00:00Z --end 2025-03-01T23:59:59Z --out logs/gdelt_slice.csv
"""

import argparse
import os
import pathlib
from typing import Optional

import pandas as pd
import requests


def parse_iso(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True, errors="raise")


def newsapi_fetch(start: pd.Timestamp, end: pd.Timestamp, query: str, page_size: int = 100) -> pd.DataFrame:
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("Set NEWSAPI_KEY in the environment for NewsAPI access.")
    url = "https://newsapi.org/v2/everything"
    headers = {"X-Api-Key": api_key}
    params = {
        "q": query or "",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "page": 1,
    }
    articles = []
    while True:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", [])
        for a in arts:
            articles.append(
                {
                    "ts": a.get("publishedAt"),
                    "title": a.get("title"),
                    "source": (a.get("source") or {}).get("name"),
                    "url": a.get("url"),
                    "description": a.get("description"),
                }
            )
        if len(arts) < page_size:
            break
        params["page"] += 1
    df = pd.DataFrame(articles)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def gdelt_fetch(start: pd.Timestamp, end: pd.Timestamp, query: Optional[str], maxrows: int, timeout: float = 15.0) -> pd.DataFrame:
    """
    Fetch GDELT events summary within [start, end]; query is optional keyword filter.
    """
    def to_gdelt(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y%m%d%H%M%S")

    url = "https://api.gdeltproject.org/api/v2/events/summary"
    params = {
        "maxrecords": maxrows,
        "format": "json",
        "startdatetime": to_gdelt(start),
        "enddatetime": to_gdelt(end),
    }
    if query:
        params["query"] = query
    r = requests.get(url, params=params, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # attach response for downstream handlers
        e.response = r  # type: ignore[attr-defined]
        raise
    data = r.json()
    events = data.get("events", [])
    df = pd.DataFrame(events)
    if not df.empty:
        if "eventtime" in df.columns:
            df["ts"] = pd.to_datetime(df["eventtime"], utc=True, errors="coerce")
        elif "SQLDATE" in df.columns:
            df["ts"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", utc=True, errors="coerce")
    return df


def rss_fetch(feed_urls, start: pd.Timestamp, end: pd.Timestamp, maxrows: int = 200, timeout: float = 10.0) -> pd.DataFrame:
    """
    Minimal RSS/Atom fetcher (no API key). Filters items within [start, end].
    """
    import xml.etree.ElementTree as ET

    rows = []

    def parse_time(text: str) -> Optional[pd.Timestamp]:
        if not text:
            return None
        ts = pd.to_datetime(text, utc=True, errors="coerce")
        return ts

    for url in feed_urls:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            print(f"[rss] failed {url}: {e}")
            continue
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            print(f"[rss] parse error {url}: {e}")
            continue

        # RSS 2.0: channel/item; Atom: feed/entry
        channel = root.find("channel")
        source = ""
        items = []
        if channel is not None:
            source_el = channel.findtext("title")
            source = source_el or ""
            items = channel.findall("item")
        else:
            source = root.findtext("{http://www.w3.org/2005/Atom}title") or ""
            items = root.findall("{http://www.w3.org/2005/Atom}entry")

        for it in items:
            title = it.findtext("title") or it.findtext("{http://www.w3.org/2005/Atom}title") or ""
            link_el = it.find("link")
            link = ""
            if link_el is not None:
                link = link_el.get("href") or link_el.text or ""
            pub = (
                it.findtext("pubDate")
                or it.findtext("{http://www.w3.org/2005/Atom}updated")
                or it.findtext("{http://purl.org/dc/elements/1.1/}date")
            )
            ts = parse_time(pub)
            if ts is None:
                continue
            if ts < start or ts > end:
                continue
            rows.append({"ts": ts, "title": title, "source": source, "url": link, "provider": "rss"})
            if len(rows) >= maxrows:
                break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["newsapi", "gdelt", "rss"], required=True)
    ap.add_argument("--start", required=True, help="ISO start time (UTC recommended), e.g. 2025-03-01T09:00:00Z")
    ap.add_argument("--end", required=True, help="ISO end time (UTC recommended)")
    ap.add_argument("--query", default="", help="Optional keyword filter.")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Optional CSV output path.")
    ap.add_argument("--maxrows", type=int, default=250, help="Max rows for providers that support it (GDELT/RSS).")
    ap.add_argument(
        "--feed-url",
        action="append",
        help="RSS/Atom feed URL (can be repeated). Required when --provider rss.",
    )
    args = ap.parse_args()

    start = parse_iso(args.start)
    end = parse_iso(args.end)
    if end <= start:
        raise ValueError("end must be after start")

    if args.provider == "newsapi":
        df = newsapi_fetch(start, end, args.query)
    elif args.provider == "gdelt":
        df = gdelt_fetch(start, end, args.query or None, args.maxrows)
    else:
        if not args.feed_url:
            raise ValueError("Provide at least one --feed-url when using provider=rss")
        df = rss_fetch(args.feed_url, start, end, maxrows=args.maxrows)

    if df.empty:
        print("No records returned.")
    else:
        print(df.head().to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

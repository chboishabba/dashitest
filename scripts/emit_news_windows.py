"""
Detect bad-structure windows from a trading log and fetch external event headlines (no API key required by default).

Defaults:
- Provider: GDELT (no key)
- Trigger: bad_flag == 1 and p_bad >= 0.7
- Windows: contiguous triggers merged if within --max-gap-min, padded by --pad-min.

Example:
  PYTHONPATH=. python scripts/emit_news_windows.py \
    --log logs/trading_log.csv \
    --out-dir logs/news_events \
    --p-bad-hi 0.7 \
    --max-gap-min 15 \
    --pad-min 60
"""

import argparse
import pathlib
from typing import List, Tuple

import pandas as pd

# reuse provider fetchers
from scripts.news_slice import gdelt_fetch, newsapi_fetch, rss_fetch  # type: ignore


def contiguous_windows(ts: pd.Series, max_gap: pd.Timedelta) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    ts = ts.sort_values()
    if ts.empty:
        return []
    windows = []
    start = prev = ts.iloc[0]
    for t in ts.iloc[1:]:
        if t - prev <= max_gap:
            prev = t
            continue
        windows.append((start, prev))
        start = prev = t
    windows.append((start, prev))
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=pathlib.Path, default=pathlib.Path("logs/trading_log.csv"))
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("logs/news_events"))
    ap.add_argument("--p-bad-hi", type=float, default=0.7, help="p_bad threshold to trigger a window.")
    ap.add_argument("--max-gap-min", type=float, default=15.0, help="Merge triggers separated by at most this gap (minutes).")
    ap.add_argument("--pad-min", type=float, default=60.0, help="Pad each window start/end by this many minutes.")
    ap.add_argument("--provider", choices=["gdelt", "newsapi", "rss"], default="gdelt")
    ap.add_argument("--query", default="", help="Optional keyword filter for provider.")
    ap.add_argument("--maxrows", type=int, default=250, help="Max rows for provider fetch (GDELT/RSS).")
    ap.add_argument(
        "--feed-url",
        action="append",
        help="RSS/Atom feed URL (repeatable) used when provider=rss.",
    )
    args = ap.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"log not found: {args.log}")

    df = pd.read_csv(args.log)
    if "ts" not in df.columns:
        raise ValueError("Log must contain a 'ts' column with timestamps. Re-run trader with timestamp support.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    mask = (df["bad_flag"] == 1) & (df["p_bad"] >= args.p_bad_hi)
    trigger_ts = df.loc[mask, "ts"].dropna()
    if trigger_ts.empty:
        print("No bad windows found with current thresholds.")
        return

    windows = contiguous_windows(trigger_ts, pd.Timedelta(minutes=args.max_gap_min))
    pad = pd.Timedelta(minutes=args.pad_min)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for idx, (start, end) in enumerate(windows):
        w_start = start - pad
        w_end = end + pad
        if args.provider == "gdelt":
            events = gdelt_fetch(w_start, w_end, args.query or None, args.maxrows)
        elif args.provider == "newsapi":
            events = newsapi_fetch(w_start, w_end, args.query, page_size=min(args.maxrows, 100))
        else:
            if not args.feed_url:
                raise ValueError("Provide at least one --feed-url when provider=rss")
            events = rss_fetch(args.feed_url, w_start, w_end, maxrows=args.maxrows)
        fname = args.out_dir / f"events_{idx:03d}_{w_start.strftime('%Y%m%dT%H%M%S')}_{w_end.strftime('%Y%m%dT%H%M%S')}.csv"
        events.to_csv(fname, index=False)
        top_codes = ""
        if "EventRootCode" in events.columns:
            vc = events["EventRootCode"].value_counts().head(3)
            top_codes = ";".join(f"{k}:{v}" for k, v in vc.items())
        mean_tone = events["AvgTone"].mean() if "AvgTone" in events.columns and not events.empty else float("nan")
        summaries.append(
            {
                "window_id": idx,
                "start": w_start,
                "end": w_end,
                "triggers": ((trigger_ts >= start) & (trigger_ts <= end)).sum(),
                "events": len(events),
                "top_codes": top_codes,
                "mean_tone": mean_tone,
                "path": fname,
            }
        )
        print(f"[window {idx:03d}] {w_start} → {w_end} | triggers={(trigger_ts >= start).sum()} | events={len(events)} -> {fname.name}")

    summary_df = pd.DataFrame(summaries)
    summary_path = args.out_dir / "events_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

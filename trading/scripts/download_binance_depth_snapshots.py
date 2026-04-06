from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_downloader import ensure_dir, fetch_binance_depth_snapshot


def _default_output_path(symbol: str) -> pathlib.Path:
    stamp = time.strftime("%Y-%m-%d", time.gmtime())
    return pathlib.Path("data/raw/binance_depth") / f"{symbol.upper()}_depth_{stamp}.ndjson"


def collect_depth_snapshots(
    *,
    symbol: str,
    limit: int,
    count: int,
    poll_interval: float,
    out_path: pathlib.Path,
    flush_every: int,
) -> pathlib.Path:
    ensure_dir(out_path.parent)
    written = 0
    buffer: list[str] = []

    with out_path.open("a", encoding="utf-8") as fh:
        while count <= 0 or written < count:
            snapshot = fetch_binance_depth_snapshot(symbol=symbol, limit=limit)
            buffer.append(json.dumps(snapshot, separators=(",", ":")))
            written += 1

            if len(buffer) >= flush_every:
                fh.write("\n".join(buffer) + "\n")
                fh.flush()
                buffer.clear()

            if count > 0 and written >= count:
                break
            time.sleep(max(0.0, poll_interval))

        if buffer:
            fh.write("\n".join(buffer) + "\n")
            fh.flush()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect periodic public Binance depth snapshots into NDJSON."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol, e.g. BTCUSDT.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        choices=[5, 10, 20, 50, 100, 500, 1000, 5000],
        help="Depth levels returned by Binance.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of snapshots to collect. Use 0 or less to run indefinitely.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between snapshots.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="NDJSON output path. Defaults to data/raw/binance_depth/<symbol>_depth_<date>.ndjson",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Buffered writes before flush.",
    )
    args = parser.parse_args()

    out_path = args.out or _default_output_path(args.symbol)
    path = collect_depth_snapshots(
        symbol=args.symbol,
        limit=args.limit,
        count=args.count,
        poll_interval=args.poll_interval,
        out_path=out_path,
        flush_every=max(1, args.flush_every),
    )
    print(f"[binance-depth] wrote snapshots to {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timezone


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rotate_dir(src_dir: Path, archive_dir: Path, keep_latest_link: Path | None) -> int:
    src_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(src_dir.glob("*.csv.gz"))
    if len(chunks) <= 1:
        # nothing to rotate if only latest chunk exists
        return 0

    newest = chunks[-1]
    rotated = 0
    for p in chunks[:-1]:
        dst = archive_dir / p.name
        if not dst.exists():
            shutil.move(str(p), str(dst))
            rotated += 1

    if keep_latest_link is not None and newest.exists():
        keep_latest_link.parent.mkdir(parents=True, exist_ok=True)
        tmp = keep_latest_link.with_suffix(f".tmp.{utc_stamp()}")
        shutil.copyfile(newest, tmp)
        tmp.replace(keep_latest_link)

    return rotated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="logs/binance_stream", help="Directory where live gz chunks are written")
    ap.add_argument("--archive", default="logs/binance_stream/archive", help="Archive directory for rotated chunks")
    ap.add_argument("--latest", default="logs/binance_stream/latest.csv.gz", help="Copy/symlink to the most recent closed chunk")
    args = ap.parse_args()

    moved = rotate_dir(Path(args.src), Path(args.archive), Path(args.latest))
    print(f"rotated_chunks={moved}")


if __name__ == "__main__":
    main()

import datetime
import re
from pathlib import Path


_TS_RE = re.compile(r".*_[0-9]{8}T[0-9]{6}Z$")


def timestamped_path(path: str) -> Path:
    if path is None:
        raise ValueError("path must be provided")
    p = Path(path)
    if _TS_RE.match(p.stem):
        return p
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if p.suffix:
        return p.with_name(f"{p.stem}_{ts}{p.suffix}")
    return p.with_name(f"{p.name}_{ts}")


def timestamped_prefix(prefix: str) -> str:
    if prefix is None:
        raise ValueError("prefix must be provided")
    if _TS_RE.match(prefix):
        return prefix
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"

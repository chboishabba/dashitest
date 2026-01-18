#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _coerce_ts_ms(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e14:
            return int(ts / 1e6)  # ns -> ms
        if ts > 1e11:
            return int(ts)  # ms
        if ts > 1e9:
            return int(ts * 1000.0)  # seconds -> ms
        return int(ts * 1000.0)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000.0)
        except ValueError:
            return None
    return None


def _load_phase8(path: Path, symbol_filter: str | None) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            symbol = str(entry.get("symbol") or entry.get("target") or "default")
            if symbol_filter and symbol != symbol_filter:
                continue
            ts_ms = _coerce_ts_ms(entry.get("timestamp"))
            if ts_ms is None:
                continue
            entry["_ts_ms"] = ts_ms
            buckets.setdefault(symbol, []).append(entry)
    for entries in buckets.values():
        entries.sort(key=lambda item: item["_ts_ms"])
    return buckets


def _pick_phase8(
    entries: list[dict],
    idx: int,
    ts_ms: int,
) -> tuple[int, dict | None]:
    if not entries:
        return idx, None
    while idx + 1 < len(entries) and entries[idx + 1]["_ts_ms"] <= ts_ms:
        idx += 1
    if entries[idx]["_ts_ms"] <= ts_ms:
        return idx, entries[idx]
    return idx, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge Phase-8 readiness log into tower NDJSON.")
    ap.add_argument("--tower-log", required=True, help="Tower NDJSON log to enrich.")
    ap.add_argument("--phase8-log", required=True, help="Phase-8 audit NDJSON log.")
    ap.add_argument("--out", required=True, help="Output NDJSON path.")
    ap.add_argument(
        "--max-delta-ms",
        type=int,
        default=120000,
        help="Max time delta (ms) to apply a Phase-8 snapshot to a tower record.",
    )
    ap.add_argument("--symbol", default="", help="Optional symbol filter (exact match).")
    args = ap.parse_args()

    tower_path = Path(args.tower_log)
    phase8_path = Path(args.phase8_log)
    out_path = Path(args.out)
    symbol_filter = args.symbol.strip() or None
    max_delta = int(args.max_delta_ms)

    buckets = _load_phase8(phase8_path, symbol_filter)
    indices = {symbol: 0 for symbol in buckets}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tower_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            symbol = str(record.get("symbol") or "default")
            if symbol_filter and symbol != symbol_filter:
                continue
            ts_ms = _coerce_ts_ms(record.get("ts"))
            if ts_ms is None:
                ts_ms = _coerce_ts_ms(record.get("t"))
            entry = None
            if ts_ms is not None and symbol in buckets:
                idx, candidate = _pick_phase8(buckets[symbol], indices[symbol], ts_ms)
                indices[symbol] = idx
                if candidate is not None and abs(ts_ms - candidate["_ts_ms"]) <= max_delta:
                    entry = candidate
            if entry is not None:
                p8 = record.get("P8") or {}
                ready = entry.get("phase8_ready")
                ready_count = entry.get("phase8_ready_count", entry.get("ready_count"))
                required = entry.get("phase8_required", entry.get("required"))
                window = entry.get("phase8_window", entry.get("window_size"))
                p8.update(
                    {
                        "available": True,
                        "ready_count": ready_count,
                        "required": required,
                        "window": window,
                        "open": ready,
                        "reason": entry.get("phase8_reason"),
                    }
                )
                if ready_count is not None and required:
                    try:
                        p8["A8"] = min(1.0, float(ready_count) / float(required))
                    except (TypeError, ValueError, ZeroDivisionError):
                        p8["A8"] = None
                elif ready is not None:
                    p8["A8"] = 1.0 if ready else 0.0
                record["P8"] = p8
            dst.write(json.dumps(record, ensure_ascii=True, allow_nan=False))
            dst.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

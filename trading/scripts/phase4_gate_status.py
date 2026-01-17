#!/usr/bin/env python3
"""Summarize the latest Phase-4 gate decisions per tape per day."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def _parse_log(log_path: Path) -> dict[tuple[str, str], tuple[datetime, Mapping[str, object]]]:
    entries: dict[tuple[str, str], tuple[datetime, Mapping[str, object]]] = {}
    if not log_path.exists():
        raise SystemExit(f"missing monitor log: {log_path}")
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = payload.get("timestamp")
            target = payload.get("target", "unknown")
            if not timestamp:
                continue
            try:
                ts = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
            date = ts.date().isoformat()
            key = (date, target)
            existing = entries.get(key)
            if existing is None or ts > existing[0]:
                entries[key] = (ts, payload)
    return entries


def _format_counts(counts: Mapping[str, int]) -> str:
    parts = []
    for name in ("T", "R"):
        value = counts.get(name)
        if value is not None:
            parts.append(f"{name}={value}")
    return " ".join(parts) if parts else "no counts"


def _build_markdown(entries: dict[tuple[str, str], tuple[datetime, Mapping[str, object]]], days: int) -> str:
    lines: list[str] = []
    lines.append("# Phase-4 Gate Status")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()} UTC")
    lines.append("")
    dates: dict[str, list[tuple[str, Mapping[str, object]]]] = {}
    for (date, target), (_, payload) in entries.items():
        dates.setdefault(date, []).append((target, payload))
    recent_dates = sorted(dates.keys(), reverse=True)[:days]
    if not recent_dates:
        lines.append("No gate activity recorded yet. Run the Phase-4 density monitor to create entries.")
        return "\n".join(lines)
    for date in recent_dates:
        lines.append(f"## {date}")
        for target, payload in sorted(dates[date]):
            gate_status = "OPEN" if payload.get("gate_open") else "closed"
            gate_reasons = payload.get("gate_reasons") or []
            blocking_reason = payload.get("blocking_reason", "not recorded")
            counts = payload.get("counts", {})
            phase7_ready = payload.get("phase7_ready")
            phase7_reason = payload.get("phase7_reason")
            lines.append(f"### {target}")
            lines.append(f"- Status: {gate_status}")
            lines.append(f"- Counts: {_format_counts(counts)}")
            if phase7_ready is None:
                lines.append("- Phase-07: (not recorded)")
            else:
                lines.append(f"- Phase-07: {'ready' if phase7_ready else 'blocked'}")
                if phase7_reason:
                    lines.append(f"- Phase-07 reason: {phase7_reason}")
            if gate_reasons:
                lines.append(f"- Gate reasons: {', '.join(str(r) for r in gate_reasons)}")
            else:
                lines.append("- Gate reasons: (none)")
            lines.append(f"- Blocking reason: {blocking_reason or '(none)'}")
            lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a Markdown summary of the Phase-4 gate log.")
    ap.add_argument(
        "--log",
        type=Path,
        default=Path("logs/phase4/density_monitor.log"),
        help="Path to the density monitor log (JSON lines).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("phase4_gate_status.md"),
        help="Markdown file to overwrite with the latest gate summary.",
    )
    ap.add_argument(
        "--days",
        type=int,
        default=3,
        help="How many recent calendar days to include in the summary.",
    )
    args = ap.parse_args()

    entries = _parse_log(args.log)
    markdown = _build_markdown(entries, args.days)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote gate status summary ({len(entries)} entries considered) to {args.output}")


if __name__ == "__main__":
    main()

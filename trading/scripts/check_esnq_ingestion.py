#!/usr/bin/env python3
"""Check the Phase-4 ingestion contract (proposal + price files + monitor log)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _load_targets(config_path: Path) -> Iterable[dict[str, Path]]:
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    raw = json.loads(config_path.read_text())
    targets = raw.get("targets")
    if not isinstance(targets, list):
        raise SystemExit("config must contain a 'targets' list")
    for entry in targets:
        name = entry.get("name")
        proposal = entry.get("proposal_log")
        prices = entry.get("prices_csv")
        if not (name and proposal and prices):
            raise SystemExit("each target needs 'name', 'proposal_log', and 'prices_csv'")
        yield {
            "name": name,
            "proposal_log": Path(proposal),
            "prices_csv": Path(prices),
        }


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        return [col.strip() for col in next(reader, [])]


def _count_rows(path: Path, max_rows: int = 1000) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        next(fh, None)
        for _ in fh:
            count += 1
            if count >= max_rows:
                break
    return count


def _load_monitor_entries(log_path: Path) -> dict[str, tuple[datetime, dict[str, object]]]:
    entries: dict[str, tuple[datetime, dict[str, object]]] = {}
    if not log_path.exists():
        return entries
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            target = payload.get("target")
            timestamp = payload.get("timestamp")
            if not target or not timestamp:
                continue
            try:
                ts = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
            existing = entries.get(target)
            if existing is None or ts > existing[0]:
                entries[target] = (ts, payload)
    return entries


def _format_columns(columns: Iterable[str]) -> str:
    return ", ".join(columns)


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirm ES/NQ ingestion files satisfy the Phase-4 contract.")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase4_monitor_targets.json"),
        help="Config listing proposal + price targets.",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=Path("logs/phase4/density_monitor.log"),
        help="Phase-4 density monitor log path.",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Maximum proposal rows to count per file (for a quick sanity check).",
    )
    args = ap.parse_args()

    entries = _load_monitor_entries(args.log)
    problems: list[str] = []
    print("ES/NQ ingestion contract status")
    print("Config:", args.config)
    print("Monitor log:", args.log)
    print()
    expected_proposal_columns = {"ts", "i", "action", "ontology_k", "size_pred"}
    expected_price_columns = {"timestamp", "close"}

    for target in _load_targets(args.config):
        name = target["name"]
        proposal_path = target["proposal_log"]
        prices_path = target["prices_csv"]
        print(f"Target: {name}")
        if not proposal_path.exists():
            msg = f"- Proposal log missing: {proposal_path}"
            print(msg)
            problems.append(msg)
        else:
            header = _read_header(proposal_path)
            missing = expected_proposal_columns - set(header)
            status = "OK"
            if missing:
                status = f"missing columns: {', '.join(sorted(missing))}"
                problems.append(f"Target {name} proposals missing {missing}")
            rows = _count_rows(proposal_path, max_rows=args.samples)
            print(f"- Proposal rows (sampled): {rows} / header: {len(header)} cols")
            print(f"- Proposal columns: {_format_columns(header)}")
            print(f"- Proposal status: {status}")
        if not prices_path.exists():
            msg = f"- Prices CSV missing: {prices_path}"
            print(msg)
            problems.append(msg)
        else:
            header = _read_header(prices_path)
            missing = expected_price_columns - set(header)
            status = "OK" if not missing else f"missing columns: {', '.join(sorted(missing))}"
            if missing:
                problems.append(f"Target {name} prices missing {missing}")
            print(f"- Prices columns: {_format_columns(header)}")
            print(f"- Prices status: {status}")
        entry = entries.get(name)
        if entry:
            ts, payload = entry
            gate_open = payload.get("gate_open")
            blocking_reason = payload.get("blocking_reason") or "(none logged)"
            gate_reasons = payload.get("gate_reasons") or []
            reasons_line = ", ".join(str(r) for r in gate_reasons) if gate_reasons else "(none)"
            print(f"- Last monitor entry: {ts.isoformat()} (gate_open={gate_open})")
            print(f"  gate_reasons: {reasons_line}")
            print(f"  blocking_reason: {blocking_reason}")
        else:
            print("- No monitor entry found for this target yet.")
            problems.append(f"target {name} has no log entry")
        print("---")

    if problems:
        print("Problems detected:")
        for problem in problems:
            print("-", problem)
        raise SystemExit("ingestion contract incomplete")
    print("All targets satisfy the Phase-4 ingestion contract (or have sampled proposals).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Target:
    name: str
    proposal_log: Path
    prices_csv: Path


def _parse_target(spec: str) -> Target:
    if "=" not in spec:
        raise SystemExit(f"target spec must be NAME=proposal,prices; got '{spec}'")
    name, rest = spec.split("=", 1)
    if "," not in rest:
        raise SystemExit(f"target spec must provide proposal and prices separated by ','; got '{rest}'")
    proposal, prices = rest.split(",", 1)
    return Target(
        name=name.strip(),
        proposal_log=Path(proposal.strip()),
        prices_csv=Path(prices.strip()),
    )


def _run_diagnostics(target: Target, args: Any, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{target.name}-{stamp}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_size_per_ontology.py"),
        "--proposal-log",
        str(target.proposal_log),
        "--prices-csv",
        str(target.prices_csv),
        "--out",
        str(out_path),
        "--horizon",
        str(args.horizon),
        "--clip",
        str(args.clip),
        "--size-column",
        args.size_column,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"diagnostics failed for {target.name} ({proc.returncode}): {proc.stderr.strip()}"
        )
    payload = json.loads(proc.stdout)
    payload["__diag_out"] = str(out_path)
    return payload


def _bin_sort_key(bin_name: str) -> str | float:
    try:
        return float(bin_name)
    except Exception:
        return bin_name


def _has_monotone_run(values: list[float], length: int) -> bool:
    if length <= 1 or not values:
        return True
    inc = dec = 1
    for i in range(1, len(values)):
        if values[i] >= values[i - 1]:
            inc += 1
        else:
            inc = 1
        if values[i] <= values[i - 1]:
            dec += 1
        else:
            dec = 1
        if inc >= length or dec >= length:
            return True
    return False


def _count_sign_flips(values: list[float]) -> int:
    last_sign = None
    flips = 0
    for val in values:
        if val > 0:
            sign = 1
        elif val < 0:
            sign = -1
        else:
            continue
        if last_sign is None:
            last_sign = sign
            continue
        if sign != last_sign:
            flips += 1
            last_sign = sign
    return flips


def _evaluate_gate(payload: dict[str, Any], args: Any, history_entries: deque[dict[str, Any]]) -> tuple[bool, list[str], str]:
    counts = payload.get("counts", {})
    diagnostics = payload.get("diagnostics", {})
    blocking: list[str] = []
    required_onts = ("T", "R")

    for ont in required_onts:
        if counts.get(ont, 0) < args.min_rows_per_ontology:
            blocking.append(
                f"{ont} rows {counts.get(ont, 0)} < {args.min_rows_per_ontology}"
            )

    window_entries = list(history_entries)[-args.bin_persistence_window :]
    persistence_ready = len(window_entries) >= args.bin_persistence_required
    if not persistence_ready:
        blocking.append(
            f"waiting for {args.bin_persistence_required} persistence windows (have {len(window_entries)})"
        )

    def _bin_history(bin_name: str, ont_label: str) -> tuple[int, list[float]]:
        hits = 0
        medians: list[float] = []
        for entry in window_entries:
            ont_bins = entry.get("bins", {}).get(ont_label, {})
            data = ont_bins.get(bin_name)
            if not data:
                continue
            if data.get("count", 0) >= args.min_rows_per_bin:
                hits += 1
                medians.append(data.get("median", 0.0))
        return hits, medians

    ready_onts: list[str] = []
    for ont in required_onts:
        if counts.get(ont, 0) < args.min_rows_per_ontology:
            continue
        stats = diagnostics.get(ont, {})
        persistent_bins: list[tuple[str, float, int]] = []
        for bin_name, entry in stats.items():
            count = entry.get("count", 0)
            if count < args.min_rows_per_bin:
                continue
            hits, medians_hist = _bin_history(bin_name, ont)
            if not persistence_ready or hits < args.bin_persistence_required:
                continue
            flips = _count_sign_flips(medians_hist)
            if flips > args.median_flip_max:
                blocking.append(
                    f"{ont} bin {bin_name} flips {flips} (> {args.median_flip_max})"
                )
                continue
            persistent_bins.append((bin_name, entry.get("median", 0.0), count))
        if len(persistent_bins) < args.min_bins:
            blocking.append(
                f"{ont} has {len(persistent_bins)} persistent bins (needs {args.min_bins})"
            )
            continue
        counts_list = [count for _, _, count in persistent_bins]
        ratio = min(counts_list) / max(counts_list) if max(counts_list) > 0 else 0.0
        if ratio < args.min_bin_balance:
            blocking.append(
                f"{ont} bin balance {ratio:.2f} < {args.min_bin_balance}"
            )
            continue
        sorted_bins = sorted(persistent_bins, key=lambda tpl: _bin_sort_key(tpl[0]))
        medians = [median for _, median, _ in sorted_bins]
        effect = max(medians) - min(medians) if medians else 0.0
        if effect < args.min_effect_size:
            blocking.append(
                f"{ont} effect {effect:.6f} < {args.min_effect_size}"
            )
            continue
        if not _has_monotone_run(medians, args.min_monotone_bins):
            blocking.append(
                f"{ont} medians not monotone for {args.min_monotone_bins} bins"
            )
            continue
        ready_onts.append(ont)

    total_tr = counts.get("T", 0) + counts.get("R", 0)
    balance_ok = True
    if total_tr > 0:
        min_count = min(counts.get("T", 0), counts.get("R", 0))
        ratio = min_count / total_tr
        if ratio < args.min_ontology_ratio:
            blocking.append(
                f"ontology balance {ratio:.2f} < {args.min_ontology_ratio}"
            )
            balance_ok = False
    else:
        balance_ok = False

    gate_ready = (
        persistence_ready
        and balance_ok
        and len(ready_onts) == len(required_onts)
    )
    open_msg = f"{'&'.join(required_onts)} bins dense"
    if not gate_ready and not blocking:
        blocking.append("no ready ontology bins")
    return gate_ready, blocking, open_msg


def _summarize_bins(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    diag = payload.get("diagnostics", {})
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for ont, bins in diag.items():
        summary[ont] = {}
        for bin_name, stats in bins.items():
            summary[ont][bin_name] = {
                "count": stats.get("count", 0),
                "median": stats.get("median", 0.0),
            }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-4 density monitor.")
    ap.add_argument(
        "--target",
        action="append",
        required=True,
        help="NAME=proposal_log.csv,prices.csv (repeat for each tape)",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Seconds between iterations; 0 means run once.",
    )
    ap.add_argument(
        "--history",
        type=int,
        default=20,
        help="How many recent iterations to keep in memory for reporting.",
    )
    ap.add_argument(
        "--monitor-log",
        type=Path,
        default=Path("logs/phase4/density_monitor.log"),
        help="File path to append per-iteration summaries.",
    )
    ap.add_argument(
        "--diag-out-dir",
        type=Path,
        default=Path("logs/phase4/density_monitor"),
        help="Directory to store diagnostic payloads.",
    )
    ap.add_argument("--horizon", type=int, default=1, help="Forward horizon (h).")
    ap.add_argument("--clip", type=float, default=0.02, help="Clip for forward returns.")
    ap.add_argument("--size-column", type=str, default="size_pred", help="Size column to monitor.")
    ap.add_argument(
        "--min-rows-per-ontology",
        type=int,
        default=120,
        help="Minimum T or R rows before Phase-4 can train.",
    )
    ap.add_argument(
        "--min-bins",
        type=int,
        default=3,
        help="Minimum number of size bins required for gate-ready ontology.",
    )
    ap.add_argument(
        "--min-rows-per-bin",
        type=int,
        default=40,
        help="Minimum rows per size bin before Phase-4 can train.",
    )
    ap.add_argument(
        "--min-bin-balance",
        type=float,
        default=0.35,
        help="Require min(bin_count) / max(bin_count) ≥ this ratio.",
    )
    ap.add_argument(
        "--min-effect-size",
        type=float,
        default=0.0005,
        help="Minimum median difference between active size bins (in log-return units).",
    )
    ap.add_argument(
        "--min-monotone-bins",
        type=int,
        default=2,
        help="Minimum length of ordered medians to accept the bin shape.",
    )
    ap.add_argument(
        "--bin-persistence-window",
        type=int,
        default=8,
        help="How many past iterations count as persistence windows.",
    )
    ap.add_argument(
        "--bin-persistence-required",
        type=int,
        default=6,
        help="How many windows a bin must appear in before counting toward density.",
    )
    ap.add_argument(
        "--median-flip-max",
        type=int,
        default=1,
        help="Maximum allowed median sign flips in the persistence window.",
    )
    ap.add_argument(
        "--min-ontology-ratio",
        type=float,
        default=0.25,
        help="Require min(T, R) / (T+R) ≥ this ratio before gate can open.",
    )
    ap.add_argument(
        "--consecutive-passes",
        type=int,
        default=3,
        help="Number of consecutive raw OPENs required before gate reports OPEN.",
    )
    ap.add_argument(
        "--no-debounce",
        action="store_true",
        help="Shortcut: equivalent to `--consecutive-passes 1`.",
    )
    ap.add_argument(
        "--test-vector",
        type=str,
        default="",
        help="Optional tag describing synthetic/amplitude-injected runs so OPENs can be traced.",
    )
    args = ap.parse_args()
    if args.no_debounce:
        args.consecutive_passes = 1

    targets = [_parse_target(spec) for spec in args.target]
    for target in targets:
        if not target.proposal_log.exists():
            raise SystemExit(f"missing proposal log for {target.name}: {target.proposal_log}")
        if not target.prices_csv.exists():
            raise SystemExit(f"missing prices csv for {target.name}: {target.prices_csv}")

    args.monitor_log.parent.mkdir(parents=True, exist_ok=True)
    stats_history: dict[str, deque[dict[str, Any]]] = {
        target.name: deque(maxlen=args.history) for target in targets
    }

    def _log_entry(entry: dict[str, Any]) -> None:
        with args.monitor_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry))
            fh.write("\n")

    iteration = 0
    while True:
        iteration += 1
        now = datetime.utcnow().isoformat()
        for target in targets:
            try:
                payload = _run_diagnostics(target, args, args.diag_out_dir)
            except SystemExit as exc:
                print(f"[{now}] {target.name} diagnostics failed: {exc}", file=sys.stderr)
                continue
            history = stats_history[target.name]
            raw_gate_open, blocking_reasons, open_msg = _evaluate_gate(payload, args, history)
            consecutive_needed = max(1, args.consecutive_passes)
            consecutive_ok = True
            consecutive_count = 0
            if consecutive_needed > 1:
                needed = consecutive_needed - 1
                prev_entries = list(history)[-needed:]
                for entry in reversed(prev_entries):
                    if entry.get("raw_gate_open"):
                        consecutive_count += 1
                    else:
                        break
                consecutive_ok = consecutive_count >= needed
            gate_open = raw_gate_open and consecutive_ok
            if raw_gate_open and not consecutive_ok:
                blocking_reasons.insert(
                    0,
                    f"{consecutive_needed} consecutive OPENs required (have {consecutive_count})",
                )
            if gate_open:
                gate_reasons = [open_msg]
                blocking_reason = ""
            else:
                if not blocking_reasons:
                    blocking_reasons.append("gate blocked (no reason)")
                gate_reasons = blocking_reasons
                blocking_reason = blocking_reasons[0]
            bins_summary = _summarize_bins(payload)
            entry = {
                "timestamp": now,
                "iteration": iteration,
                "target": target.name,
                "counts": payload.get("counts", {}),
                "bins": bins_summary,
                "gate_open": gate_open,
                "raw_gate_open": raw_gate_open,
                "gate_reasons": gate_reasons,
                "blocking_reason": blocking_reason,
                "diag_out": payload.get("__diag_out"),
                "test_vector": args.test_vector or None,
            }
            history.append(entry)
            _log_entry(entry)
            status = "OPEN" if gate_open else "closed"
            counts = entry["counts"]
            t_count = counts.get("T", 0)
            r_count = counts.get("R", 0)
            open_rate = sum(1 for e in history if e["gate_open"]) / len(history)
            overflow = f" test_vector={args.test_vector}" if args.test_vector else ""
            print(
                f"[{now}] {target.name} gate={status} "
                f"T={t_count} R={r_count} bins={len(bins_summary.get('T', {}))}/"
                f"{len(bins_summary.get('R', {}))} recent_open_rate={open_rate:.2f} "
                f"reasons={';'.join(gate_reasons)}{overflow}"
            )
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

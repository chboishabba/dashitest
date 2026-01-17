#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, StatisticsError
from typing import Any, Iterable, Sequence


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tail_entries(path: Path, window: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing execution log: {path}")
    tail: deque[dict[str, Any]] = deque(maxlen=window)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tail.append(entry)
    return list(tail)


def _extract_sample(entry: dict[str, Any]) -> tuple[float, float, float] | None:
    size = _safe_float(entry.get("size"))
    direction = _safe_float(entry.get("direction"))
    if size is None or size <= 0.0 or not direction:
        return None
    fill_price = _safe_float(entry.get("fill_price"))
    price_h = _safe_float(entry.get("price_t_h"))
    if fill_price is None or price_h is None:
        return None
    gross = direction * (price_h - fill_price)
    if not math.isfinite(gross):
        return None
    slip = _safe_float(entry.get("slippage_cost")) or 0.0
    execution = _safe_float(entry.get("execution_cost")) or 0.0
    cost_per_unit = (slip + execution) / size
    if not math.isfinite(cost_per_unit):
        return None
    net = gross - cost_per_unit
    if not math.isfinite(net):
        return None
    return gross, net, cost_per_unit


def _compute_metrics(samples: Sequence[tuple[float, float, float]]) -> dict[str, float]:
    gross_vals = [gross for gross, _, _ in samples]
    net_vals = [net for _, net, _ in samples]
    cost_vals = [cost for _, _, cost in samples]
    return {
        "rho_A_gross": median(gross_vals),
        "rho_A_net": median(net_vals),
        "cost_est": median(cost_vals),
    }


def _build_payload(
    target: str,
    entries: Sequence[dict[str, Any]],
    samples: Sequence[tuple[float, float, float]],
) -> dict[str, Any]:
    window = len(entries)
    count = len(samples)
    metrics: dict[str, float | None] = {"rho_A_net": None, "rho_A_gross": None, "cost_est": None}
    if samples:
        try:
            metrics.update(_compute_metrics(samples))
        except StatisticsError:
            metrics = {"rho_A_net": None, "rho_A_gross": None, "cost_est": None}
    ready = False
    reason = "no_density_rows"
    rho_net = metrics["rho_A_net"]
    if rho_net is None:
        ready = False
    else:
        ready = rho_net > 0.0
        reason = "asymmetry_density_ok" if ready else "net_asymmetry_nonpositive"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "phase7_ready": ready,
        "phase7_reason": reason,
        "phase7_metrics": {
            "rho_A_net": metrics["rho_A_net"],
            "rho_A_gross": metrics["rho_A_gross"],
            "cost_est": metrics["cost_est"],
            "window": window,
            "count": count,
        },
    }
    return payload


def _append_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str)
        fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-07 asymmetry status emitter.")
    ap.add_argument(
        "--execution-log",
        type=Path,
        required=True,
        help="Phase-5 execution JSONL log to derive asymmetry density from.",
    )
    ap.add_argument(
        "--phase7-log",
        type=Path,
        default=Path("logs/phase7/density_status.log"),
        help="JSONL status log to append.",
    )
    ap.add_argument(
        "--target",
        type=str,
        default="default",
        help="Target name written into each Phase-07 payload.",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=256,
        help="Number of recent rows to read from the execution log.",
    )
    args = ap.parse_args()
    window = max(1, args.window)
    entries = _tail_entries(args.execution_log, window)
    samples: list[tuple[float, float, float]] = []
    for entry in entries:
        sample = _extract_sample(entry)
        if sample is not None:
            samples.append(sample)
    payload = _build_payload(args.target, entries, samples)
    _append_line(args.phase7_log, payload)
    status = "ready" if payload["phase7_ready"] else "blocked"
    print(
        f"Wrote Phase-07 {status} line for {args.target} "
        f"(window={payload['phase7_metrics']['window']}, count={payload['phase7_metrics']['count']}) "
        f"to {args.phase7_log}"
    )


if __name__ == "__main__":
    main()

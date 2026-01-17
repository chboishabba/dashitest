#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import StatisticsError, median
from typing import Any


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iter_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing execution log: {path}")
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _tail_entries(path: Path, window: int) -> list[dict[str, Any]]:
    if window <= 0:
        return _iter_entries(path)
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
            if isinstance(entry, dict):
                tail.append(entry)
    return list(tail)


def _extract_sample(entry: dict[str, Any]) -> dict[str, float] | None:
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
    abs_move = abs(price_h - fill_price)
    return {
        "gross": gross,
        "net": net,
        "cost": cost_per_unit,
        "abs_move": abs_move,
    }


def _summarize(samples: list[dict[str, float]]) -> dict[str, float | int | bool | None]:
    if not samples:
        return {
            "count": 0,
            "rho_A_gross": None,
            "rho_A_net": None,
            "cost_est": None,
            "median_abs_move": None,
            "median_gross": None,
            "median_net": None,
            "cost_dominates_move": None,
        }
    try:
        gross_vals = [s["gross"] for s in samples]
        net_vals = [s["net"] for s in samples]
        cost_vals = [s["cost"] for s in samples]
        abs_vals = [s["abs_move"] for s in samples]
        rho_gross = median(gross_vals)
        rho_net = median(net_vals)
        cost_est = median(cost_vals)
        median_abs_move = median(abs_vals)
    except StatisticsError:
        return {
            "count": len(samples),
            "rho_A_gross": None,
            "rho_A_net": None,
            "cost_est": None,
            "median_abs_move": None,
            "median_gross": None,
            "median_net": None,
            "cost_dominates_move": None,
        }
    return {
        "count": len(samples),
        "rho_A_gross": rho_gross,
        "rho_A_net": rho_net,
        "cost_est": cost_est,
        "median_abs_move": median_abs_move,
        "median_gross": rho_gross,
        "median_net": rho_net,
        "cost_dominates_move": median_abs_move < cost_est if median_abs_move is not None else None,
    }


def _summarize_pred_edge(values: list[float], cost_est: float | None, median_abs_move: float | None) -> dict[str, float | None]:
    if not values:
        return {}
    try:
        med = median(values)
        abs_med = median([abs(v) for v in values])
    except StatisticsError:
        return {}
    out: dict[str, float | None] = {
        "pred_edge_median": med,
        "pred_edge_median_abs": abs_med,
        "pred_edge_vs_cost": None,
        "pred_edge_vs_move": None,
    }
    if cost_est and cost_est != 0.0:
        out["pred_edge_vs_cost"] = abs_med / cost_est
    if median_abs_move and median_abs_move != 0.0:
        out["pred_edge_vs_move"] = abs_med / median_abs_move
    return out


def _recompute_realized_pnl(entry: dict[str, Any], cost_per_unit: float | None = None) -> float | None:
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
    if cost_per_unit is None:
        slip = _safe_float(entry.get("slippage_cost")) or 0.0
        execution = _safe_float(entry.get("execution_cost")) or 0.0
        cost_per_unit = (slip + execution) / size
    if not math.isfinite(cost_per_unit):
        return None
    net_per_unit = gross - cost_per_unit
    pnl = net_per_unit * size
    return pnl if math.isfinite(pnl) else None


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Phase-07 asymmetry diagnostics")
    print(f"entries={report['entries']} window={report['window']} samples={summary['count']}")
    print(
        "rho_A_net={rho_A_net} rho_A_gross={rho_A_gross} cost_est={cost_est} "
        "median_abs_move={median_abs_move} cost_dominates_move={cost_dominates_move}".format(**summary)
    )
    pred_edge = report.get("pred_edge") or {}
    if pred_edge:
        print(
            "pred_edge_median_abs={pred_edge_median_abs} pred_edge_vs_cost={pred_edge_vs_cost} "
            "pred_edge_vs_move={pred_edge_vs_move}".format(**pred_edge)
        )
    horizons = report.get("by_horizon") or {}
    if horizons:
        print("per_horizon:")
        for horizon_key in sorted(horizons, key=str):
            metrics = horizons[horizon_key]
            print(
                f"  horizon={horizon_key} count={metrics['count']} "
                f"rho_A_net={metrics['rho_A_net']} rho_A_gross={metrics['rho_A_gross']} "
                f"cost_est={metrics['cost_est']} median_abs_move={metrics['median_abs_move']}"
            )


def _report(args: argparse.Namespace) -> None:
    entries = _tail_entries(args.execution_log, args.window)
    samples: list[dict[str, float]] = []
    horizons: dict[str, list[dict[str, float]]] = defaultdict(list)
    pred_edges: list[float] = []
    for entry in entries:
        sample = _extract_sample(entry)
        if sample is None:
            continue
        samples.append(sample)
        if args.group_by_horizon:
            horizon_val = entry.get(args.horizon_field)
            if horizon_val is not None:
                horizons[str(horizon_val)].append(sample)
        if "pred_edge" in entry:
            pred_edge = _safe_float(entry.get("pred_edge"))
            if pred_edge is not None:
                pred_edges.append(pred_edge)
    summary = _summarize(samples)
    report: dict[str, Any] = {
        "entries": len(entries),
        "window": args.window,
        "summary": summary,
    }
    if pred_edges:
        report["pred_edge"] = _summarize_pred_edge(
            pred_edges, summary["cost_est"], summary["median_abs_move"]
        )
    if args.group_by_horizon and horizons:
        report["by_horizon"] = {key: _summarize(vals) for key, vals in horizons.items()}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)


def _zero_cost(args: argparse.Namespace) -> None:
    entries = _iter_entries(args.execution_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for entry in entries:
            entry["slippage_cost"] = 0.0
            entry["execution_cost"] = 0.0
            if "realized_pnl" in entry:
                pnl = _recompute_realized_pnl(entry, cost_per_unit=0.0)
                if pnl is not None:
                    entry["realized_pnl"] = pnl
            json.dump(entry, fh, default=str)
            fh.write("\n")


def _inject_drift(args: argparse.Namespace) -> None:
    entries = _iter_entries(args.execution_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for entry in entries:
            size = _safe_float(entry.get("size"))
            direction = _safe_float(entry.get("direction"))
            fill_price = _safe_float(entry.get("fill_price"))
            if size and size > 0.0 and direction and fill_price is not None:
                slip = _safe_float(entry.get("slippage_cost")) or 0.0
                execution = _safe_float(entry.get("execution_cost")) or 0.0
                cost_per_unit = (slip + execution) / size
                if math.isfinite(cost_per_unit):
                    entry["price_t_h"] = fill_price + direction * (cost_per_unit + args.drift_margin)
                    if "realized_pnl" in entry:
                        pnl = _recompute_realized_pnl(entry, cost_per_unit=cost_per_unit)
                        if pnl is not None:
                            entry["realized_pnl"] = pnl
            json.dump(entry, fh, default=str)
            fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-07 asymmetry diagnostics and wake-up tests.")
    sub = ap.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="Report Phase-07 medians and scale checks.")
    report.add_argument("--execution-log", type=Path, required=True, help="Phase-5 execution JSONL log.")
    report.add_argument("--window", type=int, default=256, help="Tail window size (0 for full file).")
    report.add_argument("--group-by-horizon", action="store_true", help="Summarize medians by horizon field.")
    report.add_argument("--horizon-field", type=str, default="horizon", help="Entry field to group by.")
    report.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    report.set_defaults(func=_report)

    zero_cost = sub.add_parser("zero-cost", help="Emit a zero-cost copy of an execution log.")
    zero_cost.add_argument("--execution-log", type=Path, required=True, help="Phase-5 execution JSONL log.")
    zero_cost.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    zero_cost.set_defaults(func=_zero_cost)

    inject = sub.add_parser("inject-drift", help="Emit a drift-injected execution log.")
    inject.add_argument("--execution-log", type=Path, required=True, help="Phase-5 execution JSONL log.")
    inject.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    inject.add_argument(
        "--drift-margin",
        type=float,
        default=5.0,
        help="Positive margin added on top of per-unit costs.",
    )
    inject.set_defaults(func=_inject_drift)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

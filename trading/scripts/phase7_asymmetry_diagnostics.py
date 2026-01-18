#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from statistics import StatisticsError, median
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import phase07_eigen_boundary_check as phase07


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    try:
        return float(median(values))
    except StatisticsError:
        return None


def _summarize(decisions: list[phase07.DecisionRec], cost: phase07.CostModel) -> dict[str, Any]:
    dx, dm, x_prev = phase07.compute_deltas(decisions)
    if not dx:
        return {
            "count": 0,
            "rho_A": None,
            "sum_edge": None,
            "sum_cost": None,
            "n_support": 0,
            "activity_rate": 0.0,
            "median_dx": None,
            "median_cost": None,
            "cost_dominates_dx": None,
        }

    edges: list[float] = []
    costs: list[float] = []
    dx_active: list[float] = []
    cost_active: list[float] = []
    n_support = 0

    for i in range(len(dx)):
        m_t = decisions[i + 1].m
        edge_t = x_prev[i] * float(dm[i])
        cost_t = (cost.half_spread + cost.fee + cost.slip) * float(m_t) * dx[i]
        edges.append(edge_t)
        costs.append(cost_t)
        if m_t:
            dx_active.append(dx[i])
            cost_active.append(cost_t)
            n_support += 1

    sum_edge = float(sum(edges))
    sum_cost = float(sum(costs))
    rho = sum_edge / (sum_cost + cost.eps)
    activity_rate = n_support / max(1, len(dx))
    median_dx = _median(dx_active)
    median_cost = _median(cost_active)

    return {
        "count": len(dx),
        "rho_A": rho,
        "sum_edge": sum_edge,
        "sum_cost": sum_cost,
        "n_support": n_support,
        "activity_rate": activity_rate,
        "median_dx": median_dx,
        "median_cost": median_cost,
        "cost_dominates_dx": bool(median_dx is not None and median_cost is not None and median_dx < median_cost),
    }


def _report(args: argparse.Namespace) -> None:
    symbol = args.symbol.strip() or None
    decisions = phase07.load_decisions_ndjson(Path(args.decisions_ndjson), symbol)
    if args.window > 0:
        decisions = decisions[-args.window :]

    cost = phase07.CostModel(args.half_spread, args.fee, args.slip)
    summary = _summarize(decisions, cost)
    robust_pass, robust_reason = phase07.robust_check(
        decisions,
        [],
        cost,
        rho_thresh=args.rho_thresh,
        persist_min=args.persist_min,
        eps_cost_frac=args.eps_cost_frac,
    )
    class_label = phase07.classify(summary["rho_A"] or 0.0, robust_pass)

    report: dict[str, Any] = {
        "entries": len(decisions),
        "window": args.window,
        "symbol": symbol or "",
        "summary": summary,
        "robust_pass": robust_pass,
        "robust_reason": robust_reason,
        "class_label": class_label,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Phase-07 asymmetry diagnostics")
    print(
        "entries={entries} window={window} support={support} activity_rate={activity_rate:.3f}".format(
            entries=report["entries"],
            window=report["window"],
            support=summary["n_support"],
            activity_rate=summary["activity_rate"],
        )
    )
    print(
        "rho_A={rho_A} sum_edge={sum_edge} sum_cost={sum_cost} robust={robust} ({reason})".format(
            rho_A=summary["rho_A"],
            sum_edge=summary["sum_edge"],
            sum_cost=summary["sum_cost"],
            robust="yes" if robust_pass else "no",
            reason=robust_reason,
        )
    )
    print(
        "median_dx={median_dx} median_cost={median_cost} cost_dominates_dx={cost_dominates_dx} class={class_label}".format(
            median_dx=summary["median_dx"],
            median_cost=summary["median_cost"],
            cost_dominates_dx=summary["cost_dominates_dx"],
            class_label=class_label,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-07 asymmetry diagnostics.")
    sub = ap.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="Report Phase-07 edge/cost diagnostics.")
    report.add_argument(
        "--decisions-ndjson",
        type=Path,
        required=True,
        help="Decision/action NDJSON log.",
    )
    report.add_argument("--symbol", type=str, default="", help="Filter symbol (optional).")
    report.add_argument("--window", type=int, default=256, help="Tail window size (0 for full file).")
    report.add_argument("--half-spread", type=float, default=0.0003)
    report.add_argument("--fee", type=float, default=0.0005)
    report.add_argument("--slip", type=float, default=0.0)
    report.add_argument("--rho-thresh", type=float, default=1.0)
    report.add_argument("--persist-min", type=int, default=25)
    report.add_argument("--eps-cost-frac", type=float, default=0.05)
    report.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    report.set_defaults(func=_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import phase07_eigen_boundary_check as phase07


def _load_decisions(path: Path, symbol: Optional[str], window: int) -> list[phase07.DecisionRec]:
    decisions = phase07.load_decisions_ndjson(path, symbol)
    if window > 0:
        decisions = decisions[-window:]
    return decisions


def _build_payload(
    target: str,
    symbol: Optional[str],
    decisions: list[phase07.DecisionRec],
    cost: phase07.CostModel,
    rho_thresh: float,
    persist_min: int,
    eps_cost_frac: float,
) -> dict[str, Any]:
    window = len(decisions)
    if window < 2:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "phase7_ready": False,
            "phase7_reason": "no_decisions",
            "phase7_metrics": {
                "rho_A": None,
                "sum_edge": None,
                "sum_cost": None,
                "n_support": 0,
                "activity_rate": 0.0,
                "robust_pass": False,
                "robust_reason": "no_decisions",
                "class_label": "false",
                "window": window,
                "count": 0,
            },
        }

    rho, sum_edge, sum_cost, n_support, activity_rate = phase07.rho_A_for_stream(
        decisions, [], cost
    )
    robust_pass, robust_reason = phase07.robust_check(
        decisions,
        [],
        cost,
        rho_thresh=rho_thresh,
        persist_min=persist_min,
        eps_cost_frac=eps_cost_frac,
    )
    class_label = phase07.classify(rho, robust_pass)

    if robust_pass:
        reason = "asymmetry_density_ok"
    elif robust_reason == "sparse_support":
        reason = "sparse_support"
    elif robust_reason == "below_thresh":
        reason = "rho_below_thresh"
    elif robust_reason == "cost_sensitivity":
        reason = "cost_sensitivity"
    else:
        reason = "asymmetry_density_blocked"

    metrics = {
        "rho_A": rho,
        "sum_edge": sum_edge,
        "sum_cost": sum_cost,
        "n_support": n_support,
        "activity_rate": activity_rate,
        "robust_pass": robust_pass,
        "robust_reason": robust_reason,
        "class_label": class_label,
        "rho_thresh": rho_thresh,
        "persist_min": persist_min,
        "eps_cost_frac": eps_cost_frac,
        "cost_model": {
            "half_spread": cost.half_spread,
            "fee": cost.fee,
            "slip": cost.slip,
        },
        "window": window,
        "count": n_support,
    }

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "phase7_ready": bool(robust_pass),
        "phase7_reason": reason,
        "phase7_metrics": metrics,
    }
    if symbol:
        payload["symbol"] = symbol
    return payload


def _append_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str)
        fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-07 asymmetry status emitter.")
    ap.add_argument(
        "--decisions-ndjson",
        type=Path,
        required=True,
        help="Decision/action NDJSON log (timestamp, symbol, direction, target_exposure).",
    )
    ap.add_argument("--symbol", type=str, default="", help="Filter symbol (optional).")
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
        help="Number of recent rows to read from the decisions log.",
    )

    ap.add_argument("--half-spread", type=float, default=0.0003)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0)
    ap.add_argument("--rho-thresh", type=float, default=1.0)
    ap.add_argument("--persist-min", type=int, default=25)
    ap.add_argument("--eps-cost-frac", type=float, default=0.05)

    args = ap.parse_args()

    symbol = args.symbol.strip() or None
    window = max(0, int(args.window))
    decisions = _load_decisions(args.decisions_ndjson, symbol, window)

    cost = phase07.CostModel(args.half_spread, args.fee, args.slip)
    payload = _build_payload(
        args.target,
        symbol,
        decisions,
        cost,
        rho_thresh=args.rho_thresh,
        persist_min=args.persist_min,
        eps_cost_frac=args.eps_cost_frac,
    )
    _append_line(args.phase7_log, payload)

    status = "ready" if payload["phase7_ready"] else "blocked"
    metrics = payload.get("phase7_metrics") or {}
    window_size = metrics.get("window")
    count = metrics.get("count")
    print(
        f"Wrote Phase-07 {status} line for {args.target} "
        f"(window={window_size}, count={count}) to {args.phase7_log}"
    )


if __name__ == "__main__":
    main()

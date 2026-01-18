#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import phase07_eigen_boundary_check as phase07


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row))
                fh.write("\n")
        return
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase-07 horizon sweep harness.")
    ap.add_argument("--decisions-ndjson", required=True, help="NDJSON decisions stream")
    ap.add_argument("--returns-ndjson", default="", help="Optional NDJSON returns/ohlc stream")
    ap.add_argument("--symbol", default="", help="Filter symbol (optional)")
    ap.add_argument("--half-spread", type=float, default=0.0003)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0)
    ap.add_argument("--rho-thresh", type=float, default=1.0)
    ap.add_argument("--persist-min", type=int, default=25)
    ap.add_argument("--eps-cost-frac", type=float, default=0.05)
    ap.add_argument("--horizons", default="10,30,60,120,300")
    ap.add_argument(
        "--out",
        default="logs/phase7/horizon_sweep.csv",
        help="Output CSV/JSONL path (default logs/phase7/horizon_sweep.csv).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.strip() or None
    decisions = phase07.load_decisions_ndjson(Path(args.decisions_ndjson), symbol)
    returns = []
    if args.returns_ndjson:
        returns = phase07.load_returns_ndjson(Path(args.returns_ndjson), symbol)

    if not decisions:
        print("No decisions loaded; nothing to sweep.")
        return 1

    cost = phase07.CostModel(args.half_spread, args.fee, args.slip)
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    if not horizons:
        print("No horizons provided; nothing to sweep.")
        return 1
    run_ts = datetime.utcnow().isoformat() + "Z"
    rows: list[dict[str, object]] = []

    for h in horizons:
        dec_h = phase07.decimate_by_horizon(decisions, h)
        rho, sum_edge, sum_cost, n_support, activity_rate = phase07.rho_A_for_stream(
            dec_h, returns, cost
        )
        robust_pass, robust_reason = phase07.robust_check(
            dec_h,
            returns,
            cost,
            rho_thresh=args.rho_thresh,
            persist_min=args.persist_min,
            eps_cost_frac=args.eps_cost_frac,
        )
        label = phase07.classify(rho, robust_pass)
        rows.append(
            {
                "timestamp": run_ts,
                "symbol": symbol or "",
                "horizon_s": h,
                "rho_A": rho,
                "sum_edge": sum_edge,
                "sum_cost": sum_cost,
                "n_support": n_support,
                "activity_rate": activity_rate,
                "robust_pass": robust_pass,
                "robust_reason": robust_reason,
                "class_label": label,
                "rho_thresh": args.rho_thresh,
                "persist_min": args.persist_min,
                "eps_cost_frac": args.eps_cost_frac,
                "half_spread": args.half_spread,
                "fee": args.fee,
                "slip": args.slip,
            }
        )

    out_path = Path(args.out)
    _write_rows(out_path, rows)
    best = max(rows, key=lambda r: float(r["rho_A"]))
    print(
        f"Wrote {len(rows)} horizons to {out_path} (best horizon={best['horizon_s']}s rho_A={best['rho_A']:.6f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

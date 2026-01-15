#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def _load_prices(path: Path) -> list[float]:
    df = pd.read_csv(path)
    for col in ("close", "Close", "Zamkniecie", "Zamknie"):
        if col in df.columns:
            return df[col].astype(float).tolist()
    raise SystemExit(f"Cannot find a close column in {path}")


def _parse_size_map(raw: str | None) -> dict[str, float]:
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--size-map must be valid JSON: {exc}") from exc
    return {str(k): float(v) for k, v in payload.items()}


def _serialize_rows(rows: Iterable[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str))
            fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-5 execution simulator.")
    ap.add_argument("--proposal-log", type=Path, required=True)
    ap.add_argument("--prices-csv", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=Path("logs/phase5"))
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--fee-bps", type=float, default=0.5)
    ap.add_argument("--size-scale", type=float, default=1.0)
    ap.add_argument(
        "--size-map",
        type=str,
        default=None,
        help="JSON map (string keys) from size_pred to explicit size level.",
    )
    ap.add_argument(
        "--size-default",
        type=float,
        default=1.0,
        help="Fallback size level if parsing fails and size_map has no entry.",
    )
    args = ap.parse_args()

    if not args.proposal_log.exists():
        raise SystemExit(f"Missing proposal log: {args.proposal_log}")
    if not args.prices_csv.exists():
        raise SystemExit(f"Missing prices CSV: {args.prices_csv}")

    prices = _load_prices(args.prices_csv)
    df = pd.read_csv(args.proposal_log)
    required = {"i", "action", "size_pred", "ts"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise SystemExit(f"proposal log missing columns: {missing}")

    size_map = _parse_size_map(args.size_map)
    slip_ratio = args.slip_bps * 1e-4
    fee_ratio = args.fee_bps * 1e-4
    rows: list[dict[str, object]] = []
    skipped = 0
    for _, row in df.iterrows():
        action = int(row["action"])
        if action == 0:
            continue
        idx = int(row["i"])
        if idx < 0 or idx + args.horizon >= len(prices):
            skipped += 1
            continue
        side = float(action / abs(action))
        price_t = float(prices[idx])
        price_h = float(prices[idx + args.horizon])
        size_key = str(row["size_pred"])
        if size_key in size_map:
            level = size_map[size_key]
        else:
            try:
                level = float(size_key)
            except (ValueError, TypeError):
                level = args.size_default
        size = level * args.size_scale
        fill_price = price_t * (1.0 + side * slip_ratio)
        slippage_cost = abs(size * (fill_price - price_t))
        execution_cost = size * price_t * fee_ratio
        raw_pnl = size * side * (price_h - fill_price)
        realized_pnl = raw_pnl - execution_cost - slippage_cost
        rows.append(
            {
                "i": idx,
                "ts": str(row["ts"]),
                "action": action,
                "direction": side,
                "size_pred": row["size_pred"],
                "size_level": level,
                "size": size,
                "price_t": price_t,
                "price_t_h": price_h,
                "fill_price": fill_price,
                "slip_bps": args.slip_bps,
                "fee_bps": args.fee_bps,
                "slippage_cost": slippage_cost,
                "execution_cost": execution_cost,
                "realized_pnl": realized_pnl,
                "horizon": args.horizon,
            }
        )

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = args.log_dir / f"phase5_execution_{stamp}.jsonl"
    _serialize_rows(rows, out_path)
    total_pnl = sum(r["realized_pnl"] for r in rows)
    mean_pnl = total_pnl / len(rows) if rows else 0.0
    print(
        f"Wrote {len(rows)} executions (skipped {skipped}) to {out_path} "
        f"(mean pnl {mean_pnl:.6f}, total pnl {total_pnl:.6f})"
    )


if __name__ == "__main__":
    main()

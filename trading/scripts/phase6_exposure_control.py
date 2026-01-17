#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def summarize_phase5_logs(log_dir: Path, slip_threshold: float):
    summaries = []
    for path in sorted(log_dir.glob("phase5_execution_*.jsonl")):
        rows = list(_load_jsonl(path))
        if not rows:
            continue
        total_exposure = sum(abs(r.get("size", 0.0) * r.get("price_t", 0.0)) for r in rows)
        total_pnl = sum(r.get("realized_pnl", 0.0) for r in rows)
        me = total_pnl / len(rows) if rows else 0.0
        slip = float(rows[0].get("slip_bps", 0.0))
        summaries.append(
            {
                "log": path.name,
                "entries": len(rows),
                "slip_bps": slip,
                "allowed": slip <= slip_threshold,
                "total_exposure": total_exposure,
                "mean_pnl": me,
            }
        )
    return summaries


def serialize(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Phase-6 exposure summaries with friction guard.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/phase5/es"))
    parser.add_argument("--slip-threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/phase6"))
    args = parser.parse_args()

    summaries = summarize_phase5_logs(args.log_dir, args.slip_threshold)
    if not summaries:
        raise SystemExit("No Phase-5 logs found; nothing to summarize.")

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"capital_controls_{stamp}.jsonl"
    serialize(summaries, out_path)
    print(f"Wrote {len(summaries)} capital control entries to {out_path}")
    for row in summaries:
        status = "allowed" if row["allowed"] else "clamped"
        print(f"  {row['log']}: slip={row['slip_bps']}bps → {status}, exposure={row['total_exposure']:.2f}, mean_pnl={row['mean_pnl']:.6f}")


if __name__ == "__main__":
    main()

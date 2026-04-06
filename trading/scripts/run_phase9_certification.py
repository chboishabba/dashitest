#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nashi.certification import (  # noqa: E402
    attribution_breakdown,
    certification_census,
    load_certification_frame,
    summarize_attribution,
    write_census_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Nashi step CSV or DuckDB path")
    parser.add_argument("--step-table", default="nashi_steps", help="DuckDB table name when --input is a DuckDB file")
    parser.add_argument("--window-rows", type=int, default=256, help="Rows per certification window")
    parser.add_argument("--window-seconds", type=float, default=0.0, help="Optional time window in seconds; overrides --window-rows when > 0")
    parser.add_argument("--output-csv", help="Optional path for per-window certification CSV")
    parser.add_argument("--summary-json", help="Optional path for aggregate certification summary JSON")
    parser.add_argument("--attribution-csv", help="Optional path for attribution CSV, including hazard-conditioned breakdowns when present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    frame = load_certification_frame(input_path, step_table=args.step_table)
    window_ms = int(args.window_seconds * 1000.0) if args.window_seconds and args.window_seconds > 0.0 else None
    window_rows = None if window_ms is not None else args.window_rows
    census = certification_census(frame, window_rows=window_rows, window_ms=window_ms)
    attribution = attribution_breakdown(frame)
    summary = write_census_outputs(
        census,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        summary_json=Path(args.summary_json) if args.summary_json else None,
    )
    if args.attribution_csv:
        output_path = Path(args.attribution_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        attribution.to_csv(output_path, index=False)
    if not attribution.empty:
        summary["drag_attribution"] = summarize_attribution(attribution, top_k=5)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nashi.invariants import (
    check_artifact_parity,
    duckdb_has_table,
    load_duckdb_family_rows,
    check_step_rows,
    load_duckdb_step_rows,
    load_ndjson_rows,
    load_step_rows,
    summarize_parity_violations,
    summarize_violations,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check executable invariants on Nashi step outputs.")
    ap.add_argument(
        "--step-log",
        type=Path,
        action="append",
        required=True,
        help="Nashi per-step CSV log(s) emitted by nashi/runtime.py",
    )
    ap.add_argument(
        "--microstructure-survival-floor",
        type=float,
        default=None,
        help="Override the expected cost-survival floor used for phase9 microstructure checks.",
    )
    ap.add_argument(
        "--show-limit",
        type=int,
        default=10,
        help="Maximum number of detailed violations to print.",
    )
    ap.add_argument("--duckdb", type=Path, default=None, help="Optional DuckDB artifact for nashi_steps parity.")
    ap.add_argument(
        "--decision-ndjson",
        type=Path,
        default=None,
        help="Optional decision NDJSON artifact for row-count and surface parity.",
    )
    ap.add_argument(
        "--ohlc-ndjson",
        type=Path,
        default=None,
        help="Optional OHLC NDJSON artifact for row-count and surface parity.",
    )
    args = ap.parse_args()

    violations = []
    checked_rows = 0
    all_rows = []
    for path in args.step_log:
        rows = load_step_rows(path)
        checked_rows += len(rows)
        all_rows.extend(rows)
        violations.extend(
            check_step_rows(
                rows,
                microstructure_survival_floor=args.microstructure_survival_floor,
            )
        )

    summary = summarize_violations(violations)
    print(f"checked_rows={checked_rows}")
    print(f"violation_count={len(violations)}")
    print(f"violation_summary={summary}")

    parity_violations = check_artifact_parity(
        all_rows,
        duckdb_rows=load_duckdb_step_rows(args.duckdb) if args.duckdb is not None else None,
        family_rows=(
            load_duckdb_family_rows(args.duckdb)
            if args.duckdb is not None and duckdb_has_table(args.duckdb, table="nashi_family_certifications")
            else None
        ),
        decision_rows=load_ndjson_rows(args.decision_ndjson) if args.decision_ndjson is not None else None,
        ohlc_rows=load_ndjson_rows(args.ohlc_ndjson) if args.ohlc_ndjson is not None else None,
    )
    parity_summary = summarize_parity_violations(parity_violations)
    print(f"parity_violation_count={len(parity_violations)}")
    print(f"parity_violation_summary={parity_summary}")

    if violations:
        print("sample_violations:")
        for violation in violations[: max(0, int(args.show_limit))]:
            print(
                f"  rule={violation.rule} symbol={violation.symbol or '-'} "
                f"ts={violation.ts if violation.ts is not None else '-'} "
                f"row={violation.row_index} detail={violation.detail}"
            )
        return 1
    if parity_violations:
        print("sample_parity_violations:")
        for violation in parity_violations[: max(0, int(args.show_limit))]:
            print(f"  rule={violation.rule} detail={violation.detail}")
        return 1

    print("nashi_invariants=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

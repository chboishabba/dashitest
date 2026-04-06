#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nashi.certification import certification_census, load_certification_frame, summarize_census  # noqa: E402


def main() -> None:
    input_path = ROOT / "logs" / "nashi" / "nashi_family_smoke.csv"
    frame = load_certification_frame(input_path)
    census = certification_census(frame, window_rows=64)
    summary = summarize_census(census)

    assert not census.empty
    required_columns = {
        "symbol",
        "window_id",
        "trade_certified_count",
        "preserve_certified_count",
        "ban_required_count",
        "ban_correct_count",
        "ban_missed_count",
        "ban_coverage",
    }
    assert required_columns.issubset(census.columns), census.columns.tolist()
    assert summary["window_count"] == len(census)
    assert summary["ban_required_steps"] >= summary["ban_correct_steps"]
    assert (census["ban_missed_count"] >= 0).all()
    print("nashi certification smoke: ok")


if __name__ == "__main__":
    main()

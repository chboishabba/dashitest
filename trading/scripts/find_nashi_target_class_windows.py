#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nashi.phase9 import CapitalParams  # noqa: E402
from nashi.runtime import NashiArtifacts, default_bars, run_nashi_bars  # noqa: E402
from scripts.mine_nashi_recovery_windows import _episode_summary  # noqa: E402
from scripts.scan_nashi_warning_pairs import _classify_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="Optional source CSV. Defaults to cached BTC/Stooq data.")
    parser.add_argument("--window-size", type=int, default=4000, help="Rows per sampled source window.")
    parser.add_argument("--stride", type=int, default=1000, help="Stride between sampled windows.")
    parser.add_argument("--offset-start", type=int, default=0, help="Starting row offset into the source bars.")
    parser.add_argument("--offset-stop", type=int, default=None, help="Optional exclusive upper bound for window starts.")
    parser.add_argument("--max-windows", type=int, default=12, help="Maximum number of windows to evaluate.")
    parser.add_argument("--top-k", type=int, default=12, help="Maximum ranked episodes to inspect per window.")
    parser.add_argument("--context-rows", type=int, default=2, help="Context rows per selected episode.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol label for generated runs.")
    parser.add_argument("--default-spread-bps", type=float, default=2.0, help="Synthetic spread when quotes are absent.")
    parser.add_argument("--base-size", type=float, default=1.0, help="Maximum target exposure magnitude.")
    parser.add_argument("--contextual-hazard-csv", type=Path, default=None, help="Optional offline contextual hazard CSV.")
    parser.add_argument("--progress-every", type=int, default=0, help="Emit run progress every N bars. Use 0 to disable.")
    parser.add_argument("--target-class", action="append", default=["confirmed_collapse", "recovering_after_warning"], help="Warning/response class to preserve; repeatable.")
    parser.add_argument("--tmp-prefix", default="/tmp/nashi_target_class_scan/base", help="Output stem for generated artifacts.")
    parser.add_argument("--keep-all-artifacts", action="store_true", help="Keep artifacts for non-matching windows.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV path for per-episode results.")
    parser.add_argument("--summary-out", type=Path, help="Optional JSON path for aggregate scan summary.")
    return parser.parse_args()


def _sample_starts(total_rows: int, *, window_size: int, stride: int, offset_start: int, offset_stop: int | None, max_windows: int) -> list[int]:
    if total_rows <= 0 or window_size <= 0 or max_windows <= 0:
        return []
    last_start = max(total_rows - window_size, 0)
    stop = last_start if offset_stop is None else min(max(offset_stop - 1, 0), last_start)
    starts: list[int] = []
    current = max(offset_start, 0)
    while current <= stop and len(starts) < max_windows:
        starts.append(current)
        current += max(stride, 1)
    return starts


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


def _prune_artifacts(artifacts: NashiArtifacts, pair_path: Path) -> None:
    for path in [
        artifacts.step_log_path,
        artifacts.decision_ndjson_path,
        artifacts.ohlc_ndjson_path,
        artifacts.duckdb_path,
        artifacts.duckdb_path.with_suffix(artifacts.duckdb_path.suffix + ".wal"),
        artifacts.family_csv_path,
        artifacts.family_ndjson_path,
        pair_path,
    ]:
        _safe_unlink(path)


def main() -> None:
    args = parse_args()
    bars, source_label = default_bars(args.csv, default_spread_bps=args.default_spread_bps)
    starts = _sample_starts(
        len(bars),
        window_size=int(args.window_size),
        stride=int(args.stride),
        offset_start=int(args.offset_start),
        offset_stop=args.offset_stop,
        max_windows=int(args.max_windows),
    )
    if not starts:
        raise SystemExit("no source windows selected")

    target_classes = {str(value) for value in args.target_class or []}
    phase9_params = CapitalParams()
    prefix_root = Path(args.tmp_prefix)
    prefix_root.parent.mkdir(parents=True, exist_ok=True)

    window_records: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []

    for window_idx, start in enumerate(starts):
        stop = min(start + int(args.window_size), len(bars))
        window_bars = bars.iloc[start:stop].copy()
        run_stem = prefix_root.with_name(f"{prefix_root.name}_{window_idx:02d}_{start}_{stop}")
        artifacts = NashiArtifacts(
            step_log_path=run_stem.with_suffix(".csv"),
            decision_ndjson_path=run_stem.with_name(f"{run_stem.name}_decisions.ndjson"),
            ohlc_ndjson_path=run_stem.with_name(f"{run_stem.name}_ohlc.ndjson"),
            duckdb_path=run_stem.with_suffix(".duckdb"),
            family_csv_path=run_stem.with_name(f"{run_stem.name}_family.csv"),
            family_ndjson_path=run_stem.with_name(f"{run_stem.name}_family.ndjson"),
        )

        def progress_fn(done: int, total: int, elapsed_s: float) -> None:
            if args.progress_every <= 0:
                return
            if done != total and done % args.progress_every != 0:
                return
            rate = done / elapsed_s if elapsed_s > 0 else 0.0
            print(f"[target-class] window={window_idx} rows={done}/{total} rate={rate:.1f} bars/s")

        frame = run_nashi_bars(
            window_bars,
            symbol=args.symbol,
            artifacts=artifacts,
            source_label=f"{source_label}[{start}:{stop}]",
            base_size=float(args.base_size),
            phase9_params=phase9_params,
            default_spread_bps=float(args.default_spread_bps),
            contextual_hazard_csv=args.contextual_hazard_csv,
            progress_fn=progress_fn,
        )
        pair_summary = _episode_summary(frame, top_k=int(args.top_k), context_rows=int(args.context_rows))
        pair_summary.update(
            {
                "input": str(artifacts.step_log_path),
                "source_label": f"{source_label}[{start}:{stop}]",
                "window_index": int(window_idx),
                "start_row": int(start),
                "stop_row": int(stop),
            }
        )
        pair_path = run_stem.with_name(f"{run_stem.name}_pairs.json")
        pair_path.write_text(json.dumps(pair_summary, indent=2, sort_keys=True), encoding="utf-8")

        window_match_count = 0
        for episode in pair_summary.get("selected", []):
            rows = pd.DataFrame(episode.get("rows", []))
            row_pair = _classify_episode(rows)
            klass = str(row_pair.get("warning_response_class", ""))
            if klass not in target_classes:
                continue
            window_match_count += 1
            match_records.append(
                {
                    "window_index": int(window_idx),
                    "start_row": int(start),
                    "stop_row": int(stop),
                    "pair_summary": str(pair_path),
                    "step_log": str(artifacts.step_log_path),
                    "episode_id": int(episode.get("episode_id", -1)),
                    "duration_rows": int(episode.get("duration_rows", 0)),
                    "warning_response_class": klass,
                    "warning_response_reason": str(row_pair.get("warning_response_reason", "")),
                    "entry_t": row_pair.get("entry_t"),
                    "warning_t": row_pair.get("warning_t"),
                    "response_t": row_pair.get("response_t"),
                    "realized_surplus_sum": float(episode.get("realized_surplus_sum", 0.0)),
                    "executed_expected_surplus_sum": float(episode.get("executed_expected_surplus_sum", 0.0)),
                    "entry_reason": str(episode.get("entry_reason", "")),
                    "exit_reason": str(episode.get("exit_reason", "")),
                }
            )

        window_records.append(
            {
                "window_index": int(window_idx),
                "start_row": int(start),
                "stop_row": int(stop),
                "pair_summary": str(pair_path),
                "step_log": str(artifacts.step_log_path),
                "episode_count": int(pair_summary.get("episode_count", 0)),
                "selected_episode_count": int(pair_summary.get("selected_episode_count", 0)),
                "matching_episode_count": int(window_match_count),
            }
        )

        if window_match_count == 0 and not args.keep_all_artifacts:
            _prune_artifacts(artifacts, pair_path)

    summary = {
        "source_label": source_label,
        "window_count": int(len(window_records)),
        "target_classes": sorted(target_classes),
        "match_count": int(len(match_records)),
        "matched_windows": int(sum(1 for row in window_records if row["matching_episode_count"] > 0)),
        "windows": window_records,
        "matches": match_records,
    }

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(match_records).to_csv(args.output_csv, index=False)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

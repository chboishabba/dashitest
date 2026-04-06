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
from scripts.debug_nashi_trade_pairs import (  # noqa: E402
    _episode_axis_summary,
    _episode_rows,
    _find_trade_episodes,
    _forensic_trace_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="Optional source CSV. Defaults to cached BTC/Stooq data.")
    parser.add_argument("--window-size", type=int, default=800, help="Rows per sampled source window.")
    parser.add_argument("--stride", type=int, default=2000, help="Stride between sampled source windows.")
    parser.add_argument("--max-windows", type=int, default=12, help="Maximum number of source windows to evaluate.")
    parser.add_argument("--offset-start", type=int, default=0, help="Starting row offset into the source bars.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol label for generated runs.")
    parser.add_argument("--default-spread-bps", type=float, default=2.0, help="Synthetic spread when quotes are absent.")
    parser.add_argument("--base-size", type=float, default=1.0, help="Maximum target exposure magnitude.")
    parser.add_argument("--contextual-hazard-csv", type=Path, default=None, help="Optional offline contextual hazard CSV.")
    parser.add_argument("--progress-every", type=int, default=0, help="Emit run progress every N bars. Use 0 to disable.")
    parser.add_argument("--top-k", type=int, default=12, help="Maximum ranked episodes to keep per window summary.")
    parser.add_argument("--context-rows", type=int, default=2, help="Context rows per selected episode.")
    parser.add_argument("--tmp-prefix", default="/tmp/nashi_recovery_mine", help="Output stem for generated artifacts.")
    parser.add_argument("--summary-out", type=Path, help="Optional JSON path for aggregate mining summary.")
    parser.add_argument("--pair-glob-out", type=Path, help="Optional text file containing one generated pair-summary path per line.")
    return parser.parse_args()


def _sample_starts(total_rows: int, *, window_size: int, stride: int, max_windows: int, offset_start: int) -> list[int]:
    if total_rows <= 0 or window_size <= 0 or max_windows <= 0:
        return []
    last_start = max(total_rows - window_size, 0)
    starts: list[int] = []
    current = max(offset_start, 0)
    while current <= last_start and len(starts) < max_windows:
        starts.append(current)
        current += max(stride, 1)
    if last_start not in starts and len(starts) < max_windows:
        starts.append(last_start)
    return starts


def _episode_summary(
    frame: pd.DataFrame,
    *,
    top_k: int,
    context_rows: int,
    exposure_eps: float = 1e-9,
) -> dict[str, Any]:
    ranked = _find_trade_episodes(frame, exposure_eps=exposure_eps)
    if ranked.empty:
        return {"episode_count": 0, "selected_episode_count": 0, "selected": []}

    selected = ranked.head(int(top_k)).copy()
    selected_json: list[dict[str, Any]] = []
    for _, episode in selected.iterrows():
        rows = _episode_rows(
            frame,
            symbol=str(episode["symbol"]),
            t_open=int(episode["t_open"]),
            t_close=int(episode["t_close"]),
            context_rows=int(context_rows),
        )
        continuation_summary: dict[str, Any] = {}
        if not rows.empty:
            rows, continuation_summary = _forensic_trace_analysis(rows, exposure_eps=exposure_eps)
        selected_json.append(
            {
                key: (float(value) if isinstance(value, float) else int(value) if isinstance(value, int) else value)
                for key, value in episode.to_dict().items()
            }
            | {
                "axis_summary": _episode_axis_summary(rows),
                "continuation_forensics": continuation_summary,
                "rows": rows.to_dict(orient="records"),
            }
        )

    return {
        "episode_count": int(len(ranked)),
        "selected_episode_count": int(len(selected)),
        "worst_episode_realized_surplus_sum": float(selected["realized_surplus_sum"].min()),
        "worst_episode_drag_sum": float(selected["drag_sum"].max()),
        "selected": selected_json,
    }


def main() -> None:
    args = parse_args()
    bars, source_label = default_bars(args.csv, default_spread_bps=args.default_spread_bps)
    starts = _sample_starts(
        len(bars),
        window_size=int(args.window_size),
        stride=int(args.stride),
        max_windows=int(args.max_windows),
        offset_start=int(args.offset_start),
    )
    if not starts:
        raise SystemExit("no source windows selected")

    phase9_params = CapitalParams()
    summary_records: list[dict[str, Any]] = []
    pair_paths: list[str] = []
    prefix_root = Path(args.tmp_prefix)
    prefix_root.parent.mkdir(parents=True, exist_ok=True)

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
            print(f"[mine] window={window_idx} rows={done}/{total} rate={rate:.1f} bars/s")

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

        pair_summary = _episode_summary(
            frame,
            top_k=int(args.top_k),
            context_rows=int(args.context_rows),
        )
        pair_summary.update(
            {
                "input": str(artifacts.step_log_path),
                "source_label": f"{source_label}[{start}:{stop}]",
                "window_index": window_idx,
                "start_row": int(start),
                "stop_row": int(stop),
            }
        )
        pair_path = run_stem.with_name(f"{run_stem.name}_pairs.json")
        pair_path.write_text(json.dumps(pair_summary, indent=2, sort_keys=True), encoding="utf-8")
        pair_paths.append(str(pair_path))
        summary_records.append(
            {
                "window_index": int(window_idx),
                "start_row": int(start),
                "stop_row": int(stop),
                "step_log": str(artifacts.step_log_path),
                "pair_summary": str(pair_path),
                "episode_count": int(pair_summary["episode_count"]),
                "selected_episode_count": int(pair_summary["selected_episode_count"]),
                "worst_episode_realized_surplus_sum": float(pair_summary.get("worst_episode_realized_surplus_sum", 0.0)),
            }
        )

    summary = {
        "source_label": source_label,
        "window_count": int(len(summary_records)),
        "windows_with_episodes": int(sum(1 for row in summary_records if row["episode_count"] > 0)),
        "records": summary_records,
        "pair_summaries": pair_paths,
    }

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.pair_glob_out:
        args.pair_glob_out.parent.mkdir(parents=True, exist_ok=True)
        args.pair_glob_out.write_text("\n".join(pair_paths) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

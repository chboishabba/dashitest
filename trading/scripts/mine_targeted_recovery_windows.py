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
from scripts.mine_nashi_recovery_windows import _episode_summary, _sample_starts  # noqa: E402
from scripts.scan_nashi_warning_pairs import _classify_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="Optional source CSV. Defaults to cached BTC/Stooq data.")
    parser.add_argument("--window-size", type=int, default=800, help="Rows per sampled source window.")
    parser.add_argument("--stride", type=int, default=1000, help="Stride between sampled source windows.")
    parser.add_argument("--max-windows", type=int, default=12, help="Maximum number of source windows to evaluate.")
    parser.add_argument("--offset-start", type=int, default=0, help="Starting row offset into the source bars.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol label for generated runs.")
    parser.add_argument("--default-spread-bps", type=float, default=2.0, help="Synthetic spread when quotes are absent.")
    parser.add_argument("--base-size", type=float, default=1.0, help="Maximum target exposure magnitude.")
    parser.add_argument("--contextual-hazard-csv", type=Path, default=None, help="Optional offline contextual hazard CSV.")
    parser.add_argument("--progress-every", type=int, default=0, help="Emit run progress every N bars. Use 0 to disable.")
    parser.add_argument("--top-k", type=int, default=12, help="Maximum ranked episodes to keep per window summary.")
    parser.add_argument("--context-rows", type=int, default=2, help="Context rows per selected episode.")
    parser.add_argument("--tmp-prefix", default="/tmp/nashi_targeted_recovery", help="Output stem for generated artifacts.")
    parser.add_argument("--min-continuation-opportunity", type=int, default=3, help="Minimum continuation-opportunity rows for target windows.")
    parser.add_argument("--max-flatten-ratio", type=float, default=0.8, help="Maximum flatten ratio for target windows.")
    parser.add_argument("--summary-out", type=Path, help="Optional JSON path for aggregate mining summary.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV path for per-window targeted stats.")
    parser.add_argument("--pair-glob-out", type=Path, help="Optional text file containing one generated pair-summary path per line.")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _window_stats(pair_summary: dict[str, Any]) -> dict[str, Any]:
    episodes = list(pair_summary.get("selected") or [])
    all_rows = pd.DataFrame([row for episode in episodes for row in episode.get("rows", [])])

    warning_count = 0
    nonterminal_warning_count = 0
    immediate_flatten_count = 0
    confirmed_collapse_count = 0
    recovering_after_warning_count = 0
    unresolved_warning_count = 0
    uncertain_pair_count = 0

    for episode in episodes:
        rows = pd.DataFrame(episode.get("rows", []))
        pair = _classify_episode(rows)
        klass = str(pair.get("warning_response_class", ""))
        if klass not in {"empty_episode", "insufficient_prefix"}:
            warning_count += 1
        if klass == "recovering_after_warning":
            recovering_after_warning_count += 1
        elif klass == "immediate_flatten":
            immediate_flatten_count += 1
        elif klass == "confirmed_collapse":
            confirmed_collapse_count += 1
        elif klass == "unresolved_warning":
            unresolved_warning_count += 1
        elif klass == "uncertain_pair":
            uncertain_pair_count += 1

        episode_rows = rows[rows.get("in_episode", 0).astype(bool)].reset_index(drop=True)
        if len(episode_rows) < 3:
            continue
        warning = episode_rows.iloc[1]
        response = episode_rows.iloc[2]
        warning_nonterminal = (
            str(warning.get("family", "")) != "flatten_transition"
            and abs(_safe_float(warning.get("exposure_post", warning.get("exposure")))) > 1e-9
            and str(response.get("family", "")) != "flatten_transition"
            and abs(_safe_float(response.get("exposure_post", response.get("exposure")))) > 1e-9
        )
        if warning_nonterminal:
            nonterminal_warning_count += 1

    if all_rows.empty:
        mean_abs_edge = 0.0
        mean_edge_persistence = 0.0
        mean_edge_shock = 0.0
        mean_actionability = 0.0
        high_microstructure_share = 0.0
        mean_hazard = 0.0
        carry_decay_share = 0.0
        continuation_opportunity = 0
        drag_ratio = 0.0
    else:
        in_episode = all_rows[all_rows.get("in_episode", 0).astype(bool)].copy()
        expected = pd.to_numeric(in_episode.get("executed_expected_surplus", 0.0), errors="coerce").fillna(0.0)
        exposure_post = pd.to_numeric(in_episode.get("exposure_post", in_episode.get("exposure", 0.0)), errors="coerce").fillna(0.0)
        cost_survival = pd.to_numeric(in_episode.get("cost_survival_ratio", 0.0), errors="coerce").fillna(0.0)
        micro = pd.to_numeric(in_episode.get("microstructure_pressure", 0.0), errors="coerce").fillna(0.0)
        edge = pd.to_numeric(in_episode.get("edge", 0.0), errors="coerce").fillna(0.0)
        persistence = pd.to_numeric(in_episode.get("edge_persistence", 0.0), errors="coerce").fillna(0.0)
        shock = pd.to_numeric(in_episode.get("edge_shock", 0.0), errors="coerce").fillna(0.0)
        actionability = pd.to_numeric(in_episode.get("actionability", 0.0), errors="coerce").fillna(0.0)
        hazard = pd.to_numeric(in_episode.get("hazard", 0.0), errors="coerce").fillna(0.0)
        realized = pd.to_numeric(in_episode.get("realized_surplus", 0.0), errors="coerce").fillna(0.0)

        continuation_mask = (exposure_post.abs() > 1e-9) & (expected > 1e-9)
        fallback_mask = (exposure_post.abs() > 1e-9) & (edge.abs() > 1e-3) & (cost_survival > 20.0)
        continuation_opportunity = int((continuation_mask | fallback_mask).sum())
        mean_abs_edge = float(edge.abs().mean()) if not edge.empty else 0.0
        mean_edge_persistence = float(persistence.mean()) if not persistence.empty else 0.0
        mean_edge_shock = float(shock.mean()) if not shock.empty else 0.0
        mean_actionability = float(actionability.mean()) if not actionability.empty else 0.0
        high_microstructure_share = float(((micro >= 0.65) | (in_episode.get("nashi_spread_regime", "") == "microstructure_kills_edge")).mean()) if len(in_episode) else 0.0
        mean_hazard = float(hazard.mean()) if not hazard.empty else 0.0
        carry_decay_share = float((expected <= 1e-9).mean()) if len(expected) else 0.0
        expected_sum = float(expected.sum())
        realized_sum = float(realized.sum())
        drag_ratio = max(expected_sum - realized_sum, 0.0) / max(abs(expected_sum), 1e-9)

    episode_count = int(pair_summary.get("episode_count", 0))
    durations = [_safe_float(ep.get("duration_rows"), 0.0) for ep in episodes]
    max_hold_length = int(max(durations, default=0.0))
    mean_hold_length = float(sum(durations) / len(durations)) if durations else 0.0
    flatten_ratio = immediate_flatten_count / max(warning_count, 1)
    recovery_candidate_score = recovering_after_warning_count / max(nonterminal_warning_count, 1)

    return {
        "episode_count": episode_count,
        "selected_episode_count": int(pair_summary.get("selected_episode_count", 0)),
        "warning_count": int(warning_count),
        "nonterminal_warning_count": int(nonterminal_warning_count),
        "immediate_flatten_count": int(immediate_flatten_count),
        "confirmed_collapse_count": int(confirmed_collapse_count),
        "recovering_after_warning_count": int(recovering_after_warning_count),
        "unresolved_warning_count": int(unresolved_warning_count),
        "uncertain_pair_count": int(uncertain_pair_count),
        "max_hold_length": int(max_hold_length),
        "mean_hold_length": float(mean_hold_length),
        "continuation_opportunity": int(continuation_opportunity),
        "flatten_ratio": float(flatten_ratio),
        "recovery_candidate_score": float(recovery_candidate_score),
        "mean_abs_edge": float(mean_abs_edge),
        "mean_edge_persistence": float(mean_edge_persistence),
        "mean_edge_shock": float(mean_edge_shock),
        "mean_actionability": float(mean_actionability),
        "high_microstructure_share": float(high_microstructure_share),
        "mean_hazard": float(mean_hazard),
        "carry_decay_share": float(carry_decay_share),
        "drag_ratio": float(drag_ratio),
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
    prefix_root = Path(args.tmp_prefix)
    prefix_root.parent.mkdir(parents=True, exist_ok=True)
    summary_records: list[dict[str, Any]] = []
    pair_paths: list[str] = []

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
            print(f"[targeted-mine] window={window_idx} rows={done}/{total} rate={rate:.1f} bars/s")

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
        pair_paths.append(str(pair_path))

        stats = _window_stats(pair_summary)
        targeted_keep = (
            stats["continuation_opportunity"] >= int(args.min_continuation_opportunity)
            and stats["nonterminal_warning_count"] >= 1
            and stats["flatten_ratio"] < float(args.max_flatten_ratio)
            and stats["recovery_candidate_score"] > 0.0
        )
        summary_records.append(
            {
                "window_index": int(window_idx),
                "start_row": int(start),
                "stop_row": int(stop),
                "source_label": f"{source_label}[{start}:{stop}]",
                "step_log": str(artifacts.step_log_path),
                "pair_summary": str(pair_path),
                "targeted_keep": bool(targeted_keep),
                **stats,
            }
        )

    frame = pd.DataFrame(summary_records).sort_values(
        ["targeted_keep", "recovery_candidate_score", "nonterminal_warning_count", "continuation_opportunity", "flatten_ratio"],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    summary = {
        "source_label": source_label,
        "window_count": int(len(frame)),
        "windows_with_episodes": int((frame["episode_count"] > 0).sum()) if not frame.empty else 0,
        "targeted_window_count": int(frame["targeted_keep"].sum()) if not frame.empty else 0,
        "recovery_positive_window_count": int((frame["recovering_after_warning_count"] > 0).sum()) if not frame.empty else 0,
        "records": frame.to_dict(orient="records"),
        "pair_summaries": pair_paths,
    }

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_csv, index=False)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.pair_glob_out:
        args.pair_glob_out.parent.mkdir(parents=True, exist_ok=True)
        args.pair_glob_out.write_text("\n".join(pair_paths) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_downloader import ensure_dir, fetch_binance_depth_snapshot, stream_binance_trades  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a live Binance trade window, run Nashi on it, and emit forensic artifacts."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol to capture.")
    parser.add_argument("--duration-seconds", type=float, default=120.0, help="Trade capture duration.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between Binance aggTrade polls.")
    parser.add_argument("--chunk-size-seconds", type=float, default=None, help="Optional chunk size; defaults to full capture duration.")
    parser.add_argument("--out-prefix", default="/tmp/nashi_live_capture", help="Output stem for generated artifacts.")
    parser.add_argument("--depth-ndjson", type=Path, default=None, help="Optional Binance depth NDJSON to join against.")
    parser.add_argument("--capture-depth", action="store_true", help="Collect Binance depth snapshots in the same script while live trade windows run.")
    parser.add_argument("--depth-limit", type=int, default=100, choices=[5, 10, 20, 50, 100, 500, 1000, 5000], help="Depth levels to capture when --capture-depth is enabled.")
    parser.add_argument("--depth-flush-every", type=int, default=1, help="Buffered depth snapshots before flush when --capture-depth is enabled.")
    parser.add_argument("--timestamp-tolerance", default="2s", help="Tolerance for L2 join if depth is provided.")
    parser.add_argument("--top-k", type=int, default=8, help="Episode count for trade-pair debugging.")
    parser.add_argument("--context-rows", type=int, default=2, help="Context rows for trade-pair debugging.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for subprocess steps.")
    parser.add_argument("--repeat-until-episode", action="store_true", help="Repeat captures until at least one trade episode appears.")
    parser.add_argument(
        "--require-l2-match",
        action="store_true",
        help="When depth is available, only stop once an attempt has at least one matched L2 warning/response pair.",
    )
    parser.add_argument("--max-attempts", type=int, default=10, help="Maximum capture attempts when repeat mode is enabled. Use 0 for unbounded retry.")
    parser.add_argument(
        "--target-l2-distinct-pairs",
        type=int,
        default=1,
        help="When L2 matching is active, stop only after accumulating at least this many distinct-snapshot warning/response pairs across attempts.",
    )
    parser.add_argument("--sleep-between-attempts", type=float, default=1.0, help="Seconds to sleep between repeated capture attempts.")
    parser.add_argument("--cleanup-misses", action="store_true", help="Delete per-attempt artifacts immediately for windows with zero trade episodes.")
    parser.add_argument(
        "--preserve-matched-dir",
        type=Path,
        default=None,
        help="Optional directory to copy matched-attempt forensic artifacts into. Defaults to <out-prefix>_matched.",
    )
    return parser.parse_args()


def _gunzip_csv(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def _locate_captured_chunk(stream_dir: Path) -> Path:
    latest = stream_dir / "latest.csv.gz"
    if latest.exists():
        return latest
    raw_chunks = sorted((stream_dir / "raw").glob("*.csv.gz"))
    if raw_chunks:
        return raw_chunks[-1]
    archive_chunks = sorted((stream_dir / "archive").glob("*.csv.gz"))
    if archive_chunks:
        return archive_chunks[-1]
    raise SystemExit(f"missing captured gzip under {stream_dir}")


def _run(cmd: list[str], *, capture_output: bool = False, check: bool = True) -> tuple[int, str | None]:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        capture_output=capture_output,
        text=True,
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    stdout = completed.stdout if capture_output else None
    return completed.returncode, stdout


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _backfill_pair_row_timestamps(pair_rows: Path, bars_csv: Path) -> None:
    if not pair_rows.exists():
        return
    rows = pd.read_csv(pair_rows)
    if rows.empty:
        return
    if "timestamp" in rows.columns and rows["timestamp"].notna().any():
        return
    if "sort_t" not in rows.columns:
        return

    bars = pd.read_csv(bars_csv, usecols=["timestamp"])
    if bars.empty:
        return
    bars = bars.reset_index().rename(columns={"index": "sort_t"})
    merged = rows.merge(bars, on="sort_t", how="left", suffixes=("", "_bars"))
    if "timestamp_bars" in merged.columns:
        merged["timestamp"] = merged["timestamp_bars"]
        merged = merged.drop(columns=["timestamp_bars"])
    merged.to_csv(pair_rows, index=False)


def _cleanup_attempt_artifacts(out_prefix: Path) -> None:
    parent = out_prefix.parent
    stem = out_prefix.name
    for path in parent.glob(f"{stem}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)


def _depth_output_path(base_prefix: Path) -> Path:
    return base_prefix.with_name(f"{base_prefix.name}_depth.ndjson")


def _preserve_dir(base_prefix: Path, configured: Path | None) -> Path:
    if configured is not None:
        return configured
    return base_prefix.with_name(f"{base_prefix.name}_matched")


def _depth_collector_worker(
    *,
    symbol: str,
    limit: int,
    poll_interval: float,
    out_path: Path,
    flush_every: int,
    stop_event: threading.Event,
) -> None:
    ensure_dir(out_path.parent)
    buffer: list[str] = []
    with out_path.open("a", encoding="utf-8") as fh:
        while not stop_event.is_set():
            snapshot = fetch_binance_depth_snapshot(symbol=symbol, limit=limit)
            buffer.append(json.dumps(snapshot, separators=(",", ":")))
            if len(buffer) >= flush_every:
                fh.write("\n".join(buffer) + "\n")
                fh.flush()
                buffer.clear()
            stop_event.wait(max(0.0, poll_interval))
        if buffer:
            fh.write("\n".join(buffer) + "\n")
            fh.flush()


def _start_depth_collector(args: argparse.Namespace, base_prefix: Path) -> tuple[Path | None, threading.Event | None, threading.Thread | None]:
    if not args.capture_depth:
        return args.depth_ndjson, None, None
    depth_path = args.depth_ndjson or _depth_output_path(base_prefix)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_depth_collector_worker,
        kwargs={
            "symbol": args.symbol,
            "limit": int(args.depth_limit),
            "poll_interval": float(args.poll_interval),
            "out_path": depth_path,
            "flush_every": max(1, int(args.depth_flush_every)),
            "stop_event": stop_event,
        },
        name="binance-depth-collector",
        daemon=True,
    )
    worker.start()
    return depth_path, stop_event, worker


def _capture_once(args: argparse.Namespace, out_prefix: Path) -> dict[str, Any]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    stream_dir = out_prefix.with_name(f"{out_prefix.name}_stream")
    chunk_seconds = args.chunk_size_seconds if args.chunk_size_seconds is not None else args.duration_seconds
    stream_binance_trades(
        symbol=args.symbol,
        out_dir=str(stream_dir),
        duration_minutes=0.0,
        duration_seconds=float(args.duration_seconds),
        poll_interval=float(args.poll_interval),
        chunk_size_minutes=0.0,
        chunk_size_seconds=float(chunk_seconds),
        live_ingest=False,
    )

    latest_gz = _locate_captured_chunk(stream_dir)
    bars_csv = _gunzip_csv(latest_gz, out_prefix.with_suffix(".csv"))

    log_prefix = out_prefix.with_name(f"{out_prefix.name}_nashi")
    duckdb_path = out_prefix.with_suffix(".duckdb")
    _run(
        [
            args.python,
            "run_nashi.py",
            "--csv",
            str(bars_csv),
            "--symbol",
            args.symbol,
            "--log-prefix",
            str(log_prefix),
            "--duckdb",
            str(duckdb_path),
            "--progress-every",
            "0",
        ],
        check=True,
    )

    step_log = log_prefix.with_suffix(".csv")
    pair_rows = out_prefix.with_name(f"{out_prefix.name}_pairs_rows.csv")
    pair_summary = out_prefix.with_name(f"{out_prefix.name}_pairs_summary.json")
    _, debug_stdout = _run(
        [
            args.python,
            "scripts/debug_nashi_trade_pairs.py",
            "--input",
            str(step_log),
            "--top-k",
            str(args.top_k),
            "--context-rows",
            str(args.context_rows),
            "--rows-csv",
            str(pair_rows),
            "--summary-json",
            str(pair_summary),
        ]
        ,
        capture_output=True,
        check=True,
    )
    if not pair_summary.exists():
        fallback_summary = json.loads((debug_stdout or "").strip())
        pair_summary.write_text(json.dumps(fallback_summary, indent=2), encoding="utf-8")
    _backfill_pair_row_timestamps(pair_rows, bars_csv)

    pair_meta = _load_json(pair_summary)
    result: dict[str, object] = {
        "out_prefix": str(out_prefix),
        "bars_csv": str(bars_csv),
        "step_log": str(step_log),
        "pair_rows": str(pair_rows),
        "pair_summary": str(pair_summary),
        "episode_count": int(pair_meta.get("episode_count", 0)),
    }

    if args.depth_ndjson is not None and int(pair_meta.get("episode_count", 0)) > 0:
        l2_csv = out_prefix.with_name(f"{out_prefix.name}_l2_features.csv")
        l2_summary = out_prefix.with_name(f"{out_prefix.name}_l2_summary.json")
        l2_code, l2_stdout = _run(
            [
                args.python,
                "scripts/analyze_binance_depth_support_surface.py",
                "--depth-ndjson",
                str(args.depth_ndjson),
                "--summary-json",
                str(pair_summary),
                "--rows-csv",
                str(pair_rows),
                "--timestamp-tolerance",
                str(args.timestamp_tolerance),
                "--output-csv",
                str(l2_csv),
                "--summary-out",
                str(l2_summary),
            ],
            capture_output=True,
            check=False,
        )
        if l2_code == 0:
            result["l2_features_csv"] = str(l2_csv)
            result["l2_summary_json"] = str(l2_summary)
            try:
                l2_meta = _load_json(l2_summary)
                result["l2_matched_pairs"] = int(((l2_meta.get("pair_join") or {}).get("matched_pairs")) or 0)
            except Exception:
                result["l2_matched_pairs"] = 0
            try:
                l2_rows = pd.read_csv(l2_csv)
                valid_mask = pd.to_numeric(
                    l2_rows.get("warning_to_response_snapshot_delta"),
                    errors="coerce",
                ).fillna(0) > 0
                result["l2_distinct_snapshot_pairs"] = int(valid_mask.sum())
            except Exception:
                result["l2_distinct_snapshot_pairs"] = 0
        else:
            result["l2_skipped"] = "no_warning_response_pair"
            result["l2_matched_pairs"] = 0
            result["l2_distinct_snapshot_pairs"] = 0
            if l2_stdout:
                result["l2_message"] = l2_stdout.strip()
    elif args.depth_ndjson is not None:
        result["l2_skipped"] = "no_trade_episodes"
        result["l2_matched_pairs"] = 0
        result["l2_distinct_snapshot_pairs"] = 0

    return result


def _count_distinct_l2_pairs(attempts: list[dict[str, Any]]) -> int:
    return sum(int(item.get("l2_distinct_snapshot_pairs", 0)) for item in attempts)


def _matched_attempt(result: dict[str, Any]) -> bool:
    return int(result.get("l2_distinct_snapshot_pairs", 0)) > 0


def _copy_if_exists(path_str: str | None, target_dir: Path) -> str | None:
    if not path_str:
        return None
    source = Path(path_str)
    if not source.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return str(target)


def _preserve_attempt_artifacts(result: dict[str, Any], target_dir: Path) -> dict[str, str]:
    preserved: dict[str, str] = {}
    for key in (
        "bars_csv",
        "step_log",
        "pair_rows",
        "pair_summary",
        "l2_features_csv",
        "l2_summary_json",
    ):
        copied = _copy_if_exists(result.get(key), target_dir)
        if copied is not None:
            preserved[key] = copied
    return preserved


def _build_aggregate_outputs(
    *,
    base_prefix: Path,
    attempts: list[dict[str, Any]],
    preserve_root: Path,
) -> tuple[str | None, str | None]:
    matched = [item for item in attempts if _matched_attempt(item)]
    if not matched:
        return None, None

    rows: list[pd.DataFrame] = []
    for item in matched:
        csv_path = Path(item.get("preserved", {}).get("l2_features_csv") or item.get("l2_features_csv", ""))
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["attempt"] = int(item.get("attempt", 0))
        rows.append(df)

    aggregate_csv = preserve_root / f"{base_prefix.name}_aggregate_l2_features.csv"
    aggregate_json = preserve_root / f"{base_prefix.name}_aggregate_summary.json"

    if rows:
        merged = pd.concat(rows, ignore_index=True)
    else:
        merged = pd.DataFrame()

    if not merged.empty and "warning_to_response_snapshot_delta" in merged.columns:
        delta = pd.to_numeric(merged["warning_to_response_snapshot_delta"], errors="coerce").fillna(0)
        merged = merged.loc[delta > 0].reset_index(drop=True)

    merged.to_csv(aggregate_csv, index=False)

    class_counts: dict[str, int] = {}
    support_regime_counts: dict[str, int] = {}
    if not merged.empty:
        if "class" in merged.columns:
            class_counts = merged["class"].value_counts(dropna=False).to_dict()
        elif "class_name" in merged.columns:
            class_counts = merged["class_name"].value_counts(dropna=False).to_dict()
        if "support_regime" in merged.columns:
            support_regime_counts = merged["support_regime"].value_counts(dropna=False).to_dict()

    summary = {
        "matched_attempt_count": len(matched),
        "l2_distinct_pairs_collected": _count_distinct_l2_pairs(attempts),
        "pair_row_count": int(len(merged)),
        "class_counts": class_counts,
        "support_regime_counts": support_regime_counts,
    }

    if not merged.empty:
        for col in (
            "support_restored",
            "support_dead",
            "thinning",
            "false_restoration_flag",
            "real_restoration_flag",
            "dead_support_flag",
        ):
            if col in merged.columns:
                summary[f"{col}_rate"] = float(pd.to_numeric(merged[col], errors="coerce").fillna(0).mean())
        for col in (
            "obi_restore",
            "near_mass_restore",
            "spread_change",
            "jsd_change",
            "support_persistence_mean",
            "support_positive_frac",
            "mid_follow_through_h",
            "support_decay_after_restore",
            "warning_to_response_snapshot_delta",
        ):
            if col in merged.columns:
                summary[f"{col}_mean"] = float(pd.to_numeric(merged[col], errors="coerce").dropna().mean())

        if "class" in merged.columns:
            class_summaries: dict[str, dict[str, Any]] = {}
            for klass, grp in merged.groupby("class", dropna=False):
                cls_summary: dict[str, Any] = {"count": int(len(grp))}
                if "support_regime" in grp.columns:
                    cls_summary["support_regime_counts"] = grp["support_regime"].value_counts(dropna=False).to_dict()
                for col in (
                    "support_restored",
                    "support_dead",
                    "thinning",
                    "false_restoration_flag",
                    "real_restoration_flag",
                    "dead_support_flag",
                ):
                    if col in grp.columns:
                        cls_summary[f"{col}_rate"] = float(pd.to_numeric(grp[col], errors="coerce").fillna(0).mean())
                for col in (
                    "obi_restore",
                    "near_mass_restore",
                    "spread_change",
                    "jsd_change",
                    "support_persistence_mean",
                    "support_positive_frac",
                    "mid_follow_through_h",
                    "support_decay_after_restore",
                    "warning_to_response_snapshot_delta",
                ):
                    if col in grp.columns:
                        cls_summary[f"{col}_mean"] = float(pd.to_numeric(grp[col], errors="coerce").dropna().mean())
                class_summaries[str(klass)] = cls_summary
            summary["classes"] = class_summaries

    aggregate_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(aggregate_csv), str(aggregate_json)


def _attempt_satisfies_stop(args: argparse.Namespace, result: dict[str, Any], attempts: list[dict[str, Any]]) -> bool:
    if not args.repeat_until_episode:
        return True
    if int(result.get("episode_count", 0)) <= 0:
        return False
    if args.require_l2_match or args.depth_ndjson is not None:
        return _count_distinct_l2_pairs(attempts) >= max(1, int(args.target_l2_distinct_pairs))
    return True


def main() -> None:
    args = parse_args()
    base_prefix = Path(args.out_prefix)
    preserve_root = _preserve_dir(base_prefix, args.preserve_matched_dir)
    attempts: list[dict[str, Any]] = []
    max_attempts = int(args.max_attempts)
    attempt = 1
    depth_path, depth_stop, depth_thread = _start_depth_collector(args, base_prefix)
    args.depth_ndjson = depth_path

    try:
        while True:
            suffix = f"_attempt{attempt:02d}" if args.repeat_until_episode else ""
            out_prefix = base_prefix.with_name(f"{base_prefix.name}{suffix}")
            result = _capture_once(args, out_prefix)
            result["attempt"] = attempt
            if _matched_attempt(result):
                result["preserved"] = _preserve_attempt_artifacts(
                    result,
                    preserve_root / f"attempt{attempt:02d}",
                )
            attempts.append(result)
            if _attempt_satisfies_stop(args, result, attempts):
                break
            if args.cleanup_misses:
                _cleanup_attempt_artifacts(out_prefix)
            if max_attempts > 0 and attempt >= max_attempts:
                break
            attempt += 1
            time.sleep(float(args.sleep_between_attempts))
    finally:
        if depth_stop is not None:
            depth_stop.set()
        if depth_thread is not None:
            depth_thread.join(timeout=max(5.0, float(args.poll_interval) * 2.0))

    final: dict[str, Any] = {
        "repeat_until_episode": bool(args.repeat_until_episode),
        "require_l2_match": bool(args.require_l2_match or args.depth_ndjson is not None),
        "target_l2_distinct_pairs": int(args.target_l2_distinct_pairs),
        "attempt_count": len(attempts),
        "l2_distinct_pairs_collected": _count_distinct_l2_pairs(attempts),
        "matched_attempt": next((item for idx, item in enumerate(attempts, start=1) if _attempt_satisfies_stop(args, item, attempts[:idx])), None),
        "matched_attempts": [item for item in attempts if int(item.get("l2_distinct_snapshot_pairs", 0)) > 0],
        "attempts": attempts,
    }
    if depth_path is not None:
        final["depth_ndjson"] = str(depth_path)
    final["preserve_root"] = str(preserve_root)

    aggregate_csv, aggregate_json = _build_aggregate_outputs(
        base_prefix=base_prefix,
        attempts=attempts,
        preserve_root=preserve_root,
    )
    if aggregate_csv is not None:
        final["aggregate_l2_features_csv"] = aggregate_csv
    if aggregate_json is not None:
        final["aggregate_summary_json"] = aggregate_json

    result_path = base_prefix.with_name(f"{base_prefix.name}_result.json")
    result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

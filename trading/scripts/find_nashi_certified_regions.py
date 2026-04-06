#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nashi.certification import (  # noqa: E402
    attribution_breakdown,
    certification_census,
    load_certification_frame,
    summarize_attribution,
    summarize_census,
)


DEFAULT_GLOBS = (
    "logs/nashi/*.csv",
    "logs/research/*nashi*.duckdb",
    "logs/research/*phase9*.duckdb",
    "logs/research/*smoke*.duckdb",
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", action="append", dest="globs", help="Additional artifact glob(s) relative to repo root")
    parser.add_argument("--window-rows", type=int, default=256, help="Rows per certification window")
    parser.add_argument("--window-seconds", type=float, default=0.0, help="Optional time window in seconds; overrides --window-rows when > 0")
    parser.add_argument("--top-k", type=int, default=25, help="Number of top-ranked windows to print")
    parser.add_argument("--fresh-hours", type=float, default=48.0, help="Recent-file window for focused fresh-bundle reporting. Use 0 to disable.")
    parser.add_argument("--fresh-top-k", type=int, default=12, help="Rows to print in each fresh-bundle comparison section.")
    parser.add_argument("--output-csv", help="Optional path for ranked corpus CSV")
    parser.add_argument("--summary-json", help="Optional path for corpus summary JSON")
    return parser.parse_args()


def discover_files(globs: list[str] | None) -> list[Path]:
    patterns = list(DEFAULT_GLOBS)
    if globs:
        patterns.extend(globs)
    paths: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_absolute():
            if any(ch in pattern for ch in "*?[]"):
                paths.update(candidate.parent.glob(candidate.name))
            elif candidate.is_file():
                paths.add(candidate)
            continue
        paths.update(ROOT.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def capability_flags(frame: pd.DataFrame) -> dict[str, bool]:
    cols = set(frame.columns)
    return {
        "has_family": "nashi_family_trade_certified" in cols and "nashi_family_preserve_certified" in cols,
        "has_phase9": "expected_surplus" in cols and "mw_refusal_level" in cols,
        "has_status": "nashi_status" in cols,
        "has_symbol": "symbol" in cols,
        "has_ts": "ts" in cols or "timestamp" in cols,
    }


def _bundle_name(path: Path) -> str:
    return path.stem


def _hazard_tightened_source_label(row: pd.Series) -> str:
    contextual = int(row.get("hazard_contextual_tightened_count", 0))
    synthetic = int(row.get("hazard_synthetic_tightened_count", 0))
    if contextual > 0 and synthetic > 0:
        return "mixed"
    if contextual > 0:
        return "contextual"
    if synthetic > 0:
        return "synthetic_only"
    return "none"


def _hazard_drag_slice_label(row: pd.Series) -> str:
    contextual = float(row.get("hazard_contextual_aligned_drag_share", 0.0))
    synthetic = float(row.get("hazard_synthetic_aligned_drag_share", 0.0))
    if contextual > 1e-9 and synthetic > 1e-9:
        return "mixed"
    if contextual > 1e-9:
        return "contextual"
    if synthetic > 1e-9:
        return "synthetic_only"
    return "none"


def rank_census(census: pd.DataFrame) -> pd.DataFrame:
    ranked = census.copy()
    contextual_mask = ranked.get("hazard_contextual_tightened_count", pd.Series(0, index=ranked.index)).gt(0)
    synthetic_mask = ranked.get("hazard_synthetic_tightened_count", pd.Series(0, index=ranked.index)).gt(0)
    ranked["hazard_window_slice"] = "none"
    ranked.loc[synthetic_mask & ~contextual_mask, "hazard_window_slice"] = "synthetic_only"
    ranked.loc[contextual_mask & ~synthetic_mask, "hazard_window_slice"] = "contextual_only"
    ranked.loc[contextual_mask & synthetic_mask, "hazard_window_slice"] = "mixed"
    ranked["hazard_window_label"] = ranked["hazard_window_slice"]
    if "hazard_contextual_mode" in ranked.columns:
        labels = ranked["hazard_contextual_mode"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        ranked.loc[ranked["hazard_window_slice"].eq("contextual_only"), "hazard_window_label"] = (
            "contextual:" + labels.loc[ranked["hazard_window_slice"].eq("contextual_only")]
        )
        ranked.loc[ranked["hazard_window_slice"].eq("mixed"), "hazard_window_label"] = (
            "mixed:" + labels.loc[ranked["hazard_window_slice"].eq("mixed")]
        )
    class_boost = ranked["window_class"].map(
        {
            "tradeable": 3.0,
            "preserve_only": 1.5,
            "ban_dominated": 0.5,
            "mixed": 1.0,
        }
    ).fillna(0.0)
    efficiency_bonus = ranked["realized_efficiency"].clip(lower=-1.0, upper=2.0)
    efficiency_active = ranked["executed_expected_surplus_active"].astype(float)
    aligned_efficiency_bonus = ranked.get("aligned_realized_efficiency", pd.Series(0.0, index=ranked.index)).clip(lower=-1.0, upper=2.0)
    aligned_efficiency_active = ranked.get("aligned_expected_surplus_active", pd.Series(False, index=ranked.index)).astype(float)
    execution_fill_bonus = ranked.get("execution_fill_ratio_mean", pd.Series(0.0, index=ranked.index)).clip(lower=0.0, upper=1.0)
    execution_cost_penalty = ranked.get("execution_cost_gap_sum", pd.Series(0.0, index=ranked.index)).clip(lower=0.0)
    contextual_drag_penalty = ranked.get("hazard_contextual_aligned_drag_share", pd.Series(0.0, index=ranked.index)).clip(lower=0.0, upper=2.0)
    synthetic_drag_penalty = ranked.get("hazard_synthetic_aligned_drag_share", pd.Series(0.0, index=ranked.index)).clip(lower=0.0, upper=2.0)
    contextual_efficiency_bonus = ranked.get("hazard_contextual_aligned_efficiency", pd.Series(0.0, index=ranked.index)).clip(lower=-1.0, upper=2.0)
    synthetic_efficiency_bonus = ranked.get("hazard_synthetic_aligned_efficiency", pd.Series(0.0, index=ranked.index)).clip(lower=-1.0, upper=2.0)
    ranked["ranking_score"] = class_boost + (
        4.0 * ranked["trade_certified_share"]
        + 2.5 * ranked["preserve_certified_share"]
        + 2.0 * ranked["ban_coverage"]
        + 0.5 * ranked["ban_correct_share"]
        + 1.5 * efficiency_bonus * efficiency_active
        + 2.0 * aligned_efficiency_bonus * aligned_efficiency_active
        + 0.75 * contextual_efficiency_bonus
        + 0.35 * synthetic_efficiency_bonus
        + 0.25 * execution_fill_bonus
        + 1e-6 * ranked["executed_expected_surplus_sum"].clip(lower=0.0)
        + 1e-6 * ranked.get("aligned_expected_surplus_sum", pd.Series(0.0, index=ranked.index)).clip(lower=0.0)
        + 2.5e-7 * ranked["proposed_expected_surplus_sum"].clip(lower=0.0)
        + 5e-7 * ranked["realized_surplus_sum"].clip(lower=0.0)
        + 5e-7 * ranked.get("aligned_realized_surplus_sum", pd.Series(0.0, index=ranked.index)).clip(lower=0.0)
        - 2.5 * (ranked["ban_missed_count"] > 0).astype(float)
        - 1.5 * ranked["ban_required_share"]
        - 0.75 * contextual_drag_penalty
        - 0.35 * synthetic_drag_penalty
        - 5e-7 * execution_cost_penalty
    )
    return ranked.sort_values(
        [
            "window_class",
            "ranking_score",
            "trade_certified_share",
            "preserve_certified_share",
            "ban_coverage",
            "hazard_contextual_aligned_efficiency",
            "hazard_synthetic_aligned_efficiency",
            "hazard_contextual_aligned_drag_share",
            "hazard_synthetic_aligned_drag_share",
            "aligned_realized_efficiency",
            "realized_efficiency",
            "executed_expected_surplus_sum",
            "proposed_expected_surplus_sum",
        ],
        ascending=[True, False, False, False, False, False, False, True, True, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def scan_corpus(paths: list[Path], *, window_rows: int | None, window_ms: int | None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ranked_frames: list[pd.DataFrame] = []
    attribution_frames: list[pd.DataFrame] = []
    file_reports: list[dict[str, Any]] = []

    for path in paths:
        report: dict[str, Any] = {"path": _display_path(path)}
        try:
            frame = load_certification_frame(path)
            flags = capability_flags(frame)
            report.update(flags)
            if not all(flags[key] for key in ("has_symbol", "has_ts", "has_phase9", "has_status")):
                report["status"] = "skipped_incompatible"
                report["reason"] = "missing required emitted fields"
                file_reports.append(report)
                continue

            census = certification_census(frame, window_rows=window_rows, window_ms=window_ms)
            attribution = attribution_breakdown(frame)
            summary = summarize_census(census)
            if not attribution.empty:
                summary["drag_attribution"] = summarize_attribution(attribution, top_k=3)
            report["status"] = "ok"
            report["summary"] = summary
            file_reports.append(report)

            if census.empty:
                continue
            census = rank_census(census)
            census.insert(0, "source_path", _display_path(path))
            census.insert(1, "has_family_fields", bool(flags["has_family"]))
            census.insert(2, "bundle_name", _bundle_name(path))
            census.insert(3, "source_mtime", float(path.stat().st_mtime))
            ranked_frames.append(census)
            if not attribution.empty:
                attribution = attribution.copy()
                attribution.insert(0, "source_path", _display_path(path))
                attribution_frames.append(attribution)
        except Exception as exc:  # pragma: no cover - defensive scanner path
            report["status"] = "error"
            report["reason"] = f"{type(exc).__name__}: {exc}"
            file_reports.append(report)

    if not ranked_frames:
        return pd.DataFrame(), file_reports, pd.DataFrame()
    corpus = pd.concat(ranked_frames, ignore_index=True)
    if attribution_frames:
        attribution_corpus = pd.concat(attribution_frames, ignore_index=True)
        attribution_corpus = attribution_corpus.groupby(["axis", "label"], as_index=False).agg(
            source_count=("source_path", "nunique"),
            row_count=("row_count", "sum"),
            proposed_expected_surplus_sum=("proposed_expected_surplus_sum", "sum"),
            executed_expected_surplus_sum=("executed_expected_surplus_sum", "sum"),
            realized_surplus_sum=("realized_surplus_sum", "sum"),
            drag_surplus_sum=("drag_surplus_sum", "sum"),
            aligned_expected_surplus_sum=("aligned_expected_surplus_sum", "sum"),
            aligned_realized_surplus_sum=("aligned_realized_surplus_sum", "sum"),
            aligned_drag_surplus_sum=("aligned_drag_surplus_sum", "sum"),
            execution_cost_realized_sum=("execution_cost_realized_sum", "sum"),
            execution_cost_gap_sum=("execution_cost_gap_sum", "sum"),
            execution_fill_ratio_mean=("execution_fill_ratio_mean", "mean"),
            hazard_active_share=("hazard_active_share", "mean"),
            hazard_contextual_active_share=("hazard_contextual_active_share", "mean"),
        )
        axis_drag_total = attribution_corpus.groupby("axis")["drag_surplus_sum"].transform("sum")
        attribution_corpus["drag_share"] = 0.0
        positive_axis_drag = axis_drag_total > 0.0
        attribution_corpus.loc[positive_axis_drag, "drag_share"] = (
            attribution_corpus.loc[positive_axis_drag, "drag_surplus_sum"]
            / axis_drag_total.loc[positive_axis_drag]
        )
        axis_aligned_drag_total = attribution_corpus.groupby("axis")["aligned_drag_surplus_sum"].transform("sum")
        attribution_corpus["aligned_drag_share"] = 0.0
        positive_axis_aligned_drag = axis_aligned_drag_total > 0.0
        attribution_corpus.loc[positive_axis_aligned_drag, "aligned_drag_share"] = (
            attribution_corpus.loc[positive_axis_aligned_drag, "aligned_drag_surplus_sum"]
            / axis_aligned_drag_total.loc[positive_axis_aligned_drag]
        )
        active = attribution_corpus["executed_expected_surplus_sum"].abs() > 1e-9
        attribution_corpus["realized_efficiency"] = 0.0
        attribution_corpus.loc[active, "realized_efficiency"] = (
            attribution_corpus.loc[active, "realized_surplus_sum"]
            / attribution_corpus.loc[active, "executed_expected_surplus_sum"]
        )
        aligned_active = attribution_corpus["aligned_expected_surplus_sum"].abs() > 1e-9
        attribution_corpus["aligned_realized_efficiency"] = 0.0
        attribution_corpus.loc[aligned_active, "aligned_realized_efficiency"] = (
            attribution_corpus.loc[aligned_active, "aligned_realized_surplus_sum"]
            / attribution_corpus.loc[aligned_active, "aligned_expected_surplus_sum"]
        )
        attribution_corpus = attribution_corpus.sort_values(
            ["axis", "aligned_drag_surplus_sum", "drag_surplus_sum", "aligned_expected_surplus_sum", "row_count", "label"],
            ascending=[True, False, False, False, False, True],
            kind="stable",
        ).reset_index(drop=True)
    else:
        attribution_corpus = pd.DataFrame()
    return rank_census(corpus), file_reports, attribution_corpus


def fresh_bundle_comparison(
    corpus: pd.DataFrame,
    *,
    fresh_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if corpus.empty or fresh_hours <= 0.0 or "source_mtime" not in corpus.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cutoff = time.time() - float(fresh_hours) * 3600.0
    fresh = corpus[corpus["source_mtime"] >= cutoff].copy()
    if fresh.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    fresh["hazard_tightened_source"] = fresh.apply(_hazard_tightened_source_label, axis=1)
    fresh["hazard_drag_slice"] = fresh.apply(_hazard_drag_slice_label, axis=1)

    bundle_summary = (
        fresh.groupby(["bundle_name", "source_path"], as_index=False)
        .agg(
            window_count=("window_id", "count"),
            ranking_score_mean=("ranking_score", "mean"),
            ranking_score_max=("ranking_score", "max"),
            trade_certified_count=("trade_certified_count", "sum"),
            preserve_certified_count=("preserve_certified_count", "sum"),
            ban_required_count=("ban_required_count", "sum"),
            aligned_realized_efficiency_mean=("aligned_realized_efficiency", "mean"),
            hazard_contextual_aligned_drag_share_mean=("hazard_contextual_aligned_drag_share", "mean"),
            hazard_synthetic_aligned_drag_share_mean=("hazard_synthetic_aligned_drag_share", "mean"),
            hazard_contextual_tightened_windows=("hazard_contextual_tightened_count", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
            hazard_synthetic_tightened_windows=("hazard_synthetic_tightened_count", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
        )
        .sort_values(
            ["ranking_score_max", "aligned_realized_efficiency_mean", "hazard_contextual_aligned_drag_share_mean", "hazard_synthetic_aligned_drag_share_mean"],
            ascending=[False, False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    hazard_source_summary = (
        fresh.groupby(["bundle_name", "source_path", "hazard_tightened_source"], as_index=False)
        .agg(
            window_count=("window_id", "count"),
            ranking_score_mean=("ranking_score", "mean"),
            aligned_realized_efficiency_mean=("aligned_realized_efficiency", "mean"),
            trade_certified_count=("trade_certified_count", "sum"),
            preserve_certified_count=("preserve_certified_count", "sum"),
            ban_required_count=("ban_required_count", "sum"),
        )
        .sort_values(["bundle_name", "hazard_tightened_source", "ranking_score_mean"], ascending=[True, True, False], kind="stable")
        .reset_index(drop=True)
    )

    hazard_drag_summary = (
        fresh.groupby(["bundle_name", "source_path", "hazard_drag_slice"], as_index=False)
        .agg(
            window_count=("window_id", "count"),
            ranking_score_mean=("ranking_score", "mean"),
            aligned_realized_efficiency_mean=("aligned_realized_efficiency", "mean"),
            hazard_contextual_aligned_drag_share_mean=("hazard_contextual_aligned_drag_share", "mean"),
            hazard_synthetic_aligned_drag_share_mean=("hazard_synthetic_aligned_drag_share", "mean"),
        )
        .sort_values(["bundle_name", "hazard_drag_slice", "ranking_score_mean"], ascending=[True, True, False], kind="stable")
        .reset_index(drop=True)
    )
    return bundle_summary, hazard_source_summary, hazard_drag_summary


def corpus_summary(corpus: pd.DataFrame, file_reports: list[dict[str, Any]], attribution: pd.DataFrame) -> dict[str, Any]:
    ok_reports = [report for report in file_reports if report.get("status") == "ok"]
    skipped_reports = [report for report in file_reports if report.get("status") == "skipped_incompatible"]
    error_reports = [report for report in file_reports if report.get("status") == "error"]
    return {
        "artifact_count": len(file_reports),
        "ok_artifacts": len(ok_reports),
        "skipped_artifacts": len(skipped_reports),
        "error_artifacts": len(error_reports),
        "ranked_window_count": int(len(corpus)),
        "ranked_symbol_count": int(corpus["symbol"].nunique()) if not corpus.empty else 0,
        "trade_certified_windows": int((corpus["trade_certified_count"] > 0).sum()) if not corpus.empty else 0,
        "preserve_certified_windows": int((corpus["preserve_certified_count"] > 0).sum()) if not corpus.empty else 0,
        "ban_required_windows": int((corpus["ban_required_count"] > 0).sum()) if not corpus.empty else 0,
        "ban_missed_windows": int((corpus["ban_missed_count"] > 0).sum()) if not corpus.empty else 0,
        "positive_realized_efficiency_windows": int((corpus["realized_efficiency"] > 0.0).sum()) if not corpus.empty else 0,
        "mean_realized_efficiency": float(corpus["realized_efficiency"].mean()) if not corpus.empty else 0.0,
        "positive_aligned_realized_efficiency_windows": int((corpus["aligned_realized_efficiency"] > 0.0).sum()) if not corpus.empty and "aligned_realized_efficiency" in corpus.columns else 0,
        "mean_aligned_realized_efficiency": float(corpus["aligned_realized_efficiency"].mean()) if not corpus.empty and "aligned_realized_efficiency" in corpus.columns else 0.0,
        "mean_hazard_contextual_aligned_efficiency": float(corpus["hazard_contextual_aligned_efficiency"].mean()) if not corpus.empty and "hazard_contextual_aligned_efficiency" in corpus.columns else 0.0,
        "mean_hazard_synthetic_aligned_efficiency": float(corpus["hazard_synthetic_aligned_efficiency"].mean()) if not corpus.empty and "hazard_synthetic_aligned_efficiency" in corpus.columns else 0.0,
        "mean_hazard_contextual_aligned_drag_share": float(corpus["hazard_contextual_aligned_drag_share"].mean()) if not corpus.empty and "hazard_contextual_aligned_drag_share" in corpus.columns else 0.0,
        "mean_hazard_synthetic_aligned_drag_share": float(corpus["hazard_synthetic_aligned_drag_share"].mean()) if not corpus.empty and "hazard_synthetic_aligned_drag_share" in corpus.columns else 0.0,
        "window_classes": corpus["window_class"].value_counts().to_dict() if not corpus.empty else {},
        "hazard_source_windows": (
            {
                "synthetic_only": int((corpus["hazard_synthetic_tightened_count"] > 0).sum()) if "hazard_synthetic_tightened_count" in corpus.columns else 0,
                "contextual": int((corpus["hazard_contextual_tightened_count"] > 0).sum()) if "hazard_contextual_tightened_count" in corpus.columns else 0,
            }
            if not corpus.empty
            else {}
        ),
        "hazard_window_slices": corpus["hazard_window_slice"].value_counts().to_dict() if not corpus.empty and "hazard_window_slice" in corpus.columns else {},
        "drag_attribution": summarize_attribution(attribution, top_k=5) if not attribution.empty else {},
        "top_sources": (
            corpus["source_path"].value_counts().head(10).to_dict() if not corpus.empty else {}
        ),
        "files": file_reports,
    }


def print_fresh_bundle_comparison(
    bundle_summary: pd.DataFrame,
    hazard_source_summary: pd.DataFrame,
    hazard_drag_summary: pd.DataFrame,
    *,
    top_k: int,
) -> None:
    if bundle_summary.empty:
        return
    print("[fresh:bundles]")
    print(
        bundle_summary.loc[
            :,
            [
                "bundle_name",
                "source_path",
                "window_count",
                "ranking_score_max",
                "ranking_score_mean",
                "trade_certified_count",
                "preserve_certified_count",
                "ban_required_count",
                "aligned_realized_efficiency_mean",
                "hazard_contextual_tightened_windows",
                "hazard_synthetic_tightened_windows",
                "hazard_contextual_aligned_drag_share_mean",
                "hazard_synthetic_aligned_drag_share_mean",
            ],
        ].head(top_k).to_csv(index=False)
    )
    if not hazard_source_summary.empty:
        print("[fresh:hazard_tightened_source]")
        print(
            hazard_source_summary.loc[
                :,
                [
                    "bundle_name",
                    "source_path",
                    "hazard_tightened_source",
                    "window_count",
                    "ranking_score_mean",
                    "aligned_realized_efficiency_mean",
                    "trade_certified_count",
                    "preserve_certified_count",
                    "ban_required_count",
                ],
            ].head(top_k).to_csv(index=False)
        )
    if not hazard_drag_summary.empty:
        print("[fresh:hazard_drag_slice]")
        print(
            hazard_drag_summary.loc[
                :,
                [
                    "bundle_name",
                    "source_path",
                    "hazard_drag_slice",
                    "window_count",
                    "ranking_score_mean",
                    "aligned_realized_efficiency_mean",
                    "hazard_contextual_aligned_drag_share_mean",
                    "hazard_synthetic_aligned_drag_share_mean",
                ],
            ].head(top_k).to_csv(index=False)
        )


def print_top_windows(corpus: pd.DataFrame, *, top_k: int) -> None:
    if corpus.empty:
        print("No compatible certification windows found.")
        return
    columns = [
        "source_path",
        "symbol",
        "window_id",
        "window_class",
        "ranking_score",
        "trade_certified_count",
        "preserve_certified_count",
        "ban_required_count",
        "ban_correct_count",
        "ban_missed_count",
        "aligned_realized_efficiency",
        "hazard_contextual_aligned_efficiency",
        "hazard_synthetic_aligned_efficiency",
        "hazard_contextual_aligned_drag_share",
        "hazard_synthetic_aligned_drag_share",
        "realized_efficiency",
        "execution_fill_ratio_mean",
        "execution_cost_gap_sum",
        "aligned_expected_surplus_sum",
        "aligned_realized_surplus_sum",
        "executed_expected_surplus_active",
        "executed_expected_surplus_sum",
        "proposed_expected_surplus_sum",
        "realized_surplus_sum",
        "hazard_contextual_tightened_count",
        "hazard_synthetic_tightened_count",
        "hazard_window_slice",
        "hazard_window_label",
        "family_mode",
        "family_capability",
        "family_reason_mode",
        "refusal_mode",
    ]
    for window_class in ("tradeable", "preserve_only", "ban_dominated", "mixed"):
        subset = corpus[corpus["window_class"] == window_class]
        if subset.empty:
            continue
        print(f"[{window_class}]")
        print(subset.loc[:, columns].head(top_k).to_csv(index=False))

    for hazard_slice in ("contextual_only", "synthetic_only", "mixed", "none"):
        subset = corpus[corpus["hazard_window_slice"] == hazard_slice]
        if subset.empty:
            continue
        print(f"[hazard:{hazard_slice}]")
        print(subset.loc[:, columns].head(top_k).to_csv(index=False))


def print_drag_attribution(attribution: pd.DataFrame, *, top_k: int) -> None:
    if attribution.empty:
        return
    columns = [
        "label",
        "source_count",
        "row_count",
        "drag_share",
        "drag_surplus_sum",
        "aligned_drag_share",
        "aligned_drag_surplus_sum",
        "executed_expected_surplus_sum",
        "realized_surplus_sum",
        "realized_efficiency",
        "aligned_expected_surplus_sum",
        "aligned_realized_surplus_sum",
        "aligned_realized_efficiency",
        "execution_cost_realized_sum",
        "execution_cost_gap_sum",
        "execution_fill_ratio_mean",
        "hazard_active_share",
        "hazard_contextual_active_share",
    ]
    for axis in ("refusal_mode", "refusal_reason", "actionability_band", "spread_regime", "hazard_source", "hazard_tightened_source", "hazard_drag_slice", "hazard_name", "hazard_contextual_label", "family_class", "family_reason"):
        subset = attribution[attribution["axis"] == axis]
        if subset.empty:
            continue
        print(f"[drag:{axis}]")
        print(subset.loc[:, columns].head(top_k).to_csv(index=False))


def main() -> None:
    args = parse_args()
    paths = discover_files(args.globs)
    window_ms = int(args.window_seconds * 1000.0) if args.window_seconds and args.window_seconds > 0.0 else None
    window_rows = None if window_ms is not None else args.window_rows
    corpus, file_reports, attribution = scan_corpus(paths, window_rows=window_rows, window_ms=window_ms)
    fresh_bundle_summary, fresh_hazard_source_summary, fresh_hazard_drag_summary = fresh_bundle_comparison(
        corpus,
        fresh_hours=args.fresh_hours,
    )
    summary = corpus_summary(corpus, file_reports, attribution)
    if not fresh_bundle_summary.empty:
        summary["fresh_bundles"] = {
            "count": int(len(fresh_bundle_summary)),
            "bundle_names": fresh_bundle_summary["bundle_name"].head(args.fresh_top_k).tolist(),
            "hazard_tightened_source_modes": fresh_hazard_source_summary["hazard_tightened_source"].value_counts().to_dict() if not fresh_hazard_source_summary.empty else {},
            "hazard_drag_slice_modes": fresh_hazard_drag_summary["hazard_drag_slice"].value_counts().to_dict() if not fresh_hazard_drag_summary.empty else {},
        }

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        corpus.to_csv(output_csv, index=False)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2, sort_keys=True))
    print_top_windows(corpus, top_k=args.top_k)
    print_fresh_bundle_comparison(
        fresh_bundle_summary,
        fresh_hazard_source_summary,
        fresh_hazard_drag_summary,
        top_k=args.fresh_top_k,
    )
    print_drag_attribution(attribution, top_k=args.top_k)


if __name__ == "__main__":
    main()

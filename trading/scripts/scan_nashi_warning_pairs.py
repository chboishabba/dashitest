#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", action="append", help="Pair-summary JSON; repeatable")
    parser.add_argument("--glob", action="append", help="Glob for pair-summary JSON files; repeatable")
    parser.add_argument("--output-csv", help="Optional CSV path for per-episode classifications")
    parser.add_argument("--summary-out", help="Optional JSON path for aggregate summary")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _normalize_lineage(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_pairs"):
        stem = stem[: -len("_pairs")]
    stem = stem.replace("_default", "").replace("_context", "")
    stem = stem.replace("_trade", "")
    return stem


def _load_paths(args: argparse.Namespace) -> list[Path]:
    found: dict[str, Path] = {}
    for raw in args.summary_json or []:
        path = Path(raw)
        found[str(path)] = path
    for pattern in args.glob or []:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw)
            found[str(path)] = path
    paths = [path for path in found.values() if path.exists()]
    if not paths:
        raise SystemExit("no summary JSON files found")
    return sorted(paths)


def _episode_signature(episode: dict[str, Any]) -> dict[str, Any]:
    forensic = dict(episode.get("continuation_forensics") or {})
    return {
        "first_non_interior_t": forensic.get("first_non_interior_t"),
        "first_non_interior_family": forensic.get("first_non_interior_family"),
        "first_non_interior_lead_signal": forensic.get("first_non_interior_lead_signal"),
        "first_adverse_t": forensic.get("first_adverse_t"),
        "first_adverse_family": forensic.get("first_adverse_family"),
        "first_adverse_lead_signal": forensic.get("first_adverse_lead_signal"),
        "terminal_family": forensic.get("terminal_family"),
        "duration_rows": int(_safe_float(episode.get("duration_rows"), 0.0)),
    }


def _classify_episode(rows: pd.DataFrame) -> dict[str, Any]:
    episode = rows[rows.get("in_episode", 0).astype(bool)].copy().reset_index(drop=True)
    if episode.empty:
        return {
            "warning_response_class": "empty_episode",
            "warning_response_reason": "episode has no in-episode rows",
        }

    if len(episode) < 2:
        return {
            "warning_response_class": "insufficient_prefix",
            "warning_response_reason": "need at least entry row plus warning row",
        }

    entry = episode.iloc[0].to_dict()
    warning = episode.iloc[1].to_dict()
    response = episode.iloc[2].to_dict() if len(episode) > 2 else {}

    warning_family = str(warning.get("family", ""))
    response_family = str(response.get("family", ""))
    warning_lead = str(warning.get("lead_signal", ""))
    response_lead = str(response.get("lead_signal", ""))
    warning_expected = _safe_float(warning.get("executed_expected_surplus"))
    response_expected = _safe_float(response.get("executed_expected_surplus"))
    warning_survivability = _safe_float(warning.get("proposed_survivability_score"))
    response_survivability = _safe_float(response.get("proposed_survivability_score"))
    warning_support_collapsed = bool(warning.get("support_collapsed", warning_expected <= 1e-9))
    response_support_collapsed = bool(response.get("support_collapsed", response_expected <= 1e-9))
    response_present = len(episode) > 2
    response_spread = str(response.get("nashi_spread_regime", ""))
    response_fill = _safe_float(response.get("fill"))
    response_exposure_post = _safe_float(response.get("exposure_post", response.get("exposure")))

    if warning_family == "flatten_transition":
        klass = "immediate_flatten"
        reason = "warning row is already flatten_transition"
    elif (
        not response_present
        and warning_expected <= 1e-9
        and abs(_safe_float(warning.get("exposure_post", warning.get("exposure")))) <= 1e-9
        and _safe_float(warning.get("fill")) < 0.0
    ):
        klass = "immediate_flatten"
        reason = "warning row already flattened the position in a 2-row episode"
    elif response_present and response_family == "flatten_transition" and response_fill < 0.0 and abs(response_exposure_post) <= 1e-9:
        klass = "immediate_flatten"
        reason = "response row immediately flattens the position"
    elif (
        warning_lead == "continuation_support_collapse"
        and warning_support_collapsed
        and response_present
        and response_expected > 1e-9
        and response_lead in {"", "none"}
        and not response_support_collapsed
    ):
        klass = "recovering_after_warning"
        reason = "next held row restores executed support after warning"
    elif (
        warning_expected <= 1e-9
        and response_present
        and response_expected > 1e-9
        and response_spread != "microstructure_kills_edge"
        and not response_support_collapsed
    ):
        klass = "recovering_after_warning"
        reason = "support is restored on the response row after a dead-support warning"
    elif (
        warning_lead == "continuation_support_collapse"
        and warning_support_collapsed
        and (
            not response_present
            or response_expected <= 1e-9
            or response_support_collapsed
            or response_survivability <= 0.05
            or response_spread == "microstructure_kills_edge"
            or response_family == "adverse_continuation"
        )
    ):
        klass = "confirmed_collapse" if response_present else "unresolved_warning"
        reason = (
            "continuation support stays dead on the response row"
            if response_present
            else "warning row exists but no response row is available"
        )
    elif (
        warning_expected <= 1e-9
        and response_present
        and (
            response_expected <= 1e-9
            or response_survivability <= 0.05
            or response_spread == "microstructure_kills_edge"
            or response_family == "adverse_continuation"
        )
    ):
        klass = "confirmed_collapse"
        reason = "dead-support warning is followed by another dead or hostile support row"
    elif str(warning.get("family", "")) == "adverse_continuation" or str(warning.get("lead_signal", "")) == "negative_efficiency_drift":
        klass = "entry_already_adverse"
        reason = "warning row is already adverse before recovery/collapse split"
    elif warning_lead == "continuation_support_collapse":
        klass = "unresolved_warning"
        reason = "warning row exists but response row does not cleanly restore or confirm collapse"
    else:
        klass = "uncertain_pair"
        reason = "episode does not fit the warning-response contrast surface"

    return {
        "entry_t": int(_safe_float(entry.get("sort_t"), 0.0)),
        "warning_t": int(_safe_float(warning.get("sort_t"), 0.0)),
        "warning_family": warning_family,
        "warning_lead_signal": warning_lead,
        "warning_expected_support_dead": warning_expected <= 1e-9,
        "warning_support_collapsed": warning_support_collapsed,
        "warning_survivability_score": warning_survivability,
        "response_t": int(_safe_float(response.get("sort_t"), 0.0)) if response_present else None,
        "response_family": response_family if response_present else None,
        "response_lead_signal": response_lead if response_present else None,
        "response_expected_support_dead": response_expected <= 1e-9 if response_present else None,
        "response_support_collapsed": response_support_collapsed if response_present else None,
        "response_survivability_score": response_survivability if response_present else None,
        "response_spread_regime": response_spread if response_present else None,
        "warning_response_class": klass,
        "warning_response_reason": reason,
    }


def main() -> None:
    args = parse_args()
    paths = _load_paths(args)

    records: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        lineage = _normalize_lineage(path)
        for episode in data.get("selected", []):
            rows = pd.DataFrame(episode.get("rows", []))
            row_pair = _classify_episode(rows)
            sig = _episode_signature(episode)
            realized = _safe_float(episode.get("realized_surplus_sum"))
            expected = _safe_float(episode.get("executed_expected_surplus_sum"))
            realized_eff = _safe_float(episode.get("realized_efficiency"))
            records.append(
                {
                    "artifact": path.name,
                    "artifact_lineage": lineage,
                    "input": str(data.get("input", "")),
                    "episode_id": int(_safe_float(episode.get("episode_id"), -1)),
                    "episode_type": str(episode.get("episode_type", "")),
                    "duration_rows": int(_safe_float(episode.get("duration_rows"), 0.0)),
                    "realized_surplus_sum": realized,
                    "executed_expected_surplus_sum": expected,
                    "realized_efficiency": realized_eff,
                    "entry_reason": str(episode.get("entry_reason", "")),
                    "exit_reason": str(episode.get("exit_reason", "")),
                    "entry_hazard_tightened_source": str(episode.get("entry_hazard_tightened_source", "")),
                    "exit_hazard_tightened_source": str(episode.get("exit_hazard_tightened_source", "")),
                    **sig,
                    **row_pair,
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise SystemExit("no selected episodes found in supplied summaries")

    class_counts = frame["warning_response_class"].value_counts(dropna=False).to_dict()
    lineage_counts = {
        klass: int(frame.loc[frame["warning_response_class"] == klass, "artifact_lineage"].nunique())
        for klass in sorted(frame["warning_response_class"].astype(str).unique())
    }
    class_metrics: dict[str, dict[str, Any]] = {}
    for klass, subset in frame.groupby("warning_response_class", sort=True):
        class_metrics[str(klass)] = {
            "episode_count": int(len(subset)),
            "artifact_lineage_count": int(subset["artifact_lineage"].nunique()),
            "realized_surplus_mean": float(subset["realized_surplus_sum"].mean()),
            "realized_efficiency_mean": float(subset["realized_efficiency"].mean()),
            "duration_rows_median": float(subset["duration_rows"].median()),
        }

    summary = {
        "artifact_count": int(frame["artifact"].nunique()),
        "artifact_lineage_count": int(frame["artifact_lineage"].nunique()),
        "episode_count": int(len(frame)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "artifact_lineage_per_class": lineage_counts,
        "class_metrics": class_metrics,
        "hold_gate_candidate": {
            "eligible_for_future_policy_candidate": bool(
                class_counts.get("confirmed_collapse", 0) >= 5
                and class_counts.get("immediate_flatten", 0) >= 5
                and class_counts.get("recovering_after_warning", 0) >= 5
                and lineage_counts.get("confirmed_collapse", 0) >= 2
                and lineage_counts.get("immediate_flatten", 0) >= 2
                and lineage_counts.get("recovering_after_warning", 0) >= 2
            ),
            "reason": "requires 5/5/5 class breadth and at least 2 lineages per core class",
        },
    }

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values(["warning_response_class", "artifact", "episode_id"], kind="stable").to_csv(out, index=False)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

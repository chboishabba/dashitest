#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        action="append",
        required=True,
        help="Debugger summary JSON from scripts/debug_nashi_trade_pairs.py; repeat for multiple artifacts",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional display label aligned by position with --summary-json",
    )
    parser.add_argument(
        "--prefix-rows",
        type=int,
        default=5,
        help="How many in-episode rows to compare from each selected episode",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which selected episode to compare from each summary JSON",
    )
    parser.add_argument(
        "--episode-selector",
        action="append",
        help="Optional per-artifact episode selection as label:index or basename:index; overrides --episode-index for matching artifacts",
    )
    parser.add_argument("--output-csv", help="Optional output path for the row-aligned comparison table")
    parser.add_argument("--summary-out", help="Optional output path for the JSON comparison summary")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _load_episode(
    path: Path,
    *,
    episode_index: int | None,
    episode_id: int | None,
    prefix_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected", [])
    if episode_id is not None:
        matches = [episode for episode in selected if int(_safe_float(episode.get("episode_id"), -1)) == int(episode_id)]
        if not matches:
            raise ValueError(f"{path} has no selected episode_id={episode_id}")
        episode = matches[0]
    else:
        if episode_index is None:
            raise ValueError("episode_index or episode_id is required")
        if episode_index >= len(selected):
            raise ValueError(f"{path} has only {len(selected)} selected episodes")
        episode = selected[episode_index]
    rows = pd.DataFrame(episode.get("rows", []))
    if rows.empty:
        raise ValueError(f"{path} selected episode has no rows")
    rows = rows[rows.get("in_episode", 0).astype(bool)].copy().reset_index(drop=True)
    rows.insert(0, "prefix_index", range(len(rows)))
    return episode, rows.head(prefix_rows)


def _parse_episode_selectors(raw_items: list[str] | None) -> dict[str, tuple[str, int]]:
    selectors: dict[str, tuple[str, int]] = {}
    for raw in raw_items or []:
        key, sep, value = raw.rpartition(":")
        if not sep or not key:
            raise SystemExit(f"invalid --episode-selector {raw!r}; expected label:index or label:id=3")
        if value.startswith("id="):
            selectors[key] = ("id", int(value.split("=", 1)[1]))
        elif value.startswith("index="):
            selectors[key] = ("index", int(value.split("=", 1)[1]))
        else:
            selectors[key] = ("index", int(value))
    return selectors


def _episode_signature(episode: dict[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    first_row = rows.iloc[0].to_dict() if not rows.empty else {}
    second_row = rows.iloc[1].to_dict() if len(rows) > 1 else {}
    third_row = rows.iloc[2].to_dict() if len(rows) > 2 else {}
    return {
        "episode_type": str(episode.get("episode_type")),
        "t_open": int(_safe_float(episode.get("t_open"))),
        "t_close": int(_safe_float(episode.get("t_close"))),
        "duration_rows": int(_safe_float(episode.get("duration_rows"))),
        "realized_surplus_sum": _safe_float(episode.get("realized_surplus_sum")),
        "executed_expected_surplus_sum": _safe_float(episode.get("executed_expected_surplus_sum")),
        "entry_reason": str(episode.get("entry_reason")),
        "exit_reason": str(episode.get("exit_reason")),
        "first_prefix_family": str(first_row.get("family", "")),
        "second_prefix_family": str(second_row.get("family", "")),
        "third_prefix_family": str(third_row.get("family", "")),
        "second_prefix_lead_signal": str(second_row.get("lead_signal", "")),
        "third_prefix_lead_signal": str(third_row.get("lead_signal", "")),
    }


def _derive_prefix_pattern(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"pattern_class": "empty", "pattern_reason": "no_rows"}

    row1 = rows.iloc[1].to_dict() if len(rows) > 1 else {}
    row2 = rows.iloc[2].to_dict() if len(rows) > 2 else {}

    row1_family = str(row1.get("family", ""))
    row2_family = str(row2.get("family", ""))
    row1_lead = str(row1.get("lead_signal", ""))
    row2_lead = str(row2.get("lead_signal", ""))
    row1_expected = _safe_float(row1.get("executed_expected_surplus"))
    row2_expected = _safe_float(row2.get("executed_expected_surplus"))
    row1_survivability = _safe_float(row1.get("proposed_survivability_score"))
    row2_survivability = _safe_float(row2.get("proposed_survivability_score"))
    row1_spread = str(row1.get("nashi_spread_regime", ""))
    row2_spread = str(row2.get("nashi_spread_regime", ""))

    if row1_family == "flatten_transition":
        return {
            "pattern_class": "immediate_flatten",
            "pattern_reason": "warning row is already flatten_transition",
        }
    if row1_family == "interior_softening" and row2_family == "adverse_continuation":
        return {
            "pattern_class": "confirmed_collapse",
            "pattern_reason": "warning row softens and next held row confirms adverse continuation",
        }
    if row1_family == "interior_softening" and row2_expected > 1e-9 and row2_lead in {"", "none"}:
        return {
            "pattern_class": "recoverable_softening",
            "pattern_reason": "warning row softens but next held row restores support",
        }
    if row1_lead == "continuation_support_collapse" and row2_lead == "continuation_support_collapse":
        return {
            "pattern_class": "confirmed_collapse",
            "pattern_reason": "support collapse persists across two held rows",
        }
    if row1_expected <= 1e-9 and row2_expected > 1e-9 and row2_spread != "microstructure_kills_edge":
        return {
            "pattern_class": "recovering_after_warning",
            "pattern_reason": "next held row restores executed expectation after a dead-support warning",
        }
    if row1_family == "adverse_continuation":
        return {
            "pattern_class": "entry_already_adverse",
            "pattern_reason": "episode is adverse from the entry row",
        }
    if row1_survivability <= 1e-9 and row2_survivability <= 1e-9 and row2_spread == "microstructure_kills_edge":
        return {
            "pattern_class": "confirmed_collapse",
            "pattern_reason": "survivability and spread regime both stay collapsed",
        }
    return {
        "pattern_class": "uncertain_prefix",
        "pattern_reason": "prefix does not yet separate recovery from failure cleanly",
    }


def _derive_row_pair_classifier(rows: pd.DataFrame) -> dict[str, Any]:
    if len(rows) < 2:
        return {
            "warning_response_class": "insufficient_prefix",
            "warning_response_reason": "need at least entry row plus warning row",
        }

    entry = rows.iloc[0].to_dict()
    warning = rows.iloc[1].to_dict()
    response = rows.iloc[2].to_dict() if len(rows) > 2 else {}

    warning_family = str(warning.get("family", ""))
    response_family = str(response.get("family", ""))
    warning_lead = str(warning.get("lead_signal", ""))
    response_lead = str(response.get("lead_signal", ""))
    warning_expected = _safe_float(warning.get("executed_expected_surplus"))
    response_expected = _safe_float(response.get("executed_expected_surplus"))
    warning_survivability = _safe_float(warning.get("proposed_survivability_score"))
    response_survivability = _safe_float(response.get("proposed_survivability_score"))
    warning_spread = str(warning.get("nashi_spread_regime", ""))
    response_spread = str(response.get("nashi_spread_regime", ""))
    warning_support_dead = warning_expected <= 1e-9
    response_present = len(rows) > 2
    response_support_dead = response_present and response_expected <= 1e-9
    response_restores_support = warning_support_dead and response_present and response_expected > 1e-9
    response_confirms_collapse = response_support_dead and response_lead == "continuation_support_collapse"
    response_flatten = response_family == "flatten_transition"
    warning_softening = warning_family == "interior_softening"

    if warning_family == "flatten_transition":
        response_class = "immediate_flatten"
        response_reason = "warning row is already terminal flatten"
    elif not response_present:
        response_class = "warning_without_response"
        response_reason = "warning row exists but no held response row is available"
    elif response_restores_support and response_spread != "microstructure_kills_edge":
        response_class = "recovering_after_warning"
        response_reason = "warning row loses support but next held row restores executed support"
    elif warning_softening and response_family == "adverse_continuation":
        response_class = "confirmed_collapse"
        response_reason = "warning row is followed by adverse continuation"
    elif warning_lead == "continuation_support_collapse" and response_confirms_collapse:
        response_class = "confirmed_collapse"
        response_reason = "continuation support stays dead across the response row"
    elif response_flatten:
        response_class = "immediate_flatten"
        response_reason = "response row is flatten transition before recovery can form"
    elif warning_family == "adverse_continuation":
        response_class = "entry_already_adverse"
        response_reason = "entry prefix is already adverse before a warning/recovery split"
    elif warning_softening:
        response_class = "warning_without_confirmation"
        response_reason = "warning row exists but response row does not cleanly restore or confirm collapse"
    else:
        response_class = "uncertain_pair"
        response_reason = "row pair does not fit the current continuation contrast surface"

    return {
        "entry_family": str(entry.get("family", "")),
        "warning_t": int(_safe_float(warning.get("sort_t"), 0.0)),
        "warning_family": warning_family,
        "warning_lead_signal": warning_lead,
        "warning_expected_support_dead": warning_support_dead,
        "warning_survivability_score": warning_survivability,
        "warning_spread_regime": warning_spread,
        "response_t": int(_safe_float(response.get("sort_t"), 0.0)) if response_present else None,
        "response_family": response_family,
        "response_lead_signal": response_lead,
        "response_expected_support_dead": response_support_dead,
        "response_survivability_score": response_survivability,
        "response_spread_regime": response_spread,
        "response_restores_support": response_restores_support,
        "response_confirms_collapse": response_confirms_collapse,
        "response_is_flatten": response_flatten,
        "response_present": response_present,
        "warning_response_class": response_class,
        "warning_response_reason": response_reason,
    }


def _contrast_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "artifact_count": len(records),
        "labels": [record["label"] for record in records],
        "episode_types": {record["label"]: record["signature"]["episode_type"] for record in records},
        "durations": {record["label"]: record["signature"]["duration_rows"] for record in records},
        "realized_surplus_sum": {
            record["label"]: record["signature"]["realized_surplus_sum"] for record in records
        },
        "prefix_pattern_class": {
            record["label"]: record["prefix_pattern"]["pattern_class"] for record in records
        },
        "prefix_pattern_reason": {
            record["label"]: record["prefix_pattern"]["pattern_reason"] for record in records
        },
        "warning_response_class": {
            record["label"]: record["row_pair_classifier"]["warning_response_class"] for record in records
        },
        "warning_response_reason": {
            record["label"]: record["row_pair_classifier"]["warning_response_reason"] for record in records
        },
        "first_three_family_chain": {
            record["label"]: [
                record["signature"]["first_prefix_family"],
                record["signature"]["second_prefix_family"],
                record["signature"]["third_prefix_family"],
            ]
            for record in records
        },
        "first_three_lead_chain": {
            record["label"]: [
                record["signature"]["second_prefix_lead_signal"],
                record["signature"]["third_prefix_lead_signal"],
            ]
            for record in records
        },
    }


def _transition_signature(rows: pd.DataFrame, limit: int = 3) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for _, row in rows.head(limit).iterrows():
        signature.append(
            {
                "prefix_index": int(_safe_float(row.get("prefix_index"), 0.0)),
                "sort_t": int(_safe_float(row.get("sort_t"), 0.0)),
                "family": str(row.get("family", "")),
                "lead_signal": str(row.get("lead_signal", "")),
                "continuation_candidate_active": bool(row.get("continuation_candidate_active", False)),
                "continuation_post_exec_active": bool(row.get("continuation_post_exec_active", False)),
                "continuation_transition_row": bool(row.get("continuation_transition_row", False)),
                "mw_reason": str(row.get("mw_reason", "")),
                "executed_expected_surplus": _safe_float(row.get("executed_expected_surplus")),
                "proposed_survivability_score": _safe_float(row.get("proposed_survivability_score")),
            }
        )
    return signature


def _breakpoint_summary(combined_rows: pd.DataFrame) -> dict[str, Any]:
    if combined_rows.empty:
        return {}
    watched = [
        "family",
        "lead_signal",
        "continuation_post_exec_active",
        "mw_reason",
        "executed_expected_surplus",
    ]
    out: dict[str, Any] = {}
    for prefix_index, subset in combined_rows.groupby("prefix_index", sort=True):
        for field in watched:
            if field not in subset.columns:
                continue
            values = [str(value) if field != "executed_expected_surplus" else f"{_safe_float(value):.12g}" for value in subset[field]]
            if len(set(values)) > 1:
                out = {
                    "first_divergence_prefix_index": int(prefix_index),
                    "field": field,
                    "values": {
                        str(label): (
                            _safe_float(value) if field == "executed_expected_surplus" else (
                                bool(value) if field == "continuation_post_exec_active" else str(value)
                            )
                        )
                        for label, value in zip(subset["label"], subset[field], strict=True)
                    },
                }
                return out
    return {"first_divergence_prefix_index": None}


def main() -> None:
    args = parse_args()
    paths = [Path(raw) for raw in args.summary_json]
    labels = list(args.label or [])
    if labels and len(labels) != len(paths):
        raise SystemExit("--label count must match --summary-json count when provided")
    if not labels:
        labels = [path.stem for path in paths]
    selectors = _parse_episode_selectors(args.episode_selector)

    records: list[dict[str, Any]] = []
    row_tables: list[pd.DataFrame] = []
    for label, path in zip(labels, paths, strict=True):
        selector_mode, selector_value = selectors.get(
            label,
            selectors.get(path.name, ("index", int(args.episode_index))),
        )
        episode, rows = _load_episode(
            path,
            episode_index=selector_value if selector_mode == "index" else None,
            episode_id=selector_value if selector_mode == "id" else None,
            prefix_rows=int(args.prefix_rows),
        )
        signature = _episode_signature(episode, rows)
        pattern = _derive_prefix_pattern(rows)
        row_pair_classifier = _derive_row_pair_classifier(rows)
        records.append(
            {
                "label": label,
                "path": str(path),
                "episode_index": int(_safe_float(episode.get("episode_id"), selector_value))
                if selector_mode == "id"
                else int(selector_value),
                "episode_selector_mode": selector_mode,
                "signature": signature,
                "prefix_pattern": pattern,
                "row_pair_classifier": row_pair_classifier,
                "transition_signature": _transition_signature(rows),
            }
        )
        table = rows.copy()
        table.insert(0, "label", label)
        table.insert(1, "episode_selector_mode", selector_mode)
        table.insert(2, "episode_selector_value", int(selector_value))
        row_tables.append(table)

    combined_rows = pd.concat(row_tables, ignore_index=True) if row_tables else pd.DataFrame()
    numeric_cols = [
        "edge",
        "edge_persistence",
        "edge_shock",
        "actionability",
        "microstructure_pressure",
        "hazard",
        "cost_survival_ratio",
        "executed_expected_surplus",
        "realized_surplus",
        "proposed_survivability_score",
    ]
    prefix_summary: list[dict[str, Any]] = []
    if not combined_rows.empty:
        for prefix_index, subset in combined_rows.groupby("prefix_index", sort=True):
            row: dict[str, Any] = {"prefix_index": int(prefix_index)}
            for column in numeric_cols:
                if column in subset.columns:
                    values = [_safe_float(value) for value in subset[column]]
                    row[column] = {label: _safe_float(value) for label, value in zip(subset["label"], values, strict=True)}
            for column in ("family", "lead_signal", "mw_reason", "nashi_spread_regime"):
                if column in subset.columns:
                    row[column] = {label: str(value) for label, value in zip(subset["label"], subset[column], strict=True)}
            prefix_summary.append(row)

    summary = {
        "artifacts": records,
        "contrast": _contrast_summary(records),
        "breakpoint_summary": _breakpoint_summary(combined_rows),
        "prefix_rows": prefix_summary,
    }

    if args.output_csv and not combined_rows.empty:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined_rows.to_csv(out, index=False)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

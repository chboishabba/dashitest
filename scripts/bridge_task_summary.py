#!/usr/bin/env python3
"""
Produce a markdown summary for Task B bridge runs (codec/DNA).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence


FIGURE_SUFFIXES = [
    ("bridge_mse", "Bridge MSE (observed)"),
    ("bridge_quotient", "Bridge MSE (quotient)"),
    ("bridge_tree_band_quotient", "Bridge leakage (tree-band quotient)"),
    ("bridge_prediction", "Representative prediction vs truth"),
]
OPERATOR_FIGURE_SUFFIXES = [
    ("operator_one_step", "Operator one-step predictions (first band)"),
    ("operator_rollout", "Operator rollout error norm"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize bridge-task JSON.")
    ap.add_argument("--json", type=Path, help="Bridge metrics JSON file.")
    ap.add_argument(
        "--pattern",
        type=str,
        default="outputs/bridge_metrics_*.json",
        help="Glob pattern to discover the newest bridge run.",
    )
    ap.add_argument(
        "--leakage-threshold",
        type=float,
        default=2.0,
        help="Leakage ratio threshold for Task B closure.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="Optional path to write the markdown snippet; defaults to stdout.",
    )
    return ap.parse_args()


def find_latest_json(pattern: str) -> Path:
    matches = sorted(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no runs match {pattern}")
    return matches[-1]


def load_metrics(json_path: Path) -> Mapping[str, float]:
    with json_path.open() as fh:
        return json.load(fh)


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3e}"


def ratio_or_na(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0.0:
        return None
    return numerator / denominator


def build_summary(json_path: Path, metrics: Mapping[str, float], threshold: float) -> str:
    base = json_path.with_suffix("")
    rbf_tree_leak = metrics.get("rbf_bridge_tree_band_q_mse")
    tree_tree_leak = metrics.get("tree_bridge_tree_band_q_mse")
    leak_ratio = ratio_or_na(rbf_tree_leak, tree_tree_leak)
    ratio_text = f"{leak_ratio:.2f}×" if leak_ratio is not None else "n/a"
    pass_flag = leak_ratio is not None and leak_ratio >= threshold
    lines = []
    lines.append(f"## Task B bridge summary ({json_path.stem})")
    lines.append("")
    lines.append("### Key metrics")
    lines.append(
        f"- Raw MSE: tree={format_value(metrics['tree_bridge_mse'])} vs rbf={format_value(metrics['rbf_bridge_mse'])}."
    )
    lines.append(
        f"- Quotient MSE: tree={format_value(metrics['tree_bridge_q_mse'])} vs rbf={format_value(metrics['rbf_bridge_q_mse'])}."
    )
    lines.append(
        f"- Leakage (tree-band) ratio: {ratio_text} "
        f"({'pass' if pass_flag else 'fail'} at threshold {threshold:.1f}×)."
    )
    samples = metrics.get("bridge_train_samples", 0) + metrics.get("bridge_test_samples", 0)
    lines.append(
        f"- Samples: {int(metrics.get('bridge_train_samples', 0))} train, "
        f"{int(metrics.get('bridge_test_samples', 0))} test "
        f"(total windows {int(metrics.get('bridge_windows', 0))})."
    )
    lines.append("")
    lines.append("### Figures")
    for suffix, description in FIGURE_SUFFIXES:
        fig = base.with_name(f"{base.name}_{suffix}.png")
        exists_note = "" if fig.exists() else " (missing)"
        lines.append(f"- `{fig}`: {description}.{exists_note}")

    op_one_step = metrics.get("operator_baseline_one_step_mse")
    if op_one_step is not None:
        op_rollout = metrics.get("operator_baseline_rollout_mse")
        rollout_steps = metrics.get("operator_baseline_rollout_steps", 0)
        alpha = metrics.get("operator_baseline_alpha")
        spectral = metrics.get("operator_baseline_spectral_norm")
        lines.append("")
        lines.append("### Operator baseline")
        lines.append(
            f"- One-step MSE: {format_value(op_one_step)}; rollout ({rollout_steps} steps) MSE: "
            f"{format_value(op_rollout)}."
        )
        if alpha is not None and spectral is not None:
            lines.append(
                f"- Contractivity: α={alpha:.2f}, spectral norm={format_value(spectral)}."
            )
        for suffix, description in OPERATOR_FIGURE_SUFFIXES:
            fig = base.with_name(f"{base.name}_{suffix}.png")
            exists_note = "" if fig.exists() else " (missing)"
            lines.append(f"- `{fig}`: {description}.{exists_note}")
    return "\n".join(lines)


def write_output(text: str, path: Optional[Path]) -> None:
    if path:
        path.write_text(text)
    else:
        print(text)


def main() -> None:
    args = parse_args()
    json_path = args.json or find_latest_json(args.pattern)
    metrics = load_metrics(json_path)
    summary = build_summary(json_path, metrics, args.leakage_threshold)
    write_output(summary, args.out)


if __name__ == "__main__":
    main()

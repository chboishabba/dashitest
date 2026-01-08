#!/usr/bin/env python3
"""
Generate the Codec Task A summary markdown snippet from a metrics JSON run.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


FIGURE_SUFFIXES = [
    ("rollout_mse", "raw rollout MSE curves for RBF vs Tree"),
    ("rollout_quotient", "subtree quotient MSE over rollout time"),
    ("rollout_tree_quotient", "tree-level quotient separation (depth-wise energy preservation)"),
    ("rollout_tree_band_quotient", "per-band quotient energy (leakage vs depth)"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Emit Codec Task A summary markdown.")
    ap.add_argument(
        "--json",
        type=Path,
        help="Run JSON (`outputs/tree_diffusion_metrics_*.json`).",
    )
    ap.add_argument(
        "--pattern",
        default="outputs/tree_diffusion_metrics_*.json",
        help="Glob pattern to discover the latest run JSON when --json is omitted.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="Optional file to write the markdown snippet; otherwise prints to stdout.",
    )
    return ap.parse_args()


def find_latest_json(pattern: str) -> Path:
    matches = sorted(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no JSON matches for pattern {pattern}")
    return matches[-1]


def load_metrics(json_path: Path) -> Mapping[str, float]:
    with json_path.open() as fh:
        return json.load(fh)


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3e}"


def ratio_description(tree_val: Optional[float], rbf_val: Optional[float]) -> Optional[str]:
    if tree_val and rbf_val:
        ratio = rbf_val / tree_val
        return f" (rbf/tree ≈ {ratio:.1f}×)"
    return None


def make_sample_lines(metrics: Mapping[str, float]) -> Sequence[str]:
    parts = []
    for key, label in (
        ("train_samples", "train"),
        ("test_samples", "test"),
        ("bridge_train_samples", "bridge train"),
        ("bridge_test_samples", "bridge test"),
    ):
        value = metrics.get(key)
        if value is not None:
            parts.append(f"{int(value):,} {label}")
    if not parts:
        return []
    return [", ".join(parts)]


def generate_summary(json_path: Path, metrics: Mapping[str, float]) -> str:
    base = json_path.stem
    fig_dir = json_path.parent

    lines = []
    lines.append(f"## Codec Task A summary ({base})")
    lines.append("")
    lines.append("### Key metrics")

    tree_rollout = metrics.get("tree_rollout_mse")
    rbf_rollout = metrics.get("rbf_rollout_mse")
    ratio = ratio_description(tree_rollout, rbf_rollout)
    lines.append(
        f"- Raw rollout MSE: tree={format_value(tree_rollout)} vs rbf={format_value(rbf_rollout)}"
        f"{ratio or ''}."
    )

    tree_bridge = metrics.get("tree_bridge_mse")
    rbf_bridge = metrics.get("rbf_bridge_mse")
    ratio_bridge = ratio_description(tree_bridge, rbf_bridge)
    tree_bridge_band = metrics.get("tree_bridge_tree_band_q_mse")
    rbf_bridge_band = metrics.get("rbf_bridge_tree_band_q_mse")
    lines.append(
        f"- Bridge error: tree={format_value(tree_bridge)} vs rbf={format_value(rbf_bridge)}"
        f"{ratio_bridge or ''}; tree-band leakage {format_value(tree_bridge_band)} vs {format_value(rbf_bridge_band)}."
    )

    tree_one_step = metrics.get("tree_one_step_mse")
    rbf_one_step = metrics.get("rbf_one_step_mse")
    if tree_one_step or rbf_one_step:
        lines.append(
            f"- One-step MSE: tree={format_value(tree_one_step)} vs rbf={format_value(rbf_one_step)}."
        )

    sample_lines = make_sample_lines(metrics)
    if sample_lines:
        lines.append(f"- Samples: {sample_lines[0]}.")

    lines.append("")
    lines.append("### Figures (required rollout diagnostics)")
    for suffix, description in FIGURE_SUFFIXES:
        figure_path = fig_dir / f"{base}_{suffix}.png"
        exists_note = "" if figure_path.exists() else " (missing)"
        lines.append(f"- `{figure_path}`: {description}.{exists_note}")

    return "\n".join(lines)


def write_output(text: str, out_path: Optional[Path]) -> None:
    if out_path:
        out_path.write_text(text)
    else:
        print(text)


def main() -> None:
    args = parse_args()
    json_path = args.json or find_latest_json(args.pattern)
    metrics = load_metrics(json_path)
    summary = generate_summary(json_path, metrics)
    write_output(summary, args.out)


if __name__ == "__main__":
    main()

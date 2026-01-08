#!/usr/bin/env python3
"""
Run OperatorLearner v0 on a band-energy sequence and report contractive dynamics metrics.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from dashilearn.operator_learner import OperatorLearner


FIGURE_SUFFIXES = [
    ("operator_one_step", "One-step predictions (first band square root)"),
    ("operator_rollout", "Rollout normed error curve"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train OperatorLearner on E_seq and emit metrics.")
    ap.add_argument(
        "--energy-seq",
        type=Path,
        required=True,
        help="Path to `E_seq.npy` (band-energy sequence).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/operator_metrics.json"),
        help="Base path for the operator metrics JSON (timestamp appended).",
    )
    ap.add_argument(
        "--train-steps",
        type=int,
        default=400,
        help="Number of steps to allocate to training (consumes `train_steps + 1` rows).",
    )
    ap.add_argument(
        "--val-steps",
        type=int,
        default=200,
        help="Number of steps to reserve for validation/rollout.",
    )
    ap.add_argument(
        "--rollout-steps",
        type=int,
        default=20,
        help="Steps to roll out from the start of the validation segment.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Contractivity constant for the linear map (spectral norm <= alpha).",
    )
    ap.add_argument(
        "--bridge-json",
        type=Path,
        help="Optional Task B bridge metrics JSON to annotate with operator baselines.",
    )
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Write diagnostic plots next to the metrics JSON.",
    )
    return ap.parse_args()


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error across all bands."""
    return float(np.mean(np.square(pred - target)))


def _ensure_split(seq: np.ndarray, train_steps: int, val_steps: int) -> tuple[np.ndarray, np.ndarray]:
    if seq.ndim != 2:
        raise ValueError("E_seq must be 2D (T, B)")
    if seq.shape[0] < train_steps + val_steps + 1:
        raise ValueError(
            f"Not enough steps ({seq.shape[0]}) for train={train_steps} + val={val_steps} + 1"
        )
    train_seq = seq[: train_steps + 1]
    val_seq = seq[train_steps : train_steps + val_steps + 1]
    return train_seq, val_seq


def _plot_one_step(pred: np.ndarray, truth: np.ndarray, base: Path) -> None:
    history = min(50, pred.shape[0])
    band = 0
    fig, ax = plt.subplots(figsize=(5.5, 3))
    idx = np.arange(pred.shape[0])
    ax.plot(idx[-history:], truth[-history:, band], label="truth", linewidth=1.5)
    ax.plot(idx[-history:], pred[-history:, band], label="operator", linewidth=1.5)
    ax.set_xlabel("validation step")
    ax.set_ylabel(f"band {band} energy")
    ax.set_title("Operator one-step predictions")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":")
    out_path = base.with_name(f"{base.name}_operator_one_step.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_rollout(rollout_seq: np.ndarray, target_seq: np.ndarray, base: Path) -> None:
    preds = rollout_seq[1:]
    targets = target_seq[1 : len(preds) + 1]
    errors = np.linalg.norm(preds - targets, axis=1)
    steps = np.arange(1, errors.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot(steps, errors, marker="o", markersize=3)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("norm error")
    ax.set_title("Operator rollout error")
    ax.grid(True, linestyle=":")
    out_path = base.with_name(f"{base.name}_operator_rollout.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def _annotate_bridge_json(bridge_json: Path, summary_metrics: Mapping[str, float]) -> None:
    if not bridge_json.exists():
        raise FileNotFoundError(f"bridge JSON {bridge_json} not found")
    with bridge_json.open() as fh:
        existing = json.load(fh)
    existing.update({k: v for k, v in summary_metrics.items() if isinstance(v, (float, int))})
    with bridge_json.open("w") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
    print(f"Annotated {bridge_json} with operator metrics")


def main() -> None:
    args = parse_args()
    seq = np.load(args.energy_seq)
    train_seq, val_seq = _ensure_split(seq, args.train_steps, args.val_steps)
    learner = OperatorLearner(alpha=args.alpha)
    learner.fit(train_seq)
    val_states = val_seq[:-1]
    val_targets = val_seq[1:]
    predictions = learner.predict_batch(val_states)
    one_step_mse = mse(predictions, val_targets)
    rollout_steps = min(args.rollout_steps, val_seq.shape[0] - 1)
    rollout_sequence = learner.rollout(val_seq[0], rollout_steps)
    rollout_truth = val_seq[: rollout_steps + 1]
    rollout_mse = mse(rollout_sequence[1:], rollout_truth[1:])
    state = learner.state_dict()
    metrics = {
        "operator_baseline_one_step_mse": one_step_mse,
        "operator_baseline_rollout_mse": rollout_mse,
        "operator_baseline_rollout_steps": rollout_steps,
        "operator_baseline_alpha": state["alpha"],
        "operator_baseline_spectral_norm": state["spectral_norm"],
        "operator_baseline_raw_spectral_norm": state.get("raw_spectral_norm"),
        "operator_baseline_train_loss": learner.train_loss,
        "operator_baseline_train_steps": args.train_steps,
        "operator_baseline_val_steps": args.val_steps,
        "operator_baseline_train_samples": train_seq.shape[0] - 1,
        "operator_baseline_val_samples": val_seq.shape[0] - 1,
        "operator_baseline_rollout_start_idx": args.train_steps,
        "operator_baseline_W": state["W"],
        "operator_baseline_b": state["b"],
    }
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")
    with out_path.open("w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(f"Saved metrics to {out_path}")
    if args.plots:
        prefix = out_path.with_suffix("")
        _plot_one_step(predictions, val_targets, prefix)
        _plot_rollout(rollout_sequence, rollout_truth, prefix)
    if args.bridge_json:
        summary_metrics = {
            k: metrics[k]
            for k in (
                "operator_baseline_one_step_mse",
                "operator_baseline_rollout_mse",
                "operator_baseline_rollout_steps",
                "operator_baseline_alpha",
                "operator_baseline_spectral_norm",
            )
            if k in metrics
        }
        _annotate_bridge_json(args.bridge_json, summary_metrics)


if __name__ == "__main__":
    main()

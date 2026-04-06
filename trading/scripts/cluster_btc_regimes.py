#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURES = [
    "mean_abs_edge",
    "mean_edge_persistence",
    "mean_edge_shock",
    "mean_actionability",
    "high_microstructure_share",
    "drag_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Window-stats CSV from mine_targeted_recovery_windows.py.")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters.")
    parser.add_argument("--max-iter", type=int, default=25, help="Maximum Lloyd iterations.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV path for per-window cluster assignments.")
    parser.add_argument("--summary-out", type=Path, help="Optional JSON path for cluster summary.")
    return parser.parse_args()


def _zscore(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    norms: dict[str, dict[str, float]] = {}
    matrix = []
    for column in columns:
        series = pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if std <= 1e-9:
            std = 1.0
        norms[column] = {"mean": mean, "std": std}
        matrix.append(((series - mean) / std).to_numpy(dtype=float))
    return np.column_stack(matrix), norms


def _init_centroids(matrix: np.ndarray, k: int) -> np.ndarray:
    if len(matrix) == 0:
        return np.zeros((0, 0), dtype=float)
    if k <= 1:
        return matrix[[0]].copy()
    order = np.argsort(matrix.sum(axis=1))
    picks = np.linspace(0, len(order) - 1, num=min(k, len(order)), dtype=int)
    return matrix[order[picks]].copy()


def _kmeans(matrix: np.ndarray, k: int, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    if len(matrix) == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, 0), dtype=float)
    k = max(1, min(int(k), len(matrix)))
    centroids = _init_centroids(matrix, k)
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(max(1, int(max_iter))):
        distances = np.sqrt(((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                centroids[idx] = matrix[mask].mean(axis=0)
    return labels, centroids


def _cluster_label(summary: dict[str, Any]) -> str:
    if summary["immediate_flatten_ratio"] >= 0.75 and summary["drag_ratio_mean"] >= 0.5:
        return "failure_heavy"
    if summary["targeted_keep_ratio"] > 0.0 or summary["recovery_positive_ratio"] > 0.0:
        return "recovery_capable"
    return "mixed_fragile"


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    if frame.empty:
        raise SystemExit("input CSV has no rows")

    matrix, norms = _zscore(frame, FEATURES)
    labels, centroids = _kmeans(matrix, k=int(args.k), max_iter=int(args.max_iter))
    out = frame.copy()
    out["cluster_id"] = labels

    cluster_summaries: list[dict[str, Any]] = []
    for cluster_id, subset in out.groupby("cluster_id", sort=True):
        centroid = centroids[int(cluster_id)]
        summary = {
            "cluster_id": int(cluster_id),
            "window_count": int(len(subset)),
            "targeted_keep_ratio": float(subset["targeted_keep"].mean()) if "targeted_keep" in subset.columns else 0.0,
            "recovery_positive_ratio": float((subset["recovering_after_warning_count"] > 0).mean()),
            "immediate_flatten_ratio": float(
                (subset["immediate_flatten_count"] / subset["warning_count"].clip(lower=1)).mean()
            ),
            "confirmed_collapse_ratio": float(
                (subset["confirmed_collapse_count"] / subset["warning_count"].clip(lower=1)).mean()
            ),
            "drag_ratio_mean": float(pd.to_numeric(subset["drag_ratio"], errors="coerce").fillna(0.0).mean()),
            "feature_centroid_z": {FEATURES[idx]: float(centroid[idx]) for idx in range(len(FEATURES))},
        }
        summary["cluster_label"] = _cluster_label(summary)
        cluster_summaries.append(summary)

    cluster_summaries.sort(key=lambda row: row["cluster_id"])
    summary = {
        "input_csv": str(args.input_csv),
        "window_count": int(len(out)),
        "k": int(len(cluster_summaries)),
        "features": FEATURES,
        "normalization": norms,
        "clusters": cluster_summaries,
    }

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_csv, index=False)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

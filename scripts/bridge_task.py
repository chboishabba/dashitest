#!/usr/bin/env python3
"""
Build the bridge dataset from band-energy sequences and run Task B KRR comparisons.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge-task evaluation on band energies.")
    ap.add_argument("--energy-seq", type=Path, required=True, help="Path to E_seq.npy.")
    ap.add_argument("--bridge-task-T", type=int, default=50, help="Total horizon (T).")
    ap.add_argument("--train", type=int, default=200, help="Number of train windows.")
    ap.add_argument("--test", type=int, default=100, help="Number of test windows.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible splits.")
    ap.add_argument("--rbf-ls", type=float, default=1.0, help="Lengthscale for RBF kernel.")
    ap.add_argument("--tree-ls", type=float, default=1.0, help="Lengthscale for tree kernel.")
    ap.add_argument("--reg", type=float, default=1e-3, help="Regularization weight.")
    ap.add_argument(
        "--quotient-fn",
        type=str,
        default=None,
        help="Optional module:callable that maps energies -> quotient vector.",
    )
    ap.add_argument(
        "--leakage-fn",
        type=str,
        default=None,
        help="Optional module:callable that maps energies -> leakage descriptor.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/bridge_metrics.json"),
        help="Destination JSON for bridge metrics (timestamp added).",
    )
    ap.add_argument("--plots", action="store_true", help="Write the bridge comparison plots.")
    return ap.parse_args()


def load_transform(path: Optional[str]) -> Callable[[np.ndarray], np.ndarray]:
    if path is None:
        return lambda x: x
    if ":" not in path:
        raise ValueError("transform must be in module:callable format")
    module_name, attr_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


def flatten_sequence(seq: np.ndarray) -> np.ndarray:
    if seq.ndim == 2:
        return seq.astype(float)
    first = seq.shape[0]
    return seq.reshape(first, -1).astype(float)


def build_bridge_sequence(
    energies: np.ndarray,
    quotient: np.ndarray,
    leakage: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if horizon < 2:
        raise ValueError("bridge-task horizon must be >= 2")
    total = energies.shape[0]
    if total <= horizon:
        raise ValueError("sequence too short for the requested bridge horizon")
    midpoint = horizon // 2
    windows = total - horizon
    raw_pairs = []
    tree_pairs = []
    targets = []
    targets_q = []
    targets_leak = []
    for start in range(windows):
        end = start + horizon
        raw_pairs.append(np.concatenate([energies[start], energies[end]]))
        tree_pairs.append(np.concatenate([quotient[start], quotient[end]]))
        target_index = start + midpoint
        targets.append(energies[target_index])
        targets_q.append(quotient[target_index])
        targets_leak.append(leakage[target_index])
    return (
        np.stack(raw_pairs),
        np.stack(tree_pairs),
        np.stack(targets),
        np.stack(targets_q),
        np.stack(targets_leak),
    )


def split_windows(
    X_raw: np.ndarray, X_tree: np.ndarray, Y: np.ndarray, Y_q: np.ndarray, Y_leak: np.ndarray, train: int, test: int
) -> tuple[np.ndarray, ...]:
    windows = X_raw.shape[0]
    if windows < 2:
        raise ValueError("need at least two bridge windows")
    train = max(1, min(train, windows - 1))
    remaining = windows - train
    test = max(1, min(test, remaining))
    end = train + test
    return (
        X_raw[:train],
        X_raw[train:end],
        X_tree[:train],
        X_tree[train:end],
        Y[:train],
        Y[train:end],
        Y_q[:train],
        Y_q[train:end],
        Y_leak[:train],
        Y_leak[train:end],
        train,
        test,
        windows,
    )


def rbf_kernel(X: np.ndarray, Y: np.ndarray, lengthscale: float) -> np.ndarray:
    x2 = np.sum(X ** 2, axis=1)[:, None]
    y2 = np.sum(Y ** 2, axis=1)[None, :]
    d2 = x2 + y2 - 2.0 * (X @ Y.T)
    return np.exp(-d2 / (2.0 * lengthscale ** 2))


def krr_fit(X: np.ndarray, Y: np.ndarray, lengthscale: float, reg: float) -> dict:
    K = rbf_kernel(X, X, lengthscale)
    n = K.shape[0]
    alpha = np.linalg.solve(K + reg * np.eye(n), Y)
    return {"X": X, "alpha": alpha, "lengthscale": lengthscale}


def krr_predict(model: dict, X: np.ndarray) -> np.ndarray:
    K = rbf_kernel(X, model["X"], model["lengthscale"])
    return K @ model["alpha"]


def ridge_fit(X: np.ndarray, Y: np.ndarray, reg: float) -> np.ndarray:
    gram = X.T @ X
    n = gram.shape[0]
    rhs = X.T @ Y
    return np.linalg.solve(gram + reg * np.eye(n), rhs)


def ridge_predict(coeff: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ coeff


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def plot_bar(prefix: Path, suffix: str, rbf_value: float, tree_value: float, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([0, 1], [rbf_value, tree_value], color=["tab:blue", "tab:orange"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RBF", "Tree"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for x, value in zip([0, 1], [rbf_value, tree_value]):
        ax.text(x, value, f"{value:.2e}", ha="center", va="bottom")
    plt.tight_layout()
    out_path = prefix.with_name(f"{prefix.name}_{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_prediction(prefix: Path, Y_true: np.ndarray, rbf_pred: np.ndarray, tree_pred: np.ndarray) -> None:
    if Y_true.shape[0] == 0:
        return
    true_vector = Y_true[0]
    rbf_vector = rbf_pred[0]
    tree_vector = tree_pred[0]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    data = [
        (true_vector, "Target"),
        (rbf_vector, "RBF"),
        (tree_vector, "Tree"),
    ]
    for ax, (vec, label) in zip(axes, data):
        ax.plot(vec, linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("band")
        ax.set_ylabel("energy")
    plt.tight_layout()
    out_path = prefix.with_name(f"{prefix.name}_bridge_prediction.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    energies = flatten_sequence(np.load(args.energy_seq))
    quotient_fn = load_transform(args.quotient_fn)
    leakage_fn = load_transform(args.leakage_fn)
    quotient_seq = np.stack([quotient_fn(vec) for vec in energies])
    leakage_seq = np.stack([leakage_fn(vec) for vec in energies])
    (
        X_raw,
        X_tree,
        Y,
        Y_q,
        Y_leak,
    ) = build_bridge_sequence(energies, quotient_seq, leakage_seq, args.bridge_task_T)
    (
        X_raw_train,
        X_raw_test,
        X_tree_train,
        X_tree_test,
        Y_train,
        Y_test,
        Y_q_train,
        Y_q_test,
        Y_leak_train,
        Y_leak_test,
        bridge_train,
        bridge_test,
        bridge_windows,
    ) = split_windows(X_raw, X_tree, Y, Y_q, Y_leak, args.train, args.test)
    ridge_coeff = ridge_fit(X_raw_train, Y_train, args.reg)
    rbf_model = krr_fit(X_raw_train, Y_train, args.rbf_ls, args.reg)
    tree_model = krr_fit(X_tree_train, Y_train, args.tree_ls, args.reg)
    ridge_pred = ridge_predict(ridge_coeff, X_raw_test)
    rbf_bridge_pred = krr_predict(rbf_model, X_raw_test)
    tree_bridge_pred = krr_predict(tree_model, X_tree_test)
    rbf_bridge_q_pred = np.stack([quotient_fn(vec) for vec in rbf_bridge_pred])
    tree_bridge_q_pred = np.stack([quotient_fn(vec) for vec in tree_bridge_pred])
    rbf_bridge_leak_pred = np.stack([leakage_fn(vec) for vec in rbf_bridge_pred])
    tree_bridge_leak_pred = np.stack([leakage_fn(vec) for vec in tree_bridge_pred])
    ridge_mse = mse(ridge_pred, Y_test)
    metrics = {
        "rbf_bridge_mse": mse(rbf_bridge_pred, Y_test),
        "tree_bridge_mse": mse(tree_bridge_pred, Y_test),
        "rbf_bridge_q_mse": mse(rbf_bridge_q_pred, Y_q_test),
        "tree_bridge_q_mse": mse(tree_bridge_q_pred, Y_q_test),
        "rbf_bridge_tree_band_q_mse": mse(rbf_bridge_leak_pred, Y_leak_test),
        "tree_bridge_tree_band_q_mse": mse(tree_bridge_leak_pred, Y_leak_test),
        "bridge_windows": bridge_windows,
        "bridge_train_samples": bridge_train,
        "bridge_test_samples": bridge_test,
    }
    print("Bridge metrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.6e}" if isinstance(metrics[key], float) else f"  {key}: {metrics[key]}")
    print(f"  ridge_bridge_mse: {ridge_mse:.6e}")
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")
    with out_path.open("w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(f"Saved {out_path}")
    if args.plots:
        plot_prefix = out_path.with_suffix("")
        plot_bar(plot_prefix, "bridge_mse", metrics["rbf_bridge_mse"], metrics["tree_bridge_mse"], "MSE", "Bridge MSE (observed)")
        plot_bar(plot_prefix, "bridge_quotient", metrics["rbf_bridge_q_mse"], metrics["tree_bridge_q_mse"], "MSE", "Bridge MSE (quotient)")
        plot_bar(plot_prefix, "bridge_tree_band_quotient", metrics["rbf_bridge_tree_band_q_mse"], metrics["tree_bridge_tree_band_q_mse"], "MSE", "Bridge Leakage (tree band)")
        plot_prediction(plot_prefix, Y_test, rbf_bridge_pred, tree_bridge_pred)


if __name__ == "__main__":
    main()

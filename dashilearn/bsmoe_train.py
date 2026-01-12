"""
bsmoe_train.py
--------------
Block-sparse MoE training demo:
  - Gate selects active tiles (block-level)
  - Tile masks are derived from per-output gate activity
  - Dense int8 matmul microkernel runs only on active tiles
  - Emit once per block
"""

import argparse
import time
import os
import sys
import math
import json
import ctypes
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Ensure repo root is on sys.path so top-level Vulkan helpers import reliably.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from vulkan_compute.frame_capture import VulkanFrameCapture
except Exception:
    VulkanFrameCapture = None


def gate_prob_for_tile_density(tile_active, tile):
    if tile_active <= 0.0:
        return 0.0
    if tile_active >= 1.0:
        return 1.0
    return 1.0 - (1.0 - tile_active) ** (1.0 / (tile * tile))


def tiles_from_gate_mask(gate_mask, tile=32):
    M, N = gate_mask.shape
    tiles = np.zeros(((M + tile - 1) // tile, (N + tile - 1) // tile), dtype=bool)
    for ti in range(tiles.shape[0]):
        for tj in range(tiles.shape[1]):
            i0, j0 = ti * tile, tj * tile
            i1, j1 = min(i0 + tile, M), min(j0 + tile, N)
            tiles[ti, tj] = np.any(gate_mask[i0:i1, j0:j1])
    return tiles


SHEET_OUT_PATH = Path(__file__).with_name("sheet_energy.npy")
VULKAN_CAPTURE_THRESHOLD = 128
DEFAULT_GATE_DENSITY_THRESHOLD = 0.5
DEFAULT_GATEDENSITY_BINS: int | None = None
DEFAULT_ALTERNATION_INTERVAL = 4
PLAN_HIT_EXPERIMENT_DEFAULT_BLOCK = 6
PLAN_HIT_EXPERIMENT_DEFAULT_PERMS = 256
DEFAULT_PLAN_STABLE_LENGTH = 4
DEFAULT_CACHE_HIT_BINS: int | None = None
DEFAULT_PHASE3_RADIAL_BINS = 4


def _parse_bool_flag(value: str) -> bool:
    return value.lower() in {"1", "true", "t", "yes", "y"}


def tile_energy_map(C, plan):
    energy = np.zeros(plan.tile_grid_shape, dtype=np.float32)
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        row = i0 // plan.tile
        col = j0 // plan.tile
        block = C[i0:i1, j0:j1].astype(np.float32)
        energy[row, col] = float(np.sum(block * block))
    return energy


def dump_sheet_energy(energy, path=SHEET_OUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, energy)
    tmp.replace(path)


def _sheet_values_for_capture(
    sheet_energy: np.ndarray,
    *,
    epoch: int,
    gate_density: float,
    sheet_h: int,
    sheet_w: int,
) -> np.ndarray:
    if sheet_h <= 0 or sheet_w <= 0:
        return np.zeros_like(sheet_energy, dtype=np.float32)
    if sheet_energy.size == 0:
        return np.zeros((sheet_h, sheet_w), dtype=np.float32)
    sheet = np.zeros((sheet_h, sheet_w), dtype=np.float32)
    i, j = _semantic_tile_coord(epoch, sheet_h, sheet_w)
    sheet[i, j] = np.clip(gate_density, 0.0, 1.0)
    return sheet


def _semantic_tile_coord(epoch: int, sheet_h: int, sheet_w: int) -> tuple[int, int]:
    if sheet_h <= 0 or sheet_w <= 0:
        return 0, 0
    i = epoch % sheet_h
    j = (epoch // sheet_h) % sheet_w
    return i, j


def _tile_block_mean(frame: np.ndarray, tile_i: int, tile_j: int, block_px: int) -> float:
    height, width = frame.shape[:2]
    y0 = tile_i * block_px
    y1 = min(y0 + block_px, height)
    x0 = tile_j * block_px
    x1 = min(x0 + block_px, width)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    return float(frame[y0:y1, x0:x1].mean())


def _plan_hit_tile_stats(frame: np.ndarray, sheet_h: int, sheet_w: int, block_px: int):
    height, width = frame.shape[:2]
    tile_means = np.zeros((sheet_h, sheet_w), dtype=np.float32)
    tile_fracs = np.zeros((sheet_h, sheet_w), dtype=np.float32)
    thr = VULKAN_CAPTURE_THRESHOLD
    for i in range(sheet_h):
        y0 = i * block_px
        y1 = min(y0 + block_px, height)
        if y1 <= y0:
            continue
        for j in range(sheet_w):
            x0 = j * block_px
            x1 = min(x0 + block_px, width)
            if x1 <= x0:
                continue
            patch = frame[y0:y1, x0:x1].astype(np.float32)
            if patch.size == 0:
                continue
            tile_means[i, j] = float(patch.mean())
            tile_fracs[i, j] = float((patch > thr).mean())
    return tile_means, tile_fracs


def _observer_candidate_features(
    tile_means_list: list[np.ndarray],
    tile_fracs_list: list[np.ndarray],
    observer_class: str,
) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    if not tile_means_list:
        return candidates
    if observer_class in ("scalar", "corr"):
        mean_stack = np.vstack([tm.reshape(-1) for tm in tile_means_list])
        frac_stack = np.vstack([tf.reshape(-1) for tf in tile_fracs_list])
        for idx in range(mean_stack.shape[1]):
            candidates.append((f"mean_{idx}", mean_stack[:, idx]))
        for idx in range(frac_stack.shape[1]):
            candidates.append((f"frac_{idx}", frac_stack[:, idx]))
    if observer_class == "corr":
        corr_feats = _correlation_features(tile_means_list)
        for name, values in corr_feats.items():
            candidates.append((name, values))
    return candidates


def _grayscale_frames_from_observations(observations: list[dict]) -> np.ndarray:
    if not observations:
        return np.zeros((0, 1, 0, 0), dtype=np.float32)
    frames = np.stack(
        [np.asarray(obs["frame"], dtype=np.float32) for obs in observations], axis=0
    )
    if frames.ndim == 4:
        frames = frames.mean(axis=-1)
    if frames.ndim == 3:
        frames = frames[:, None, :, :]
    return frames.astype(np.float32, copy=False)


def _tiny_cnn_init(rng: np.random.Generator, out_channels=4, kernel_size=3, num_classes=2) -> dict:
    in_channels = 1
    conv_scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
    conv = rng.normal(0.0, conv_scale, size=(out_channels, in_channels, kernel_size, kernel_size))
    fc_scale = np.sqrt(1.0 / out_channels)
    fc = rng.normal(0.0, fc_scale, size=(num_classes, out_channels))
    return {
        "conv": np.asarray(conv, dtype=np.float32),
        "conv_bias": np.zeros(out_channels, dtype=np.float32),
        "fc": np.asarray(fc, dtype=np.float32),
        "fc_bias": np.zeros(num_classes, dtype=np.float32),
    }


def _conv_forward(inputs: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad = kernel.shape[2] // 2
    padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    patches = sliding_window_view(padded, (kernel.shape[2], kernel.shape[3]), axis=(2, 3))
    B, C, H, W = inputs.shape
    patches = patches.reshape(B, C, H, W, kernel.shape[2], kernel.shape[3])
    patches = patches.transpose(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(B, H, W, C * kernel.shape[2] * kernel.shape[3])
    kernel_flat = kernel.reshape(kernel.shape[0], -1)
    conv = patches @ kernel_flat.T
    conv = np.transpose(conv, (0, 3, 1, 2))
    conv += bias.reshape(1, -1, 1, 1)
    return conv, patches


def _tiny_cnn_forward(inputs: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    conv_out, patches = _conv_forward(inputs, params["conv"], params["conv_bias"])
    relu = np.maximum(conv_out, 0.0)
    pooled = relu.mean(axis=(2, 3))
    logits = pooled @ params["fc"].T + params["fc_bias"]
    cache = {
        "patches": patches,
        "pooled": pooled,
        "relu_mask": relu > 0.0,
        "spatial_size": max(1, relu.shape[2] * relu.shape[3]),
    }
    return logits, cache


def _tiny_cnn_loss_and_grads(
    logits: np.ndarray,
    label_indices: np.ndarray,
    cache: dict,
    params: dict,
) -> tuple[float, dict]:
    B = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(B), label_indices] = 1.0
    loss = -np.sum(one_hot * np.log(np.clip(probs, 1e-9, 1.0))) / B
    d_logits = (probs - one_hot) / B
    d_fc = d_logits.T @ cache["pooled"]
    d_fc_bias = d_logits.sum(axis=0)
    d_pooled = d_logits @ params["fc"]
    d_relu = d_pooled[:, :, None, None] / cache["spatial_size"]
    relu_mask = cache["relu_mask"].astype(np.float32)
    d_conv_out = d_relu * relu_mask
    patches = cache["patches"]
    d_kernel_flat = np.einsum("bhwm,bhwo->om", patches, d_conv_out.transpose(0, 2, 3, 1))
    grads = {
        "conv": d_kernel_flat.T.reshape(params["conv"].shape),
        "conv_bias": d_conv_out.sum(axis=(0, 2, 3)),
        "fc": d_fc,
        "fc_bias": d_fc_bias,
    }
    return loss, grads


def _train_eval_tiny_cnn(
    frames: np.ndarray,
    label_indices: np.ndarray,
    num_classes: int,
    rng: np.random.Generator,
    steps: int = 120,
    lr: float = 0.35,
) -> float:
    params = _tiny_cnn_init(rng, num_classes=num_classes)
    for _ in range(steps):
        logits, cache = _tiny_cnn_forward(frames, params)
        _, grads = _tiny_cnn_loss_and_grads(logits, label_indices, cache, params)
        params["conv"] -= lr * grads["conv"]
        params["conv_bias"] -= lr * grads["conv_bias"]
        params["fc"] -= lr * grads["fc"]
        params["fc_bias"] -= lr * grads["fc_bias"]
    logits, _ = _tiny_cnn_forward(frames, params)
    preds = np.argmax(logits, axis=1)
    return float((preds == label_indices).mean())


def _correlation_features(tile_means_list: list[np.ndarray]) -> dict[str, np.ndarray]:
    hor, vert, grad, lap = [], [], [], []
    for tm in tile_means_list:
        hor.append(_tile_corr(tm[:, :-1], tm[:, 1:]))
        vert.append(_tile_corr(tm[:-1, :], tm[1:, :]))
        grad.append(_tile_gradient_energy(tm))
        lap.append(_tile_laplacian_energy(tm))
    return {
        "hor_corr": np.array(hor, dtype=np.float32),
        "vert_corr": np.array(vert, dtype=np.float32),
        "grad_energy": np.array(grad, dtype=np.float32),
        "lap_energy": np.array(lap, dtype=np.float32),
    }


def _tile_corr(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return 0.0
    a = A.reshape(-1).astype(np.float32)
    b = B.reshape(-1).astype(np.float32)
    a_mean = a.mean()
    b_mean = b.mean()
    num = np.mean((a - a_mean) * (b - b_mean))
    denom = np.sqrt(np.mean((a - a_mean) ** 2) * np.mean((b - b_mean) ** 2))
    return float(num / denom) if denom > 1e-6 else 0.0


def _tile_gradient_energy(tile_means: np.ndarray) -> float:
    h, w = tile_means.shape
    energy = 0.0
    count = 0
    if w > 1:
        diff = tile_means[:, :-1] - tile_means[:, 1:]
        energy += float(np.mean(diff * diff))
        count += 1
    if h > 1:
        diff = tile_means[:-1, :] - tile_means[1:, :]
        energy += float(np.mean(diff * diff))
        count += 1
    return float(energy / count) if count else 0.0


def _tile_laplacian_energy(tile_means: np.ndarray) -> float:
    padded = np.pad(tile_means, 1, mode="reflect")
    center = padded[1:-1, 1:-1]
    neighbors = (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
    )
    lap = 4 * center - neighbors
    return float(np.mean(lap * lap))


def _regime_label_for_mode(
    mode: str,
    *,
    gate_density: float,
    epoch: int,
    plan_hit: bool,
    gate_density_threshold: float,
    alternation_interval: int,
    gate_density_bins: int | None,
    plan_stable_length: int,
    stable_run_len: int,
    cache_hit_fraction: float,
    cache_hit_bins: int | None,
) -> int:
    if mode == "plan-hit":
        return 1 if plan_hit else 0
    if mode == "gate-density":
        if gate_density_bins and gate_density_bins > 1:
            idx = int(gate_density * gate_density_bins)
            idx = max(0, min(gate_density_bins - 1, idx))
            return idx
        return 1 if gate_density >= gate_density_threshold else 0
    if mode == "plan-phase":
        if not plan_hit:
            return -1
        return 1 if stable_run_len >= plan_stable_length else 0
    if mode == "cache-hit":
        value = cache_hit_fraction if cache_hit_fraction >= 0 else 0.0
        if cache_hit_bins and cache_hit_bins > 1:
            idx = int(value * cache_hit_bins)
            idx = max(0, min(cache_hit_bins - 1, idx))
            return idx
        return 1 if value >= 0.66 else (0 if value >= 0.33 else -1)
    if mode == "alternating":
        if alternation_interval <= 0:
            return 1 if plan_hit else 0
        period = alternation_interval
        phase = (epoch // period) % 2
        return 1 if phase == 0 else 0
    raise ValueError(f"unknown regime mode: {mode}")


def _sheet_to_frame(sheet: np.ndarray, block_px: int):
    sheet = np.asarray(sheet, dtype=np.float32)
    if sheet.size == 0 or block_px <= 0:
        return np.zeros((0, 0, 4), dtype=np.float32)
    expanded = np.repeat(np.repeat(sheet, block_px, axis=0), block_px, axis=1)
    frame = np.stack([expanded] * 4, axis=-1)
    return frame


def _train_logistic(X: np.ndarray, y: np.ndarray, steps=2000, lr=0.4):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n, d = X.shape
    w = np.zeros(d + 1, dtype=np.float32)
    for _ in range(steps):
        logits = w[0] + X @ w[1:]
        logits = np.clip(logits, -20.0, 20.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y
        grad = X.T @ error / n
        bias_grad = float(error.mean())
        w[1:] -= lr * grad
        w[0] -= lr * bias_grad
    return w


def _logistic_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    logits = w[0] + X @ w[1:]
    logits = np.clip(logits, -20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= 0.5).astype(np.int32)


def _best_threshold_accuracy(values: np.ndarray, labels: np.ndarray) -> tuple[float, tuple[float, int] | None]:
    if labels.size == 0:
        return 0.0, None
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_labels = labels[order]
    candidates = []
    if sorted_vals.size:
        candidates.append(sorted_vals[0] - 1.0)
        for i in range(sorted_vals.size - 1):
            candidates.append(0.5 * (sorted_vals[i] + sorted_vals[i + 1]))
        candidates.append(sorted_vals[-1] + 1.0)
    best_acc = 0.0
    best_thr = None
    for thr in candidates:
        for sense in (1, -1):
            if sense > 0:
                preds = values > thr
            else:
                preds = values < thr
            acc = float((preds == labels).mean())
            if acc > best_acc:
                best_acc = acc
                best_thr = (thr, sense)
    return best_acc, best_thr


def _permute_labels_by_blocks(labels: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 0:
        return labels.copy()
    n = labels.size
    blocks = []
    positions = []
    pos = 0
    while pos < n:
        end = min(pos + block_size, n)
        blocks.append(labels[pos:end])
        positions.append((pos, end))
        pos = end
    blocks_by_length: dict[int, list[np.ndarray]] = {}
    positions_by_length: dict[int, list[int]] = {}
    for idx, block in enumerate(blocks):
        length = block.size
        blocks_by_length.setdefault(length, []).append(block)
        positions_by_length.setdefault(length, []).append(idx)
    permuted = np.empty_like(labels)
    for length, block_list in blocks_by_length.items():
        pos_indices = positions_by_length[length]
        perm_indices = list(range(len(block_list)))
        rng.shuffle(perm_indices)
        for target_pos, source_pos in zip(pos_indices, perm_indices):
            start, end = positions[target_pos]
            permuted[start:end] = block_list[source_pos]
    return permuted


def _best_candidate_accuracy(
    candidates: list[tuple[str, np.ndarray]],
    labels: np.ndarray,
) -> tuple[float, tuple[str, tuple[float, int] | None] | None]:
    best_acc = 0.0
    best_info = None
    for name, values in candidates:
        acc, thr = _best_threshold_accuracy(values, labels)
        if acc > best_acc:
            best_acc = acc
            best_info = (name, thr)
    return best_acc, best_info


def _regime_stats(labels: np.ndarray) -> tuple[dict[int, int], float]:
    if labels.size == 0:
        return {}, 0.0
    unique, counts = np.unique(labels, return_counts=True)
    total = float(counts.sum())
    probs = counts / total
    mask = probs > 0
    entropy = float(-np.sum(probs[mask] * np.log2(probs[mask])))
    if abs(entropy) < 1e-12:
        entropy = 0.0
    return {int(u): int(c) for u, c in zip(unique, counts)}, entropy


def _plan_hit_stage_a(observations: list[dict]) -> tuple[float, np.ndarray | None]:
    X = []
    y = []
    for obs in observations:
        tile_i, tile_j = obs["tile"]
        X.append([float(tile_i), float(tile_j), float(obs["local_mean"])])
        y.append(1.0 if obs["plan_hit"] else 0.0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if y.size == 0 or y.min() == y.max():
        return 0.5, None
    w = _train_logistic(X, y)
    preds = _logistic_predict(w, X)
    acc = float((preds == y).mean())
    return acc, w


def _regime_stage_b(
    observations: list[dict],
    sheet_h: int,
    sheet_w: int,
    block_px: int,
    block_size: int,
    perms: int,
    observer_class: str,
) -> tuple[float, tuple[str, tuple[float, int] | None] | None, float, dict[int, int], float]:
    labels = np.array([int(obs["regime_label"]) for obs in observations], dtype=np.int32)
    label_stats, label_entropy = _regime_stats(labels)
    if labels.size == 0 or labels.min() == labels.max():
        return 0.5, None, 1.0, label_stats, label_entropy
    if observer_class == "cnn":
        frames = _grayscale_frames_from_observations(observations)
        unique_labels, label_indices = np.unique(labels, return_inverse=True)
        label_indices = label_indices.astype(np.int32, copy=False)
        label_map = {int(value): int(idx) for idx, value in enumerate(unique_labels)}
        rng = np.random.default_rng(0)
        seed = rng.integers(0, 1 << 31)
        observed_acc = _train_eval_tiny_cnn(
            frames,
            label_indices,
            unique_labels.size,
            np.random.default_rng(seed),
        )
        perm_accs: list[float] = []
        for _ in range(perms):
            permuted_labels = _permute_labels_by_blocks(labels, block_size, rng)
            perm_indices = np.array(
                [label_map[int(lbl)] for lbl in permuted_labels], dtype=np.int32
            )
            perm_seed = rng.integers(0, 1 << 31)
            perm_rng = np.random.default_rng(perm_seed)
            perm_acc = _train_eval_tiny_cnn(
                frames,
                perm_indices,
                unique_labels.size,
                perm_rng,
            )
            perm_accs.append(perm_acc)
        p_value = (1 + sum(1 for acc in perm_accs if acc >= observed_acc)) / (1 + perms)
        return observed_acc, ("tiny_cnn", None), p_value, label_stats, label_entropy
    tile_means = []
    tile_fracs = []
    for obs in observations:
        means, fracs = _plan_hit_tile_stats(obs["frame"], sheet_h, sheet_w, block_px)
        tile_means.append(means)
        tile_fracs.append(fracs)
    candidates = _observer_candidate_features(tile_means, tile_fracs, observer_class)
    observed_acc, best_info = _best_candidate_accuracy(candidates, labels)
    rng = np.random.default_rng(0)
    perm_accs = []
    for _ in range(perms):
        permuted_labels = _permute_labels_by_blocks(labels, block_size, rng)
        acc, _ = _best_candidate_accuracy(candidates, permuted_labels)
        perm_accs.append(acc)
    p_value = (1 + sum(1 for acc in perm_accs if acc >= observed_acc)) / (1 + perms)
    return observed_acc, best_info, p_value, label_stats, label_entropy


def run_regime_experiment(
    observations: list[dict],
    sheet_h: int,
    sheet_w: int,
    block_px: int,
    block_size: int,
    perms: int,
    regime_mode: str,
    observer_class: str,
) -> None:
    print(f"{regime_mode.capitalize()} experiment: Stage A (instrumented) + Stage B (blind observer)")
    stage_a_acc = 0.5
    weight_str = "skipped"
    if regime_mode == "plan-hit":
        stage_a_acc, weights = _plan_hit_stage_a(observations)
        if weights is not None:
            bias = weights[0]
            coefs = ", ".join(f"{w:.3f}" for w in weights[1:])
            weight_str = f"bias={bias:.3f}, coefs=[{coefs}]"
        else:
            weight_str = "n/a"
    print(f"Stage A accuracy (tile_i, tile_j, local_mean): {stage_a_acc:.3f} ({weight_str})")
    stage_b_acc, best_info, p_value, label_stats, label_entropy = _regime_stage_b(
        observations, sheet_h, sheet_w, block_px, block_size, perms, observer_class
    )
    print(f"Regime stats: counts={label_stats}, H={label_entropy:.3f} bits")
    if best_info is None:
        print("Stage B: insufficient variation to evaluate.")
    else:
        feature_name, thr_data = best_info
        thr_descr = f"threshold={thr_data[0]:.3f} sense={'>' if thr_data[1] > 0 else '<'}" if thr_data else "n/a"
        print(f"Stage B best feature: {feature_name} acc={stage_b_acc:.3f} ({thr_descr})")
        print(f"Blocked permutation test (block_size={block_size}, perms={perms}) p-value={p_value:.3f}")


def make_data(M=256, K=256, N=256, tiles_active=0.5, tile=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(M, K), dtype=np.int8)
    W = rng.integers(0, 3, size=(K, N), dtype=np.int8)
    # gate mask -> active tiles (tile-level any-activation)
    gate_prob = gate_prob_for_tile_density(tiles_active, tile)
    gate_mask = rng.random((M, N)) < gate_prob
    tiles = tiles_from_gate_mask(gate_mask, tile=tile)
    return X, W, tiles


def dense_matmul(X, W):
    return X.astype(np.int32) @ W.astype(np.int32)


@dataclass
class TilePlan:
    tile: int
    tile_grid_shape: tuple
    i0: np.ndarray
    i1: np.ndarray
    j0: np.ndarray
    j1: np.ndarray
    tile_ids: np.ndarray

    @property
    def count(self):
        return int(self.i0.size)


def build_tile_plan(tiles, tile, M, N):
    coords = np.argwhere(tiles)
    if coords.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return TilePlan(
            tile=tile,
            tile_grid_shape=tiles.shape,
            i0=empty,
            i1=empty,
            j0=empty,
            j1=empty,
            tile_ids=empty,
        )
    i0 = (coords[:, 0] * tile).astype(np.int32)
    j0 = (coords[:, 1] * tile).astype(np.int32)
    i1 = np.minimum(i0 + tile, M).astype(np.int32)
    j1 = np.minimum(j0 + tile, N).astype(np.int32)
    tile_ids = (coords[:, 0] * tiles.shape[1] + coords[:, 1]).astype(np.int32)
    return TilePlan(
        tile=tile,
        tile_grid_shape=tiles.shape,
        i0=i0,
        i1=i1,
        j0=j0,
        j1=j1,
        tile_ids=tile_ids,
    )


def _permute_plan(plan: TilePlan, rng: np.random.Generator) -> TilePlan:
    count = plan.count
    if count <= 1:
        return plan
    perm = rng.permutation(count)
    return TilePlan(
        tile=plan.tile,
        tile_grid_shape=plan.tile_grid_shape,
        i0=plan.i0[perm],
        i1=plan.i1[perm],
        j0=plan.j0[perm],
        j1=plan.j1[perm],
        tile_ids=plan.tile_ids[perm],
    )


def _apply_regime_noise(
    mode: str,
    tiles: np.ndarray,
    gate_mask: np.ndarray,
    tile: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    info: dict[str, object] = {
        "mode": mode,
        "desc": None,
        "force_replan": False,
        "permute_plan": False,
    }
    mode = (mode or "").lower()
    noisy_tiles = tiles.copy()
    noisy_gate_mask = gate_mask
    if mode in ("", "none"):
        info["desc"] = "no noise"
        return noisy_tiles, noisy_gate_mask, info
    if mode == "tile_shuffle":
        flat = noisy_tiles.ravel()
        rng.shuffle(flat)
        noisy_tiles = flat.reshape(noisy_tiles.shape)
        info["desc"] = "tile order shuffled"
    elif mode == "cache_poison":
        flat = noisy_tiles.ravel()
        true_idx = np.flatnonzero(flat)
        drop_frac = 0.25
        drop_count = int(len(true_idx) * drop_frac)
        if drop_count > 0:
            drop_idx = rng.choice(true_idx, size=drop_count, replace=False)
            flat[drop_idx] = False
        noisy_tiles = flat.reshape(noisy_tiles.shape)
        info["desc"] = f"cache poison drop={drop_count}"
    elif mode == "gate_jitter":
        jitter = rng.uniform(-0.3, 0.3, size=gate_mask.shape)
        gate_float = gate_mask.astype(np.float32) + jitter
        noisy_gate_mask = (gate_float.clip(0.0, 1.0) > 0.5).astype(bool)
        noisy_tiles = tiles_from_gate_mask(noisy_gate_mask, tile=tile)
        info["desc"] = "gate density jittered"
    elif mode == "replan_always":
        info["force_replan"] = True
        info["desc"] = "force plan rebuild"
    elif mode == "schedule_jitter":
        info["permute_plan"] = True
        info["desc"] = "plan execution order permuted"
    else:
        info["desc"] = "unhandled noise"
    return noisy_tiles, noisy_gate_mask, info


def _load_training_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        W = np.asarray(data["W"], dtype=np.int8)
        gate_mask = np.asarray(data["gate_mask"], dtype=bool)
    return W, gate_mask


def _save_training_state(path: Path, W: np.ndarray, gate_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, W=W, gate_mask=gate_mask.astype(np.uint8))


def jaccard_similarity(a_ids, b_ids):
    if a_ids.size == 0 and b_ids.size == 0:
        return 1.0
    if a_ids.size == 0 or b_ids.size == 0:
        return 0.0
    a = set(a_ids.tolist())
    b = set(b_ids.tolist())
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 1.0


def update_gate_mask(gate_mask, flip_prob, rng):
    if flip_prob <= 0.0:
        return gate_mask
    flips = rng.random(gate_mask.shape) < flip_prob
    return np.where(flips, ~gate_mask, gate_mask)


def _load_vnni_kernel():
    lib_path = os.path.join(os.path.dirname(__file__), "vnni_kernel.so")
    if not os.path.exists(lib_path):
        return None
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        return None
    fn = lib.vnni_tile_i8
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int8),  # A
        ctypes.POINTER(ctypes.c_int8),  # B
        ctypes.POINTER(ctypes.c_int32),  # C
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        ctypes.c_int,  # lda
        ctypes.c_int,  # ldb
        ctypes.c_int,  # ldc
    ]
    return fn


_VNNI_KERNEL = _load_vnni_kernel()


def vnni_microkernel(Ablk, Bblk):
    if _VNNI_KERNEL is None:
        return Ablk.astype(np.int32) @ Bblk.astype(np.int32)
    A = np.ascontiguousarray(Ablk, dtype=np.int8)
    B = np.ascontiguousarray(Bblk, dtype=np.int8)
    C_view = np.zeros((Ablk.shape[0], Bblk.shape[1]), dtype=np.int32)
    _VNNI_KERNEL(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        C_view.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        A.shape[0],
        B.shape[1],
        A.shape[1],
        A.strides[0] // A.itemsize,
        B.strides[0] // B.itemsize,
        C_view.strides[0] // C_view.itemsize,
    )
    return C_view


def block_sparse_matmul(X, W, tiles, tile=32, microkernel=vnni_microkernel):
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.int32)
    for ti in range(tiles.shape[0]):
        for tj in range(tiles.shape[1]):
            if not tiles[ti, tj]:
                continue
            i0, j0 = ti * tile, tj * tile
            i1, j1 = min(i0 + tile, M), min(j0 + tile, N)
            for k0 in range(0, K, tile):
                k1 = min(k0 + tile, K)
                Ablk = X[i0:i1, k0:k1]
                Bblk = W[k0:k1, j0:j1]
                C[i0:i1, j0:j1] += microkernel(Ablk, Bblk)
    return C


def block_sparse_matmul_plan(X, W, plan, microkernel=vnni_microkernel):
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.int32)
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        for k0 in range(0, K, plan.tile):
            k1 = min(k0 + plan.tile, K)
            Ablk = X[i0:i1, k0:k1]
            Bblk = W[k0:k1, j0:j1]
            C[i0:i1, j0:j1] += microkernel(Ablk, Bblk)
    return C


def activation_plan(C, plan, clamp_min=0):
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        block = C[i0:i1, j0:j1]
        C[i0:i1, j0:j1] = np.maximum(block, clamp_min)
    return C


def energy_plan(C, plan):
    energies = np.zeros(plan.count, dtype=np.float32)
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        block = C[i0:i1, j0:j1]
        energies[idx] = float(np.sum(block * block))
    return energies


def _normalized_tile_energy(C: np.ndarray, plan: TilePlan) -> np.ndarray:
    energy = tile_energy_map(C, plan).astype(np.float32)
    return np.log1p(energy)


def radial_bins_energy(C: np.ndarray, bins: int = 4) -> np.ndarray:
    if bins <= 0 or C.size == 0:
        return np.zeros(bins, dtype=np.float32)
    height, width = C.shape
    if height <= 0 or width <= 0:
        return np.zeros(bins, dtype=np.float32)
    y, x = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = float(r.max()) if r.size else 0.0
    if rmax > 0:
        r = r / rmax
    else:
        r = np.zeros_like(r)
    E = np.zeros(bins, dtype=np.float32)
    for bin_idx in range(bins):
        lower = bin_idx / bins
        upper = (bin_idx + 1) / bins if bin_idx < bins - 1 else 1.0 + 1e-6
        mask = (r >= lower) & (r < upper)
        if not mask.any():
            continue
        block = C[mask].astype(np.float32)
        E[bin_idx] = float(np.sum(block * block))
    return np.log1p(E)


def _phase3_artifact_paths(timestamp: str) -> tuple[Path, Path]:
    log_dir = Path("logs/bsmoe_train")
    log_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"bsmoe_phase3_{timestamp}.json"
    plot_file = outputs_dir / f"bsmoe_phase3_{timestamp}.png"
    return log_file, plot_file


def _phase3_metric_stats(history: list[dict]) -> dict:
    if not history:
        return {"metrics": {}}
    epochs = np.array([entry["epoch"] for entry in history], dtype=np.float64)
    metric_stats = {}
    for metric in ("task_loss", "quotient_loss", "mdl_cost"):
        values = np.array([entry[metric] for entry in history], dtype=np.float64)
        if values.size == 0:
            continue
        scale = float(np.max(np.abs(values))) if np.max(np.abs(values)) > 0 else 1.0
        deg = 0
        if values.size >= 3:
            deg = 2
        elif values.size == 2:
            deg = 1
        coeffs = np.polyfit(epochs, values, deg=deg)
        poly = np.poly1d(coeffs)
        fit_type = {0: "constant", 1: "linear", 2: "quadratic"}[deg]
        fit_values = poly(epochs).tolist()
        metric_stats[metric] = {
            "scale": scale,
            "best_fit": {
                "type": fit_type,
                "coefficients": coeffs.tolist(),
            },
            "fit_values": fit_values,
        }
    return {"metrics": metric_stats}


def _dump_phase3_plot(history: list[dict], plot_path: Path, stats: dict) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib unavailable; skipping Phase-3 plot", file=sys.stderr)
        return
    epochs = [entry["epoch"] for entry in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_stats = stats.get("metrics", {})
    for metric in ("task_loss", "quotient_loss", "mdl_cost"):
        values = [entry[metric] for entry in history]
        if not values:
            continue
        ax.plot(epochs, values, label=metric)
        fit_info = metric_stats.get(metric)
        if fit_info:
            ax.plot(
                epochs,
                fit_info["fit_values"],
                label=f"{metric}_fit",
                linestyle="--",
                linewidth=1,
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Phase-3 run metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def _dump_phase3_artifacts(history: list[dict], timestamp: str) -> None:
    stats = _phase3_metric_stats(history)
    for metric, info in stats["metrics"].items():
        scale = max(info.get("scale", 1.0), 1e-9)
        for entry in history:
            entry[f"{metric}_scaled"] = float(entry[metric]) / scale
    log_file, plot_file = _phase3_artifact_paths(timestamp)
    payload = {"timestamp": timestamp, "history": history, "stats": stats}
    with log_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _dump_phase3_plot(history, plot_file, stats)
    print(f"Phase-3 artifacts saved: {log_file}, {plot_file}")


def _plan_mdl_cost(plan: TilePlan, *, total_tiles: int, plan_changed: bool) -> float:
    if total_tiles <= 0:
        return 0.0
    active_frac = float(plan.count) / float(total_tiles)
    change_pen = 1.0 if plan_changed else 0.0
    return active_frac + 0.25 * change_pen


def train_epoch_phase3(
    X: np.ndarray,
    W: np.ndarray,
    plan_r1: TilePlan,
    plan_r0: TilePlan,
    *,
    lr: float,
    alpha: float,
    beta: float,
    total_tiles: int,
    plan_changed: bool,
    radial_bins: int = 0,
    microkernel=vnni_microkernel,
) -> tuple[np.ndarray, float, dict, np.ndarray]:
    C1 = block_sparse_matmul_plan(X, W, plan_r1, microkernel=microkernel)
    C0 = block_sparse_matmul_plan(X, W, plan_r0, microkernel=microkernel)
    V1 = _normalized_tile_energy(C1, plan_r1)
    V0 = _normalized_tile_energy(C0, plan_r0)
    task_loss = float((C1.astype(np.float32) ** 2).mean())
    q_loss = float(((V1 - V0) ** 2).mean())
    mdl = _plan_mdl_cost(plan_r1, total_tiles=total_tiles, plan_changed=plan_changed)
    total_loss = task_loss + alpha * q_loss + beta * mdl
    delta_V = V1 - V0
    delta_V_flat = delta_V.reshape(-1)
    tile_ids = plan_r1.tile_ids
    err_q = np.zeros_like(C1, dtype=np.float32)
    C1_float = C1.astype(np.float32)
    for idx in range(plan_r1.count):
        i0 = int(plan_r1.i0[idx])
        j0 = int(plan_r1.j0[idx])
        i1 = int(plan_r1.i1[idx])
        j1 = int(plan_r1.j1[idx])
        tile_id = int(tile_ids[idx]) if idx < tile_ids.size else idx
        err_q[i0:i1, j0:j1] += 4.0 * float(delta_V_flat[tile_id]) * C1_float[i0:i1, j0:j1]
    err = C1_float + alpha * err_q
    gradW = np.zeros_like(W, dtype=np.float32)
    for idx in range(plan_r1.count):
        i0 = int(plan_r1.i0[idx])
        j0 = int(plan_r1.j0[idx])
        i1 = int(plan_r1.i1[idx])
        j1 = int(plan_r1.j1[idx])
        X_blk = X[i0:i1, :].astype(np.float32)
        err_blk = err[i0:i1, j0:j1]
        gradW[:, j0:j1] += X_blk.T @ err_blk
    W = W - lr * gradW.astype(W.dtype)
    radial_metrics: dict[str, float | list[float]] = {}
    if radial_bins > 0:
        radial1 = radial_bins_energy(C1, bins=radial_bins)
        radial0 = radial_bins_energy(C0, bins=radial_bins)
        radial_metrics = {
            "radial_bins": radial1.tolist(),
            "radial_quotient_loss": float(((radial1 - radial0) ** 2).mean()),
        }
    metrics = {
        "task_loss": task_loss,
        "quotient_loss": q_loss,
        "mdl_cost": float(mdl),
        "alpha": float(alpha),
        "beta": float(beta),
    }
    metrics.update(radial_metrics)
    return W, float(total_loss), metrics, C1


def fused_sequence(X, W, plan, microkernel=vnni_microkernel):
    C = block_sparse_matmul_plan(X, W, plan, microkernel=microkernel)
    C = activation_plan(C, plan, clamp_min=0)
    energies = energy_plan(C, plan)
    return C, energies


def train_epoch(X, W, plan, lr=1e-3, microkernel=vnni_microkernel):
    # forward block-sparse
    C = block_sparse_matmul_plan(X, W, plan, microkernel=microkernel)
    # fake target: zeros
    err = C
    # backward: simple gradient on W for active tiles
    gradW = np.zeros_like(W, dtype=np.int32)
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        # X_blk: (tile, K) -> transpose -> (K, tile)
        X_blk = X[i0:i1, :].astype(np.int32)  # (tile, K)
        err_blk = err[i0:i1, j0:j1]           # (tile, tile)
        gradW[:, j0:j1] += X_blk.T @ err_blk
    W = W - lr * gradW.astype(W.dtype)
    loss = float((err ** 2).mean())
    return W, loss, C


def bench(fn, *args, reps=3, **kwargs):
    fn(*args, **kwargs)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        dur = (t1 - t0) * 1e3
        best = dur if best is None or dur < best else best
    return best


def main():
    parser = argparse.ArgumentParser(description="dashilearn block-sparse demo")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--stay-open",
        type=float,
        default=0.0,
        help="Seconds to keep running after training finishes (writes the last sheet each second)",
    )
    parser.add_argument("--stay-interval", type=float, default=1.0, help="Seconds between sheet refresh while staying open")
    parser.add_argument(
        "--capture-vulkan",
        action="store_true",
        help="Capture the Vulkan sheet frame once per epoch via the Task B wiring helper",
    )
    parser.add_argument("--vulkan-block-px", type=int, default=16, help="Pixels per sheet tile in the captured frame")
    parser.add_argument("--vulkan-alpha", type=float, default=0.97, help="Sheet fade alpha for Vulkan capture")
    parser.add_argument("--vulkan-vmin", type=float, default=0.0, help="Minimum sheet value clamp")
    parser.add_argument("--vulkan-vmax", type=float, default=1.0, help="Maximum sheet value clamp")
    parser.add_argument(
        "--vulkan-clamp",
        action="store_true",
        help="Clamp sheet values before the Vulkan shader runs",
    )
    parser.add_argument(
        "--vk-icd",
        type=Path,
        help="Optional Vulkan ICD JSON used when capturing the frame",
    )
    parser.add_argument(
        "--plan-hit-experiment",
        action="store_true",
        help="Run the plan-hit observability experiment (Stage A + Stage B) after training.",
    )
    parser.add_argument(
        "--regime-mode",
        choices=("plan-hit", "gate-density", "alternating", "plan-phase", "cache-hit"),
        default="plan-hit",
        help="Which regime definition to test in Stage B.",
    )
    parser.add_argument(
        "--gate-density-threshold",
        type=float,
        default=DEFAULT_GATE_DENSITY_THRESHOLD,
        help="Threshold (0-1) for gate-density regimes (>= threshold -> high).",
    )
    parser.add_argument(
        "--regime-alternation-interval",
        type=int,
        default=DEFAULT_ALTERNATION_INTERVAL,
        help="Epoch block size for the synthetic alternating regime.",
    )
    parser.add_argument(
        "--gate-density-bins",
        type=int,
        default=DEFAULT_GATEDENSITY_BINS,
        help="Number of bins for gate-density regime (overrides threshold when >1).",
    )
    parser.add_argument(
        "--cache-hit-bins",
        type=int,
        default=DEFAULT_CACHE_HIT_BINS,
        help="Number of bins for cache-hit / plan-overlap regime (overrides thresholds when >1).",
    )
    parser.add_argument(
        "--plan-stable-length",
        type=int,
        default=DEFAULT_PLAN_STABLE_LENGTH,
        help="Minimum consecutive plan hits before the plan-phase label toggles to 1.",
    )
    parser.add_argument(
        "--observer-class",
        choices=("scalar", "corr", "cnn"),
        default="scalar",
        help="Observer feature set used by Stage B (scalar tiles vs correlation features, or tiny CNN).",
    )
    parser.add_argument(
        "--plan-hit-block-size",
        type=int,
        default=PLAN_HIT_EXPERIMENT_DEFAULT_BLOCK,
        help="Frames-per-block used in the blocked permutation test.",
    )
    parser.add_argument(
        "--plan-hit-perms",
        type=int,
        default=PLAN_HIT_EXPERIMENT_DEFAULT_PERMS,
        help="Number of permutations for the Stage B permutation test.",
    )
    parser.add_argument(
        "--blocked-permutations",
        type=_parse_bool_flag,
        default=False,
        help="Legacy reminder that Stage B uses blocked permutations (no-op).",
    )
    parser.add_argument(
        "--regime-noise",
        choices=("none", "replan_always", "tile_shuffle", "cache_poison", "gate_jitter", "schedule_jitter"),
        default="none",
        help="Runtime perturbation mode for regime-robust execution testing.",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="Path to a saved training state (W + gate mask) to initialize the run.",
    )
    parser.add_argument(
        "--save-state",
        type=Path,
        help="Path to write the final training state (W + gate mask) after the run.",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Enable Phase-3 quotient-by-construction loss (plan-equivalence + MDL cost).",
    )
    parser.add_argument(
        "--phase3-alpha",
        type=float,
        default=1.0,
        help="Weight on the quotient consistency loss.",
    )
    parser.add_argument(
        "--phase3-beta",
        type=float,
        default=0.05,
        help="Weight on the MDL-style gauge cost.",
    )
    parser.add_argument(
        "--phase3-alpha-warmup",
        type=float,
        default=0.3,
        help="Fraction of epochs to ramp alpha from 0 to phase3-alpha.",
    )
    parser.add_argument(
        "--phase3-radial-bins",
        type=int,
        default=DEFAULT_PHASE3_RADIAL_BINS,
        help="Number of radial bins for the supplementary quotient diagnostic (<=0 to disable).",
    )
    args = parser.parse_args()

    M = N = K = 256
    tile_active = 0.5
    tile = 32
    epochs = args.epochs
    gate_flip = 0.01
    jaccard_thresh = 0.9
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(M, K), dtype=np.int8)
    if args.resume_state:
        if not args.resume_state.exists():
            print(f"resume state not found: {args.resume_state}", file=sys.stderr)
            sys.exit(1)
        W, gate_mask = _load_training_state(args.resume_state)
        if W.shape != (K, N) or gate_mask.shape != (M, N):
            print("resume state has incompatible shapes", file=sys.stderr)
            sys.exit(1)
    else:
        W = rng.integers(0, 3, size=(K, N), dtype=np.int8)
        gate_prob = gate_prob_for_tile_density(tile_active, tile)
        gate_mask = rng.random((M, N)) < gate_prob
    tiles = tiles_from_gate_mask(gate_mask, tile=tile)

    # Baseline dense timing
    t_dense = bench(dense_matmul, X, W)

    t_plan0 = time.perf_counter()
    plan = build_tile_plan(tiles, tile=tile, M=M, N=N)
    sheet_h, sheet_w = plan.tile_grid_shape
    plan0 = build_tile_plan(tiles, tile=tile, M=M, N=N)
    total_tiles = int(sheet_h * sheet_w)
    phase3_history = [] if args.phase3 else None
    phase3_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") if args.phase3 else None
    t_plan = (time.perf_counter() - t_plan0) * 1e3
    t_pack = 0.0

    # Block-sparse timing (plan reused)
    t_bs = bench(block_sparse_matmul_plan, X, W, plan, microkernel=vnni_microkernel)
    C_ref = block_sparse_matmul_plan(X, W, plan, microkernel=vnni_microkernel)
    t_act = bench(lambda: activation_plan(C_ref.copy(), plan, clamp_min=0))
    t_energy = bench(lambda: energy_plan(C_ref, plan))
    t_fused = bench(lambda: fused_sequence(X, W, plan, microkernel=vnni_microkernel))
    sheet_energy = tile_energy_map(C_ref, plan)
    dump_sheet_energy(sheet_energy)
    block_px = max(1, args.vulkan_block_px)
    vulkan_capture = None
    if args.capture_vulkan:
        try:
            if VulkanFrameCapture is None:
                raise RuntimeError("vulkan_compute.frame_capture is unavailable")
            if args.vk_icd:
                os.environ["VK_ICD_FILENAMES"] = str(args.vk_icd)
            width = sheet_w * block_px
            height = sheet_h * block_px
            vulkan_capture = VulkanFrameCapture(
                width=width,
                height=height,
                sheet_w=sheet_w,
                sheet_h=sheet_h,
                block_px=block_px,
                alpha=args.vulkan_alpha,
                vmin=args.vulkan_vmin,
                vmax=args.vulkan_vmax,
                use_clamp=args.vulkan_clamp,
            )
        except Exception as exc:
            print(f"warning: Vulkan frame capture disabled: {exc}", file=sys.stderr)
            vulkan_capture = None
    active_frac = float(tiles.mean()) if tiles.size else 0.0
    last_capture_sheet = None
    if vulkan_capture:
        try:
            capture_sheet = _sheet_values_for_capture(
                sheet_energy,
                epoch=0,
                gate_density=active_frac,
                sheet_h=sheet_h,
                sheet_w=sheet_w,
            )
            frame = vulkan_capture.capture(capture_sheet)
            thr = VULKAN_CAPTURE_THRESHOLD
            frac_above_thr = (frame > thr).mean()
            max_val = float(frame.max())
            tile_i, tile_j = _semantic_tile_coord(0, sheet_h, sheet_w)
            tile_mean = _tile_block_mean(frame, tile_i, tile_j, block_px)
            print(
                "Initial Vulkan frame captured:",
                frame.shape,
                f"tile=({tile_i},{tile_j}) local_mean={tile_mean:.2f}",
                f"mean={frame.mean():.2f} std={frame.std():.2f}",
                f"frac>{thr}={frac_above_thr:.3f}",
                f"max={max_val:.2f}",
            )
            last_capture_sheet = capture_sheet
        except Exception as exc:
            print(f"warning: failed to capture initial Vulkan frame: {exc}", file=sys.stderr)

    print("Block-sparse MoE-style matmul")
    print(f"M=N=K={M}, active tiles ~{active_frac*100:.1f}% (target {tile_active*100:.1f}%)")
    print(f"dense matmul      : {t_dense:6.2f} ms/call")
    print(f"block-sparse matmul: {t_bs:6.2f} ms/call   speedup x{t_dense/t_bs:5.2f}")
    if _VNNI_KERNEL is None:
        print("microkernel        : vnni_microkernel (numpy int32 fallback)")
    else:
        print("microkernel        : vnni_kernel.so (ctypes)")

    print(f"plan time         : {t_plan:6.2f} ms")
    print(f"pack time         : {t_pack:6.2f} ms")
    print(f"exec matmul       : {t_bs:6.2f} ms")
    print(f"exec activation   : {t_act:6.2f} ms")
    print(f"exec energy       : {t_energy:6.2f} ms")
    print(f"exec fused total  : {t_fused:6.2f} ms")

    # Tiny training loop (for illustration)
    sheet_energy = np.zeros(plan.tile_grid_shape, dtype=np.float32)
    plan_hits = 0
    plan_hit_observations: list[dict] = []
    stable_run_len = 0
    for e in range(epochs):
        t_gate0 = time.perf_counter()
        gate_mask = update_gate_mask(gate_mask, gate_flip, rng)
        tiles = tiles_from_gate_mask(gate_mask, tile=tile)
        tiles, gate_mask, noise_info = _apply_regime_noise(
            args.regime_noise,
            tiles,
            gate_mask,
            tile,
            rng,
        )
        t_gate = (time.perf_counter() - t_gate0) * 1e3
        next_plan = build_tile_plan(tiles, tile=tile, M=M, N=N)
        if noise_info.get("permute_plan"):
            next_plan = _permute_plan(next_plan, rng)
        jacc = jaccard_similarity(plan.tile_ids, next_plan.tile_ids)
        reuse = jacc >= jaccard_thresh
        if noise_info.get("force_replan"):
            reuse = False
        plan_changed = False
        if not reuse:
            plan = next_plan
            plan_changed = True
        plan_hit = bool(reuse)
        plan_hits += int(reuse)
        t0 = time.perf_counter()
        if args.phase3:
            warm_epochs = max(1, int(math.ceil(args.phase3_alpha_warmup * max(1, epochs))))
            alpha_scale = min(1.0, float(e) / float(warm_epochs))
            alpha = args.phase3_alpha * alpha_scale
            W, loss, phase3_metrics, C = train_epoch_phase3(
                X,
                W,
                plan_r1=plan,
                plan_r0=plan0,
                lr=1e-5,
                alpha=alpha,
                beta=args.phase3_beta,
                total_tiles=total_tiles,
                plan_changed=plan_changed,
                radial_bins=args.phase3_radial_bins,
                microkernel=vnni_microkernel,
            )
        else:
            W, loss, C = train_epoch(X, W, plan, lr=1e-5, microkernel=vnni_microkernel)
        t1 = time.perf_counter()
        print(
            f"epoch {e+1}: loss={loss:8.2e}  time={(t1-t0)*1e3:6.2f} ms  "
            f"jaccard={jacc:5.2f}  plan_hit={int(reuse)}  gate_time={t_gate:5.2f} ms"
        )
        if args.phase3:
            print(
                f"          phase3: task={phase3_metrics['task_loss']:.2e}  "
                f"q={phase3_metrics['quotient_loss']:.2e}  "
                f"mdl={phase3_metrics['mdl_cost']:.3f}  "
                f"alpha={phase3_metrics['alpha']:.3f}"
            )
            if phase3_history is not None:
                phase3_history.append(
                        {
                            "epoch": e + 1,
                            "task_loss": phase3_metrics["task_loss"],
                            "quotient_loss": phase3_metrics["quotient_loss"],
                            "mdl_cost": phase3_metrics["mdl_cost"],
                            "alpha": phase3_metrics["alpha"],
                            "plan_hit": plan_hit,
                            "plan_changed": plan_changed,
                            "radial_bins": phase3_metrics.get("radial_bins"),
                            "radial_quotient_loss": phase3_metrics.get("radial_quotient_loss"),
                            "regime_noise": args.regime_noise,
                            "regime_noise_desc": noise_info.get("desc"),
                            "jaccard": jacc,
                            "gate_time": t_gate,
                            "resume_state": str(args.resume_state) if args.resume_state else None,
                        }
                    )
        sheet_energy = tile_energy_map(C, plan)
        dump_sheet_energy(sheet_energy)
        gate_density = float(tiles.mean()) if tiles.size else 0.0
        if plan_hit and reuse:
            stable_run_len += 1
        else:
            stable_run_len = 0
        cache_hit_fraction = float(jacc)
        capture_sheet = _sheet_values_for_capture(
            sheet_energy,
            epoch=e,
            gate_density=gate_density,
            sheet_h=sheet_h,
            sheet_w=sheet_w,
        )
        frame_for_experiment = _sheet_to_frame(capture_sheet, block_px)
        tile_i, tile_j = _semantic_tile_coord(e, sheet_h, sheet_w)
        tile_mean = _tile_block_mean(frame_for_experiment, tile_i, tile_j, block_px)
        regime_label = _regime_label_for_mode(
            args.regime_mode,
            gate_density=gate_density,
            epoch=e,
            plan_hit=plan_hit,
            gate_density_threshold=args.gate_density_threshold,
            alternation_interval=args.regime_alternation_interval,
            gate_density_bins=args.gate_density_bins,
            plan_stable_length=args.plan_stable_length,
            stable_run_len=stable_run_len,
            cache_hit_fraction=cache_hit_fraction,
            cache_hit_bins=args.cache_hit_bins,
        )
        if vulkan_capture:
            try:
                frame = vulkan_capture.capture(capture_sheet)
                thr = VULKAN_CAPTURE_THRESHOLD
                frac_above_thr = (frame > thr).mean()
                max_val = float(frame.max())
                tile_mean = _tile_block_mean(frame, tile_i, tile_j, block_px)
                frame_for_experiment = frame.copy()
                print(
                    f"Vulkan frame [epoch {e+1}] tile=({tile_i},{tile_j}) local_mean={tile_mean:.2f}",
                    f"mean={frame.mean():.2f}",
                    f"std={frame.std():.2f}",
                    f"frac>{thr}={frac_above_thr:.3f}",
                    f"max={max_val:.2f}",
                )
                last_capture_sheet = capture_sheet
            except Exception as exc:
                print(f"warning: Vulkan capture failed at epoch {e+1}: {exc}", file=sys.stderr)
        if args.plan_hit_experiment:
            plan_hit_observations.append(
                {
                    "epoch": e,
                    "plan_hit": plan_hit,
                    "frame": frame_for_experiment,
                    "tile": (tile_i, tile_j),
                    "local_mean": tile_mean,
                    "cache_hit_fraction": cache_hit_fraction,
                    "stable_run_len": stable_run_len,
                    "regime_label": regime_label,
                }
            )
    if args.plan_hit_experiment and plan_hit_observations:
        run_regime_experiment(
            plan_hit_observations,
            sheet_h=sheet_h,
            sheet_w=sheet_w,
            block_px=block_px,
            block_size=args.plan_hit_block_size,
            perms=args.plan_hit_perms,
            regime_mode=args.regime_mode,
            observer_class=args.observer_class,
        )
    if args.phase3 and phase3_history:
        _dump_phase3_artifacts(phase3_history, phase3_timestamp)
    print(f"plan_hit_rate     : {plan_hits}/{epochs}")
    if args.save_state:
        _save_training_state(args.save_state, W, gate_mask)
        print(f"Training state saved to {args.save_state}")
    if args.stay_open > 0:
        stay_interval = max(0.01, args.stay_interval)
        end_time = time.time() + args.stay_open
        print(f"staying open for {args.stay_open:.1f}s (refresh every {stay_interval:.2f}s)")
        while time.time() < end_time:
            dump_sheet_energy(sheet_energy)
            if vulkan_capture and last_capture_sheet is not None:
                try:
                    vulkan_capture.capture(last_capture_sheet)
                except Exception:
                    pass
            time.sleep(stay_interval)
    if vulkan_capture:
        vulkan_capture.close()


if __name__ == "__main__":
    main()

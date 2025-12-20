"""
levin_ca_train.py
-----------------
Minimal 2D ternary cellular automaton inspired by Levin-style local rules:
  - States in {0,1,2}
  - Next state is a local functional of 3x3 neighborhood counts
  - We generate data from the ground-truth rule and train a tiny
    count-based classifier to recover it.

This is a small, self-contained training/demo script (no PyTorch).
"""

import time
import numpy as np


def levin_rule(neigh: np.ndarray) -> np.ndarray:
    """
    neigh: (..., 3, 3) neighborhood of ints in {0,1,2}
    Returns next state per center.
    Rule (heuristic, Levin-inspired):
      - Count each state in the 3x3 neighborhood (including center)
      - If state 2 dominates, go to 2
      - Else if state 1 dominates, go to 1
      - Else stay as center
    """
    counts = np.zeros(neigh.shape[:-2] + (3,), dtype=np.int16)
    for v in (0, 1, 2):
        counts[..., v] = (neigh == v).sum(axis=(-2, -1))
    center = neigh[..., 1, 1]
    dom = counts.argmax(axis=-1)
    # ties: stay as center
    maxc = counts.max(axis=-1)
    tie = (counts == maxc[..., None]).sum(axis=-1) > 1
    out = np.where(tie, center, dom)
    return out.astype(np.int8)


def step_grid(grid: np.ndarray) -> np.ndarray:
    """Apply rule to a 2D grid with wrap-around boundaries."""
    H, W = grid.shape
    # build neighborhoods with padding via roll
    neigh = np.stack(
        [
            np.roll(np.roll(grid, di, axis=0), dj, axis=1)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
        ],
        axis=0,
    ).reshape(3, 3, H, W).transpose(2, 3, 0, 1)
    return levin_rule(neigh)


def make_dataset(num_samples=2000, H=32, W=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(num_samples, H, W), dtype=np.int8)
    Y = np.empty_like(X)
    for i in range(num_samples):
        Y[i] = step_grid(X[i])
    return X, Y


def features_from_grid(grid: np.ndarray) -> np.ndarray:
    """
    Compute simple features per cell:
      - count0, count1, count2 in 3x3 neighborhood
      - center value (one-hot 3)
    Returns shape (H*W, 6).
    """
    H, W = grid.shape
    neigh = np.stack(
        [
            np.roll(np.roll(grid, di, axis=0), dj, axis=1)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
        ],
        axis=0,
    ).reshape(3, 3, H, W).transpose(2, 3, 0, 1)
    feats = np.zeros((H, W, 6), dtype=np.float32)
    for v in (0, 1, 2):
        feats[..., v] = (neigh == v).sum(axis=(-2, -1))
    center = neigh[..., 1, 1]
    for v in (0, 1, 2):
        feats[..., 3 + v] = (center == v).astype(np.float32)
    return feats.reshape(-1, 6)


def one_hot(y: np.ndarray, num_classes=3) -> np.ndarray:
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y.ravel()] = 1.0
    return out


def train_logreg(X_feats, Y, lr=1e-2, iters=200):
    """
    Simple multiclass logistic regression with SGD over all samples.
    X_feats: (N, 6)
    Y: (N,) labels in {0,1,2}
    """
    N, D = X_feats.shape
    C = 3
    W = np.zeros((D, C), dtype=np.float32)
    Y_one = one_hot(Y, C)
    for _ in range(iters):
        logits = X_feats @ W
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad = X_feats.T @ (probs - Y_one) / N
        W -= lr * grad
    return W


def train_logreg_weighted(feat_counts, lr=1e-2, iters=200):
    """
    feat_counts: list of (feature_vector, label_counts), where label_counts is length-3 counts.
    """
    feats = np.stack([f for f, _ in feat_counts], axis=0)
    weights = np.stack([c for _, c in feat_counts], axis=0)  # (F,3)
    F, D = feats.shape
    C = 3
    W = np.zeros((D, C), dtype=np.float32)
    for _ in range(iters):
        logits = feats @ W  # (F,C)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad = np.zeros_like(W)
        for f in range(F):
            # weighted cross-entropy gradient
            grad += np.outer(feats[f], (probs[f] - weights[f] / max(1, weights[f].sum())))
        grad /= F
        W -= lr * grad
    return W


def eval_logreg(W, X_feats, Y):
    logits = X_feats @ W
    pred = logits.argmax(axis=1)
    acc = (pred == Y).mean()
    return acc


def main():
    # data
    X, Y = make_dataset(num_samples=200, H=32, W=32, seed=0)
    X_test, Y_test = make_dataset(num_samples=50, H=32, W=32, seed=1)

    # featurize
    X_feats = np.concatenate([features_from_grid(x) for x in X], axis=0)
    Y_flat = np.concatenate([y.ravel() for y in Y], axis=0)
    X_feats_test = np.concatenate([features_from_grid(x) for x in X_test], axis=0)
    Y_flat_test = np.concatenate([y.ravel() for y in Y_test], axis=0)

    # build histogram of unique features for weighted training
    feat_tuples = list(map(tuple, X_feats.astype(np.int16)))
    unique_feats, idx = np.unique(feat_tuples, return_inverse=True, axis=0)
    counts = np.zeros((unique_feats.shape[0], 3), dtype=np.int32)
    for i, lab in enumerate(Y_flat):
        counts[idx[i], lab] += 1
    feat_counts = [(np.array(f, dtype=np.float32), counts[k]) for k, f in enumerate(unique_feats)]

    # train (flat and weighted)
    t0 = time.perf_counter()
    W = train_logreg(X_feats, Y_flat, lr=5e-3, iters=300)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    Ww = train_logreg_weighted(feat_counts, lr=5e-3, iters=300)
    t3 = time.perf_counter()

    # eval
    train_acc = eval_logreg(W, X_feats, Y_flat)
    test_acc = eval_logreg(W, X_feats_test, Y_flat_test)
    train_acc_w = eval_logreg(Ww, X_feats, Y_flat)
    test_acc_w = eval_logreg(Ww, X_feats_test, Y_flat_test)

    print("Ternary 2D CA rule learning (count-based logreg)")
    print(f"Flat    Train acc: {train_acc*100:.2f}%  Test acc: {test_acc*100:.2f}%  time={(t1 - t0)*1e3:.2f} ms")
    print(f"Weighted Train acc: {train_acc_w*100:.2f}%  Test acc: {test_acc_w*100:.2f}%  time={(t3 - t2)*1e3:.2f} ms")


if __name__ == "__main__":
    main()

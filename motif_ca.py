"""
motif_ca.py
-----------
Cellular automaton "motif lab" that makes M4/M7/M9 behaviors explicit:
  - Two-layer state: phenotype grid G ∈ {0,1,2} and fatigue grid F ∈ {0..fatigue_max}
  - Static anchor mask A ∈ {0,1} creates an M4 corridor (self needs anchor density)
  - Fatigue implements M7: repeated constructive state (1) can flip to harm (2)
  - Absorption implements M9: overwhelming harm (2) forces a sticky basin

Provides:
  - step_motif_ca: true rule over (G,F,A)
  - Dataset generation
  - Count-based logistic regression learner + precision/recall diagnostics (ACT=predicted 1)
Drop-in compatible style with levin_ca_train.py (pure NumPy).
"""

from dataclasses import dataclass
import numpy as np
import time


@dataclass
class MotifParams:
    anchor_floor: int = 1      # min anchors in 3x3 to sustain constructive state (M4 corridor)
    anchor_prob: float = 0.30  # probability an anchor is present at a cell (controls corridor density)
    state_probs: tuple = (0.2, 0.5, 0.3)  # initial state distribution for G
    theta9: int = 6            # harm dominance threshold in neighborhood (M9 absorption)
    fatigue_inc: int = 1       # fatigue increment when in constructive state
    fatigue_dec: int = 1       # fatigue decay otherwise
    fatigue_flip: int = 4      # fatigue level at which repeated constructive flips to harm (M7)
    fatigue_max: int = 5       # cap on fatigue
    sticky_fatigue: int = 3    # if in harm with fatigue >= sticky_fatigue, stickier to leave


def _roll_neigh(arr: np.ndarray):
    """Return 3x3 neighborhoods with wrap-around; shape (..., 3, 3)."""
    H, W = arr.shape
    neigh = np.stack(
        [
            np.roll(np.roll(arr, di, axis=0), dj, axis=1)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
        ],
        axis=0,
    ).reshape(3, 3, H, W).transpose(2, 3, 0, 1)
    return neigh


def step_motif_ca(G: np.ndarray, F: np.ndarray, A: np.ndarray, params: MotifParams):
    """
    Apply the motif CA rule.
    Inputs:
      G: phenotype grid (H,W) in {0,1,2}
      F: fatigue grid (H,W) ints
      A: anchor mask (H,W) in {0,1}
    Returns:
      G_next, F_next (same shapes)
    """
    neigh_G = _roll_neigh(G)
    neigh_A = _roll_neigh(A)
    counts = np.zeros(G.shape + (3,), dtype=np.int16)
    for v in (0, 1, 2):
        counts[..., v] = (neigh_G == v).sum(axis=(-2, -1))
    anchor_counts = (neigh_A == 1).sum(axis=(-2, -1))  # 0..9
    center = neigh_G[..., 1, 1]

    # Base dominance (Levin-like): dominant state wins, ties stay center (M5 buffer)
    dom = counts.argmax(axis=-1)
    maxc = counts.max(axis=-1)
    tie = (counts == maxc[..., None]).sum(axis=-1) > 1
    G_next = np.where(tie, center, dom)

    # M4 corridor: constructive state (1) needs anchor support
    weak_corridor = (G_next == 1) & (anchor_counts < params.anchor_floor)
    G_next = np.where(weak_corridor, 0, G_next)

    # Update fatigue
    F_next = F + (G_next == 1) * params.fatigue_inc - (G_next != 1) * params.fatigue_dec
    F_next = np.clip(F_next, 0, params.fatigue_max)

    # M7: repetition flips constructive to harm when fatigue high
    flip_to_harm = (G_next == 1) & (F_next >= params.fatigue_flip)
    G_next = np.where(flip_to_harm, 2, G_next)

    # M9: absorbing harm basin when harm dominates neighborhood
    absorb = counts[..., 2] >= params.theta9
    G_next = np.where(absorb, 2, G_next)

    # Stickiness in harm when fatigued
    sticky = (center == 2) & (F >= params.sticky_fatigue)
    G_next = np.where(sticky, 2, G_next)

    # If we left constructive, fatigue can decay more aggressively next step; already handled by dec above.
    return G_next.astype(np.int8), F_next.astype(np.int8)


def acceptable_mask(G: np.ndarray, F: np.ndarray, anchor_counts: np.ndarray, params: MotifParams):
    """
    Define legitimacy manifold: constructive, anchored, not fatigued into flip territory.
    """
    return (G == 1) & (anchor_counts >= params.anchor_floor) & (F < params.fatigue_flip)


def anchor_counts_from_mask(A: np.ndarray):
    return (_roll_neigh(A) == 1).sum(axis=(-2, -1))


def features_from_state(G: np.ndarray, F: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Per-cell features:
      - count0, count1, count2 in 3x3 of G
      - center one-hot (3)
      - anchor count in 3x3
      - fatigue value
    Returns (H*W, 8).
    """
    H, W = G.shape
    neigh_G = _roll_neigh(G)
    neigh_A = _roll_neigh(A)
    feats = np.zeros((H, W, 8), dtype=np.float32)
    for v in (0, 1, 2):
        feats[..., v] = (neigh_G == v).sum(axis=(-2, -1))
    center = neigh_G[..., 1, 1]
    for v in (0, 1, 2):
        feats[..., 3 + v] = (center == v).astype(np.float32)
    feats[..., 6] = (neigh_A == 1).sum(axis=(-2, -1))  # anchor count
    feats[..., 7] = F.astype(np.float32)
    return feats.reshape(-1, feats.shape[-1])


def one_hot(y: np.ndarray, num_classes=3) -> np.ndarray:
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y.ravel()] = 1.0
    return out


def train_logreg(X_feats, Y, lr=5e-3, iters=300):
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


def eval_logreg(W, X_feats, Y):
    logits = X_feats @ W
    pred = logits.argmax(axis=1)
    acc = (pred == Y).mean()
    return acc, pred


def make_dataset(num_samples=200, H=32, W=32, seed=0, params=MotifParams()):
    rng = np.random.default_rng(seed)
    X_feats = []
    Y = []
    acceptable = []
    A_all = []
    for _ in range(num_samples):
        G = rng.choice(3, size=(H, W), p=params.state_probs).astype(np.int8)
        F = rng.integers(0, params.fatigue_max + 1, size=(H, W), dtype=np.int8)
        A = (rng.random((H, W)) < params.anchor_prob).astype(np.int8)  # corridor density
        G_next, F_next = step_motif_ca(G, F, A, params)
        feats = features_from_state(G, F, A)
        X_feats.append(feats)
        Y.append(G_next.ravel())
        anchor_counts = anchor_counts_from_mask(A)
        acceptable.append(acceptable_mask(G_next, F_next, anchor_counts, params).ravel())
        A_all.append(anchor_counts.ravel())
    X_feats = np.concatenate(X_feats, axis=0)
    Y = np.concatenate(Y, axis=0)
    acceptable = np.concatenate(acceptable, axis=0)
    A_all = np.concatenate(A_all, axis=0)
    return X_feats, Y, acceptable, A_all


def precision_recall(pred_act: np.ndarray, acceptable: np.ndarray):
    act_hits = (pred_act & acceptable).sum()
    act_count = pred_act.sum()
    acc_count = acceptable.sum()
    precision = act_hits / act_count if act_count > 0 else np.nan
    recall = act_hits / acc_count if acc_count > 0 else np.nan
    return precision, recall


def main():
    params = MotifParams()
    X_train, Y_train, acceptable_train, _ = make_dataset(num_samples=200, params=params, seed=0)
    X_test, Y_test, acceptable_test, _ = make_dataset(num_samples=50, params=params, seed=1)

    t0 = time.perf_counter()
    W = train_logreg(X_train, Y_train, lr=5e-3, iters=400)
    t1 = time.perf_counter()

    train_acc, pred_train = eval_logreg(W, X_train, Y_train)
    test_acc, pred_test = eval_logreg(W, X_test, Y_test)

    # Engagement metrics: ACT = predict state 1; acceptable from ground truth next state
    pred_act_train = pred_train == 1
    pred_act_test = pred_test == 1
    prec_tr, rec_tr = precision_recall(pred_act_train, acceptable_train)
    prec_te, rec_te = precision_recall(pred_act_test, acceptable_test)

    print("Motif CA (M4/M7/M9) rule learning via count-based log-reg")
    print(f"Train acc: {train_acc*100:.2f}%  Test acc: {test_acc*100:.2f}%  time={(t1 - t0)*1e3:.1f} ms")
    print(f"Engagement (pred==1 vs acceptable): Train precision={prec_tr:.3f} recall={rec_tr:.3f} | Test precision={prec_te:.3f} recall={rec_te:.3f}")


if __name__ == "__main__":
    main()

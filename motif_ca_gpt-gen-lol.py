
"""
motif_ca.py

Two-layer ternary cellular automaton (G,F) + anchor field A, explicitly exhibiting:
  - M4 corridor (anchor-dependent coherence for state=1)
  - M7 tolerance/variance rim (fatigue flips 1->2 under repetition)
  - M9 retire/prohibit basin (dominant-harm absorption/stickiness)

Includes:
  1) Ground-truth CA (step_true)
  2) Tiny count-based learner (multiclass logistic regression; numpy only)
  3) Sweep harness analogous to your trading PR sweep:
       - learn tau_off, k_off until precision breaks (precision = P(acceptable | ACT))
  4) Simple visualiser (side-by-side True CA vs Learned CA)

Designed to mirror the style of your existing levin_ca_train.py + ca_visualiser.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

# ----------------------------
# Utilities
# ----------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)

def one_hot_int(x: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros(x.shape + (num_classes,), dtype=np.float32)
    # flatten write, then reshape back
    flat = out.reshape(-1, num_classes)
    idx = x.ravel().astype(np.int64)
    flat[np.arange(idx.size), idx] = 1.0
    return out

def neighborhood(grid: np.ndarray) -> np.ndarray:
    """
    Build 3x3 neighborhoods with wraparound.
    Returns neigh shape (H,W,3,3).
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
    return neigh

def counts3(neigh: np.ndarray) -> np.ndarray:
    """
    neigh: (H,W,3,3) of ints in {0,1,2}
    returns counts: (H,W,3)
    """
    H, W = neigh.shape[:2]
    c = np.zeros((H, W, 3), dtype=np.int16)
    for v in (0, 1, 2):
        c[..., v] = (neigh == v).sum(axis=(-2, -1))
    return c

def anchor_density(A: np.ndarray) -> np.ndarray:
    """
    A: (H,W) binary anchor mask
    returns local density in 3x3 neighborhood as float in [0,1]
    """
    neighA = neighborhood(A.astype(np.int8))
    dens = neighA.sum(axis=(-2, -1)).astype(np.float32) / 9.0
    return dens

# ----------------------------
# Motif CA: ground truth dynamics
# ----------------------------

@dataclass
class MotifCAParams:
    # M4 corridor
    anchor_floor: float = 0.30        # minimum local anchor density to sustain state=1
    corridor_boost: bool = True       # if True, corridor increases persistence in ties

    # M7 tolerance / fatigue
    fatigue_gain: int = 1             # add when G==1
    fatigue_decay: int = 1            # subtract when G!=1
    fatigue_flip: int = 2             # if F >= this AND base wants 1 => flip to 2
    fatigue_cap: int = 2              # F in {0,1,2}

    # M9 retire/prohibit
    theta9: int = 6                   # if count2 >= theta9 in neighborhood => force 2
    stickiness: bool = True           # make 2 hard to recover from

def step_true(G: np.ndarray, F: np.ndarray, A: np.ndarray, p: MotifCAParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ground-truth update for (G,F).
      - Base rule: dominant neighborhood state; ties stay as center (M5 buffer).
      - M4: if base wants 1 but anchor density below floor => collapse to 0.
      - M7: fatigue accumulates on 1; if too high, 1 flips to 2 under repetition.
      - M9: if neighborhood harm dominates strongly => force 2; 2 may be sticky.
    """
    neigh = neighborhood(G)
    c = counts3(neigh)                       # (H,W,3)
    center = neigh[..., 1, 1]                # (H,W)
    dom = c.argmax(axis=-1).astype(np.int8)  # (H,W)

    # tie -> stay center (M5)
    maxc = c.max(axis=-1)
    tie = (c == maxc[..., None]).sum(axis=-1) > 1
    base = np.where(tie, center, dom).astype(np.int8)

    # local anchor density
    ad = anchor_density(A)

    # M4 corridor: if trying to be constructive (1) but insufficient anchor => fall to 0
    if p.corridor_boost:
        # optional: in tie cases, corridor nudges toward 1 if anchored
        # (i.e., allow 1 persistence in ambiguous zones when anchored)
        base = np.where((tie) & (ad >= p.anchor_floor) & (center == 1), 1, base)

    base = np.where((base == 1) & (ad < p.anchor_floor), 0, base).astype(np.int8)

    # update fatigue
    F_next = F.copy().astype(np.int8)
    inc = (G == 1).astype(np.int8) * p.fatigue_gain
    dec = (G != 1).astype(np.int8) * p.fatigue_decay
    F_next = np.clip(F_next + inc - dec, 0, p.fatigue_cap).astype(np.int8)

    # M7: repetition flips
    base = np.where((base == 1) & (F_next >= p.fatigue_flip), 2, base).astype(np.int8)

    # M9: dominant harm absorption
    base = np.where(c[..., 2] >= p.theta9, 2, base).astype(np.int8)

    # Stickiness: once 2, require strong evidence + anchor to recover
    if p.stickiness:
        # allow recovery only when harm count is low AND anchored AND fatigue low
        recover = (G == 2) & (c[..., 2] <= 2) & (ad >= p.anchor_floor) & (F_next <= 0)
        base = np.where(recover, 1, base).astype(np.int8)
        # otherwise keep 2 if already 2 and neighborhood harm is moderate
        keep2 = (G == 2) & (c[..., 2] >= 3)
        base = np.where(keep2, 2, base).astype(np.int8)

    G_next = base.astype(np.int8)
    return G_next, F_next

# ----------------------------
# Features + learner
# ----------------------------

def features(G: np.ndarray, F: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Per-cell features:
      - count0,count1,count2 in 3x3 of G           (3)
      - center one-hot of G                        (3)
      - fatigue one-hot of F (0,1,2)               (3)
      - anchor density in 3x3 of A                 (1)
    Total D=10. Returns (H*W, 10).
    """
    neigh = neighborhood(G)
    c = counts3(neigh).astype(np.float32)          # (H,W,3)
    center = neigh[..., 1, 1].astype(np.int8)      # (H,W)
    oh_center = one_hot_int(center, 3)             # (H,W,3)
    oh_f = one_hot_int(F.astype(np.int8), 3)       # (H,W,3)
    ad = anchor_density(A)[..., None]              # (H,W,1)

    feats = np.concatenate([c, oh_center, oh_f, ad], axis=-1).astype(np.float32)  # (H,W,10)
    return feats.reshape(-1, feats.shape[-1])

def train_logreg(X: np.ndarray, Y: np.ndarray, lr: float = 5e-3, iters: int = 300) -> np.ndarray:
    """
    Multiclass logistic regression (SGD over full batch).
      X: (N,D), Y: (N,) in {0,1,2}
    Returns W: (D,3)
    """
    N, D = X.shape
    C = 3
    W = np.zeros((D, C), dtype=np.float32)

    Y_one = np.zeros((N, C), dtype=np.float32)
    Y_one[np.arange(N), Y.astype(np.int64)] = 1.0

    for _ in range(iters):
        logits = X @ W
        probs = softmax(logits)
        grad = X.T @ (probs - Y_one) / N
        W -= lr * grad
    return W

def predict_probs(W: np.ndarray, G: np.ndarray, F: np.ndarray, A: np.ndarray) -> np.ndarray:
    feats = features(G, F, A)
    logits = feats @ W
    probs = softmax(logits)
    # reshape to (H,W,3)
    H, Wd = G.shape
    return probs.reshape(H, Wd, 3)

def step_learned(G: np.ndarray, F: np.ndarray, A: np.ndarray, W: np.ndarray, p: MotifCAParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learned CA step: predict next G by argmax classifier, then update fatigue with same dynamics.
    (This mirrors your True vs Learned visualiser style.)
    """
    probs = predict_probs(W, G, F, A)
    G_next = probs.argmax(axis=-1).astype(np.int8)

    # update fatigue using the same rule as truth (so divergence is mostly from G prediction)
    F_next = F.copy().astype(np.int8)
    inc = (G == 1).astype(np.int8) * p.fatigue_gain
    dec = (G != 1).astype(np.int8) * p.fatigue_decay
    F_next = np.clip(F_next + inc - dec, 0, p.fatigue_cap).astype(np.int8)

    return G_next, F_next

# ----------------------------
# Dataset generation
# ----------------------------

def make_anchor(H: int, W: int, p_anchor: float = 0.08, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = (rng.random((H, W)) < p_anchor).astype(np.int8)
    return A

def make_dataset(
    num_grids: int = 200,
    H: int = 32,
    W: int = 32,
    steps_per_grid: int = 4,
    seed: int = 0,
    params: MotifCAParams = MotifCAParams(),
    anchor_p: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates training pairs (X_feats, Y_nextG_flat).
    """
    rng = np.random.default_rng(seed)
    A = make_anchor(H, W, p_anchor=anchor_p, seed=seed + 123)

    X_list = []
    Y_list = []

    for i in range(num_grids):
        G = rng.integers(0, 3, size=(H, W), dtype=np.int8)
        F = np.zeros((H, W), dtype=np.int8)

        for _ in range(steps_per_grid):
            # features at time t
            X = features(G, F, A)
            G_next, F_next = step_true(G, F, A, params)
            Y = G_next.ravel()

            X_list.append(X)
            Y_list.append(Y)

            G, F = G_next, F_next

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0).astype(np.int8)
    return X_all, Y_all

# ----------------------------
# Acceptability (Legitimacy manifold) and CA "engagement" gate
# ----------------------------

def acceptable_mask(G: np.ndarray, F: np.ndarray, A: np.ndarray, p: MotifCAParams) -> np.ndarray:
    """
    A simple legitimacy predicate for the CA world:
      acceptable iff
        - not in harm state (G != 2), AND
        - anchored enough for coherent 'self' (anchor_density >= floor), AND
        - not fatigued to rim (F < fatigue_flip)
    This creates an M4 corridor (anchor) and an M7 rim (fatigue).
    """
    ad = anchor_density(A)
    return (G != 2) & (ad >= p.anchor_floor) & (F < p.fatigue_flip)

@dataclass
class SweepResult:
    tau_off: float
    k_off: int
    acceptable: float
    precision: float
    recall: float
    act_cells: int
    hold_pct: float

def sweep_hysteresis_on_ca(
    W: np.ndarray,
    *,
    params: MotifCAParams,
    H: int = 96,
    Wd: int = 96,
    T: int = 250,
    seed: int = 0,
    anchor_p: float = 0.08,
    tau_on: float = 0.50,
    tau_off_values: Iterable[float] = (0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15),
    k_off_values: Iterable[int] = (1, 2, 3),
    precision_floor: float = 0.80,
) -> List[SweepResult]:
    """
    Run a sweep of exit thresholds (tau_off) and persistence (k_off) on CA "engagement".
    ACT is an internal per-cell engagement flag updated by hysteresis on conf1 = P(class=1).

    Metrics mirror your trading sweep:
      acceptable% = fraction of cells acceptable (over time)
      precision   = P(acceptable | ACT)
      recall      = P(ACT | acceptable)
      act_cells   = total ACT cell-ticks
      hold%       = fraction of HOLD cell-ticks
    """
    rng = np.random.default_rng(seed)
    A = make_anchor(H, Wd, p_anchor=anchor_p, seed=seed + 555)
    G = rng.integers(0, 3, size=(H, Wd), dtype=np.int8)
    F = np.zeros((H, Wd), dtype=np.int8)

    # For a fair sweep comparison, we keep the same initial state for each config
    G0, F0 = G.copy(), F.copy()

    results: List[SweepResult] = []

    for k_off in k_off_values:
        for tau_off in tau_off_values:
            # reset world
            G, F = G0.copy(), F0.copy()

            # engagement automaton state (per cell)
            act = np.zeros((H, Wd), dtype=bool)
            off_run = np.zeros((H, Wd), dtype=np.int16)

            # accumulators
            tot = 0
            acc_ok = 0
            act_tot = 0
            act_ok = 0

            for _ in range(T):
                probs = predict_probs(W, G, F, A)
                conf1 = probs[..., 1]

                ok = acceptable_mask(G, F, A, params)

                # update ACT/HOLD with hysteresis + k_off persistence
                enter = (~act) & (conf1 >= tau_on)
                act = act | enter

                low = act & (conf1 < tau_off)
                off_run = np.where(low, off_run + 1, 0)
                exit_ = act & (off_run >= k_off)
                act = act & (~exit_)
                off_run = np.where(act, off_run, 0)

                # metrics at this timestep
                tot += ok.size
                acc_ok += int(ok.sum())
                act_tot += int(act.sum())
                act_ok += int((act & ok).sum())

                # advance the true world (independent of gating)
                G, F = step_true(G, F, A, params)

            acceptable = acc_ok / max(1, tot)
            precision = (act_ok / act_tot) if act_tot > 0 else 1.0
            recall = (act_ok / acc_ok) if acc_ok > 0 else 0.0
            hold_pct = 1.0 - (act_tot / max(1, tot))

            results.append(
                SweepResult(
                    tau_off=float(tau_off),
                    k_off=int(k_off),
                    acceptable=float(acceptable),
                    precision=float(precision),
                    recall=float(recall),
                    act_cells=int(act_tot),
                    hold_pct=float(hold_pct),
                )
            )

            # optional early stop for this k_off if we already broke precision hard at high tau_off
            if precision < precision_floor and tau_off == max(tau_off_values):
                break

    return results

def write_sweep_csv(path: str, rows: List[SweepResult]) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_off", "k_off", "acceptable", "precision", "recall", "act_cells", "hold_pct"])
        for r in rows:
            w.writerow([r.tau_off, r.k_off, r.acceptable, r.precision, r.recall, r.act_cells, r.hold_pct])

# ----------------------------
# Visualiser (True vs Learned)
# ----------------------------

def run_visualiser(
    *,
    params: MotifCAParams = MotifCAParams(),
    seed: int = 0,
    H: int = 160,
    Wd: int = 160,
    anchor_p: float = 0.08,
    train_grids: int = 200,
    steps_per_grid: int = 4,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(seed)
    A = make_anchor(H, Wd, p_anchor=anchor_p, seed=seed + 555)

    # init
    G0 = rng.integers(0, 3, size=(H, Wd), dtype=np.int8)
    F0 = np.zeros((H, Wd), dtype=np.int8)

    # train learner on small grids
    Xtr, Ytr = make_dataset(
        num_grids=train_grids,
        H=32,
        W=32,
        steps_per_grid=steps_per_grid,
        seed=seed,
        params=params,
        anchor_p=anchor_p,
    )
    t0 = time.perf_counter()
    W_lr = train_logreg(Xtr, Ytr, lr=5e-3, iters=350)
    t1 = time.perf_counter()
    print(f"[train] logreg trained in {(t1-t0)*1e3:.1f} ms | N={Xtr.shape[0]} D={Xtr.shape[1]}")

    Gt, Ft = G0.copy(), F0.copy()
    Gl, Fl = G0.copy(), F0.copy()

    cmap = ListedColormap(["black", "gray", "white"])  # 0,1,2
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    im1 = ax1.imshow(Gt, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    im2 = ax2.imshow(Gl, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")

    ax1.set_title("True Motif CA (G)")
    ax2.set_title("Learned Motif CA (G)")
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    txt = fig.text(0.5, 0.01, "", ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    paused = False
    step_once = False
    delay = 0.02

    def dist(g):
        tot = g.size
        return ((g == 0).sum() / tot, (g == 1).sum() / tot, (g == 2).sum() / tot)

    def on_key(event):
        nonlocal paused, step_once, delay
        if event.key == " ":
            paused = not paused
        elif event.key == "n":
            step_once = True
        elif event.key == "r":
            nonlocal Gt, Ft, Gl, Fl
            Gt, Ft = rng.integers(0, 3, size=(H, Wd), dtype=np.int8), np.zeros((H, Wd), dtype=np.int8)
            Gl, Fl = Gt.copy(), Ft.copy()
        elif event.key in ("+", "="):
            delay = max(0.0, delay * 0.8)
        elif event.key in ("-", "_"):
            delay = min(1.0, delay / 0.8)
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)

    while plt.fignum_exists(fig.number):
        do_step = (not paused) or step_once
        step_once = False

        if do_step:
            Gt, Ft = step_true(Gt, Ft, A, params)
            Gl, Fl = step_learned(Gl, Fl, A, W_lr, params)

        im1.set_data(Gt)
        im2.set_data(Gl)

        p0, p1, p2 = dist(Gt)
        txt.set_text(f"{'paused' if paused else 'running'} | delay={delay*1000:.0f}ms | true dist: 0={p0:.2f} 1={p1:.2f} 2={p2:.2f}")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if delay > 0:
            time.sleep(delay)

# ----------------------------
# Main entry points
# ----------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "sweep", "viz"], default="sweep")
    ap.add_argument("--out", type=str, default="logs/ca_pr_curve.csv")

    # dataset/training
    ap.add_argument("--train_grids", type=int, default=200)
    ap.add_argument("--steps_per_grid", type=int, default=4)

    # sweep
    ap.add_argument("--tau_on", type=float, default=0.50)
    ap.add_argument("--tau_off", type=str, default="0.45,0.40,0.35,0.30,0.25,0.20,0.15")
    ap.add_argument("--k_off", type=str, default="1,2,3")
    ap.add_argument("--precision_floor", type=float, default=0.80)

    # world
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--W", type=int, default=96)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--anchor_p", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    params = MotifCAParams()

    if args.mode == "viz":
        run_visualiser(
            params=params,
            seed=args.seed,
            H=max(args.H, 120),
            Wd=max(args.W, 120),
            anchor_p=args.anchor_p,
            train_grids=args.train_grids,
            steps_per_grid=args.steps_per_grid,
        )
        return

    # Train learner (used by sweep and train mode)
    Xtr, Ytr = make_dataset(
        num_grids=args.train_grids,
        H=32,
        W=32,
        steps_per_grid=args.steps_per_grid,
        seed=args.seed,
        params=params,
        anchor_p=args.anchor_p,
    )
    t0 = time.perf_counter()
    W_lr = train_logreg(Xtr, Ytr, lr=5e-3, iters=350)
    t1 = time.perf_counter()
    print(f"[train] logreg trained in {(t1-t0)*1e3:.1f} ms | N={Xtr.shape[0]} D={Xtr.shape[1]}")

    if args.mode == "train":
        # quick sanity eval on a held-out small set
        Xte, Yte = make_dataset(
            num_grids=max(25, args.train_grids // 4),
            H=32,
            W=32,
            steps_per_grid=args.steps_per_grid,
            seed=args.seed + 1,
            params=params,
            anchor_p=args.anchor_p,
        )
        probs = softmax(Xte @ W_lr)
        pred = probs.argmax(axis=1).astype(np.int8)
        acc = (pred == Yte).mean()
        print(f"[eval] next-G accuracy: {acc*100:.2f}%")
        return

    # Sweep mode
    tau_off_values = [float(x) for x in args.tau_off.split(",") if x.strip()]
    k_off_values = [int(x) for x in args.k_off.split(",") if x.strip()]

    rows = sweep_hysteresis_on_ca(
        W_lr,
        params=params,
        H=args.H,
        Wd=args.W,
        T=args.T,
        seed=args.seed,
        anchor_p=args.anchor_p,
        tau_on=args.tau_on,
        tau_off_values=tau_off_values,
        k_off_values=k_off_values,
        precision_floor=args.precision_floor,
    )

    # print like your trading sweep
    for r in rows:
        print(
            f"tau_off={r.tau_off:.2f}  k_off={r.k_off}  acceptable={r.acceptable:.3f}  "
            f"precision={r.precision:.3f}  recall={r.recall:.3f}  act_cells={r.act_cells}  hold%={r.hold_pct:.3f}"
        )

    # write output
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_sweep_csv(args.out, rows)
    print(f"Wrote sweep metrics to {args.out}")

if __name__ == "__main__":
    main()

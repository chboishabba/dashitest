"""
Sweep hysteresis thresholds on the motif CA using classifier confidence as evidence.
This mirrors the trading PR sweep: ACT vs acceptable (legitimacy) with hysteresis (tau_on/tau_off, k_on/k_off).
"""

import argparse
import numpy as np
import pandas as pd
from motif_ca import (
    MotifParams,
    make_dataset as make_dataset_motif,
    train_logreg as train_logreg_motif,
    step_motif_ca,
    features_from_state,
    acceptable_mask,
    anchor_counts_from_mask,
)


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def generate_trajectory(W, params: MotifParams, H=32, Wd=32, steps=200, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.choice(3, size=(H, Wd), p=params.state_probs).astype(np.int8)
    F = rng.integers(0, params.fatigue_max + 1, size=(H, Wd), dtype=np.int8)
    A = (rng.random((H, Wd)) < params.anchor_prob).astype(np.int8)
    anchor_counts = anchor_counts_from_mask(A)

    conf_seq = []
    acceptable_seq = []
    for _ in range(steps):
        feats = features_from_state(G, F, A)  # (H*W, 8)
        probs = softmax(feats @ W)            # (H*W, 3)
        conf1 = probs[:, 1]                   # evidence for constructive state
        conf_seq.append(conf1)
        acceptable_seq.append(acceptable_mask(G, F, anchor_counts, params).ravel())
        G, F = step_motif_ca(G, F, A, params)

    conf_arr = np.stack(conf_seq, axis=0)          # (T, N)
    acceptable_arr = np.stack(acceptable_seq, axis=0)  # (T, N)
    return conf_arr, acceptable_arr


def apply_hysteresis(conf_arr, tau_on, tau_off, k_on=1, k_off=1):
    """
    conf_arr: (T, N) confidence for constructive state per cell.
    Returns act bool array (T, N) after hysteresis.
    """
    T, N = conf_arr.shape
    act = np.zeros_like(conf_arr, dtype=bool)
    engaged = np.zeros(N, dtype=bool)
    on_cnt = np.zeros(N, dtype=int)
    off_cnt = np.zeros(N, dtype=int)
    for t in range(T):
        c = conf_arr[t]
        enter = c >= tau_on
        exitc = c < tau_off
        on_cnt = np.where(enter, on_cnt + 1, 0)
        off_cnt = np.where(exitc, off_cnt + 1, 0)
        engage_now = (on_cnt >= k_on) | engaged
        drop_now = exitc & (off_cnt >= k_off)
        engaged = engage_now & (~drop_now)
        act[t] = engaged
    return act


def precision_recall(act_flat, acceptable_flat):
    hits = (act_flat & acceptable_flat).sum()
    act_count = act_flat.sum()
    acc_count = acceptable_flat.sum()
    precision = hits / act_count if act_count > 0 else np.nan
    recall = hits / acc_count if acc_count > 0 else np.nan
    return precision, recall, act_count


def engagement_bins(conf_arr, act_arr, acceptable_arr, bins=20):
    """
    Compute per-bin engagement rates conditioned on acceptable.
    Returns list of dicts with bin_center, engagement.
    """
    conf_flat = conf_arr.ravel()
    act_flat = act_arr.ravel()
    acc_flat = acceptable_arr.ravel()
    mask = np.isfinite(conf_flat) & acc_flat
    conf_flat = conf_flat[mask]
    act_flat = act_flat[mask]
    if conf_flat.size == 0:
        return []
    bins = max(5, bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    cat = pd.cut(conf_flat, edges, include_lowest=True, labels=labels)
    grouped = pd.Series(act_flat).groupby(cat).mean()
    rows = []
    for center, rate in grouped.items():
        rows.append({"bin_center": float(center), "engagement": float(rate)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="Time steps to simulate")
    ap.add_argument("--grid", type=int, default=32, help="Grid side length (square)")
    ap.add_argument("--train_samples", type=int, default=400, help="Samples to train motif classifier")
    ap.add_argument("--tau_on", type=float, default=0.5, help="Entry threshold")
    ap.add_argument("--tau_off", type=float, nargs="+", default=[0.30, 0.25, 0.20, 0.15, 0.10, 0.05], help="Exit thresholds to sweep")
    ap.add_argument("--k_on", type=int, default=1, help="Consecutive steps above tau_on to engage")
    ap.add_argument("--k_off", type=int, default=1, help="Consecutive steps below tau_off to disengage")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    ap.add_argument(
        "--out_bins",
        type=str,
        default=None,
        help="Optional CSV for engagement surface (long form: tau_off, bin_center, engagement).",
    )
    ap.add_argument("--bins", type=int, default=20, help="Bins for actionability engagement surface.")
    args = ap.parse_args()

    params = MotifParams()
    # Train classifier on synthetic motif dataset
    X, Y, _, _ = make_dataset_motif(num_samples=args.train_samples, H=32, W=32, seed=args.seed, params=params)
    W = train_logreg_motif(X, Y, lr=5e-3, iters=400)

    conf_arr, acceptable_arr = generate_trajectory(W, params, H=args.grid, Wd=args.grid, steps=args.steps, seed=args.seed)
    rows = []
    bin_rows = []
    for tau_off in args.tau_off:
        act = apply_hysteresis(conf_arr, tau_on=args.tau_on, tau_off=tau_off, k_on=args.k_on, k_off=args.k_off)
        precision, recall, act_count = precision_recall(act.ravel(), acceptable_arr.ravel())
        acceptable_pct = float(acceptable_arr.mean())
        hold_pct = 1.0 - act.mean()
        rows.append(
            {
                "tau_on": args.tau_on,
                "tau_off": tau_off,
                "k_on": args.k_on,
                "k_off": args.k_off,
                "acceptable_pct": acceptable_pct,
                "precision": precision,
                "recall": recall,
                "act_bars": int(act_count),
                "hold_pct": hold_pct,
            }
        )
        if args.out_bins:
            for row in engagement_bins(conf_arr, act, acceptable_arr, bins=args.bins):
                row["tau_off"] = tau_off
                bin_rows.append(row)
        print(
            f"tau_off={tau_off:.2f} precision={precision:.3f} recall={recall:.3f} acceptable={acceptable_pct:.3f} "
            f"act_cells={act_count} hold%={hold_pct:.3f}"
        )

    if args.out:
        df = pd.DataFrame(rows)
        df.to_csv(args.out, index=False)
        print(f"Wrote sweep metrics to {args.out}")
    if args.out_bins and bin_rows:
        pd.DataFrame(bin_rows).to_csv(args.out_bins, index=False)
        print(f"Wrote engagement surface to {args.out_bins}")


if __name__ == "__main__":
    main()

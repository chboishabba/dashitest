"""
plot_temporal_homology.py
-------------------------
Temporal homology persistence for acceptable manifold.
Bins actionability over time, finds connected acceptable components per time slice,
tracks births/deaths, and plots persistence length histogram + timeline of active components.

Usage:
  PYTHONPATH=. python trading/scripts/plot_temporal_homology.py --log logs/trading_log.csv --save homology_time.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def slice_components(mask_slice):
    """Return list of (start_idx, end_idx) contiguous 1-runs in a 1D binary mask."""
    comps = []
    in_run = False
    start = 0
    for i, v in enumerate(mask_slice):
        if v and not in_run:
            start = i
            in_run = True
        if not v and in_run:
            comps.append((start, i - 1))
            in_run = False
    if in_run:
        comps.append((start, len(mask_slice) - 1))
    return comps


def track_components(mask_time):
    """
    mask_time: array shape (T, B) of 0/1 (acceptable bins over time).
    Returns list of lifetimes (length in time slices) and timeline matrix of component IDs per time/bin.
    """
    T, B = mask_time.shape
    lifetimes = []
    comp_id = np.full_like(mask_time, fill_value=-1, dtype=int)
    next_id = 0
    active = {}  # id -> (start_t, last_slice_indices)
    for t in range(T):
        comps = slice_components(mask_time[t])
        used_ids = set()
        for start, end in comps:
            # try to match with previous active components by overlap
            matched_id = None
            best_overlap = 0
            for cid, (start_t, (prev_start, prev_end)) in list(active.items()):
                overlap = min(end, prev_end) - max(start, prev_start) + 1
                if overlap > 0 and overlap > best_overlap:
                    best_overlap = overlap
                    matched_id = cid
            if matched_id is None:
                matched_id = next_id
                next_id += 1
                active[matched_id] = (t, (start, end))
            else:
                active[matched_id] = (active[matched_id][0], (start, end))
            comp_id[t, start : end + 1] = matched_id
            used_ids.add(matched_id)
        # retire inactive
        for cid in list(active.keys()):
            if cid not in used_ids:
                start_t, _ = active.pop(cid)
                lifetimes.append(t - start_t)
    # close remaining
    for cid, (start_t, _) in active.items():
        lifetimes.append(T - start_t)
    return lifetimes, comp_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Trading log with acceptable/actionability/t")
    ap.add_argument("--time_bins", type=int, default=100, help="Time bins")
    ap.add_argument("--act_bins", type=int, default=20, help="Actionability bins")
    ap.add_argument("--save", type=str, default=None, help="Optional output image path")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    for col in ("acceptable", "actionability", "t"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in log.")
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)
    acc = df["acceptable"].astype(bool).to_numpy()
    mask = np.isfinite(t) & np.isfinite(a)
    t = t[mask]
    a = a[mask]
    acc = acc[mask]

    time_edges = np.linspace(t.min(), t.max(), args.time_bins + 1)
    act_edges = np.linspace(0.0, 1.0, args.act_bins + 1)
    counts, _, _ = np.histogram2d(t, a, bins=[time_edges, act_edges])
    acc_counts, _, _ = np.histogram2d(t, a, bins=[time_edges, act_edges], weights=acc.astype(int))
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(counts > 0, acc_counts / counts, 0.0)
    # binarize acceptable mask per bin (majority vote)
    mask_time = (density >= 0.5).astype(int)

    lifetimes, comp_id = track_components(mask_time)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].hist(lifetimes, bins=30, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("persistence length (time bins)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Acceptable component lifetimes")

    im = axes[1].imshow(comp_id, origin="lower", aspect="auto", cmap="tab20")
    axes[1].set_xlabel("actionability bins")
    axes[1].set_ylabel("time bins")
    axes[1].set_title("Component IDs over time")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="component id")

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()

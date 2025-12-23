"""
ca_epistemic_tape.py
--------------------
Trading-driven CA visualization ("epistemic tape"):
- X-axis: lag (new bars injected at the left, older shift right)
- Y-axis: feature channels (return sign, vol regime, stress proxy, permission proxy)
- Cell state: sign s∈{-1,0,+1}, phase φ∈{-1,0,+1}, gate g∈{-1,0,+1}, fatigue u∈[0..15]
- Local rule: gate first (M4/M7/M9 pressures), then state update with phase bias for motion.

Usage:
  PYTHONPATH=. python scripts/ca_epistemic_tape.py --csv data/raw/stooq/btc_intraday_1s.csv
  (Any CSV with a close column; falls back to synthetic if missing.)

Outputs:
  - Snapshots (gate/phase/sign) over time
  - Time-series stats (change rates, ACT/HOLD/BAN fractions, fatigue)
  - Multiscale change-rate curve (coarse vs fine)
"""

import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure project root on path for run_trader import
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run_trader import load_prices


def triadic_bin(x, pos_thr, neg_thr=None):
    """Map a numeric array to {-1,0,+1} with symmetric thresholds."""
    x = np.asarray(x, dtype=float)
    if neg_thr is None:
        neg_thr = -pos_thr
    out = np.zeros_like(x, dtype=np.int8)
    out[x > pos_thr] = 1
    out[x < neg_thr] = -1
    return out


def build_features(price, volume, window=50):
    """
    Build triadic feature channels from price/volume.
    Channels (rows):
      0: return sign (short-term)
      1: volatility regime (rolling std z-scored)
      2: stress proxy (abs return vs rolling std)
      3: permission proxy (same as return sign for simplicity)
    """
    price = np.asarray(price, dtype=float)
    vol = pd.Series(price).pct_change().rolling(window).std().to_numpy()
    ret = np.diff(price, prepend=price[0]) / np.maximum(price, 1e-9)
    ret_z = (ret - pd.Series(ret).rolling(window).mean().to_numpy()) / (vol + 1e-9)
    stress = np.abs(ret) / (vol + 1e-9)

    ch0 = triadic_bin(ret, pos_thr=np.nanmedian(np.abs(ret)) * 0.5)
    ch1 = triadic_bin(ret_z, pos_thr=0.75)
    ch2 = triadic_bin(stress, pos_thr=np.nanpercentile(stress[np.isfinite(stress)], 75))
    ch3 = ch0.copy()
    feats = np.stack([ch0, ch1, ch2, ch3], axis=0)
    # replace NaN from initial window
    feats = np.nan_to_num(feats, nan=0.0).astype(np.int8)
    return feats


def neighbor_counts(x):
    """Counts of + and - in 3x3 neighborhood (including self) on a 2D grid (channels x width)."""
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    p = np.zeros_like(x, dtype=np.int16)
    n = np.zeros_like(x, dtype=np.int16)
    for dy, dx in shifts:
        y = np.roll(np.roll(x, dy, axis=0), dx, axis=1)
        p += (y == 1)
        n += (y == -1)
    return p, n


def phase_gradient(phi):
    """Simple phase gradient proxy: difference of neighbors on 2D grid."""
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    grad = np.zeros_like(phi, dtype=np.int16)
    for dy, dx in shifts:
        y = np.roll(np.roll(phi, dy, axis=0), dx, axis=1)
        grad += y
    return grad


def run_ca_tape(feats, width=128, fatigue_thr=8, ban_conflict=6, seed=0):
    """
    Run the CA on a feature tape.
    feats: (C, T) triadic features
    width: how many lags to keep (X-axis)
    Returns snapshots, stats, and state history for multiscale analysis.
    """
    rng = np.random.default_rng(seed)
    C, T = feats.shape
    # state tensors: (C, width)
    s = np.zeros((C, width), dtype=np.int8)    # sign/content
    phi = np.zeros((C, width), dtype=np.int8)  # phase/chirality
    g = np.zeros((C, width), dtype=np.int8)    # gate
    u = np.zeros((C, width), dtype=np.int8)    # fatigue 0..15

    snapshots = {}
    stats = {k: [] for k in ["chg_s", "chg_g", "act", "hold", "ban", "fatigue", "m4", "m7", "m9"]}
    history_s = []
    snap_ts = [0, max(1, T // 4), max(1, T // 2), max(1, 3 * T // 4), T - 1]

    for t in range(T):
        obs = feats[:, t]
        # inject new column on the left
        s = np.roll(s, shift=1, axis=1)
        phi = np.roll(phi, shift=1, axis=1)
        g = np.roll(g, shift=1, axis=1)
        u = np.roll(u, shift=1, axis=1)
        s[:, 0] = obs
        phi[:, 0] = obs  # tie phase to content on injection
        g[:, 0] = np.where(obs >= 0, 1, 0)
        u[:, 0] = 0

        # neighbor summaries
        p_s, n_s = neighbor_counts(s)
        conflict = np.minimum(p_s, n_s)  # tension
        turb = p_s + n_s
        grad_phi = phase_gradient(phi)

        # gate update
        g2 = g.copy()
        m9 = (conflict >= ban_conflict) & (turb >= 8)
        g2[m9] = -1
        m4 = (p_s - n_s >= 3) & (~m9)
        g2[m4] = 1
        # fatigue
        engaged = (g == 1) & (s != 0)
        u2 = u.copy()
        u2[engaged] = np.minimum(u2[engaged] + 1, 15)
        u2[~engaged] = np.maximum(u2[~engaged] - 1, 0)
        m7 = (u2 >= fatigue_thr) & (g2 == 1) & (~m9)
        g2[m7] = -1
        hold = (g2 != -1) & (np.abs(p_s - n_s) < 2)
        g2[hold] = 0

        # state update
        s2 = s.copy()
        phi2 = phi.copy()
        # banned → quench
        s2[g2 == -1] = 0
        phi2[g2 == -1] = 0
        # act → phase-biased update
        act = (g2 == 1)
        if np.any(act):
            score = (p_s - n_s) + 0.5 * grad_phi
            s2[act & (score > 1)] = 1
            s2[act & (score < -1)] = -1
            s2[act & (np.abs(score) <= 1)] = 0
            phi2[act] = np.sign(grad_phi[act]).astype(np.int8)
        # hold → mild decay of phase
        phi2[g2 == 0] = (phi2[g2 == 0] * 0).astype(np.int8)

        # stats
        stats["chg_s"].append(float(np.mean(s2 != s)))
        stats["chg_g"].append(float(np.mean(g2 != g)))
        stats["act"].append(float(np.mean(g2 == 1)))
        stats["hold"].append(float(np.mean(g2 == 0)))
        stats["ban"].append(float(np.mean(g2 == -1)))
        stats["fatigue"].append(float(u2.mean()))
        stats["m4"].append(float(np.mean(m4)))
        stats["m7"].append(float(np.mean(m7)))
        stats["m9"].append(float(np.mean(m9)))

        s, phi, g, u = s2, phi2, g2, u2
        history_s.append(s.copy())
        if t in snap_ts:
            snapshots[t] = (s.copy(), phi.copy(), g.copy(), u.copy())

    return snapshots, stats, history_s


def coarse_change_rate(history, k):
    """Compute average change rate at coarsening factor k across history."""
    rates = []
    # history is list of (C, W) arrays (sign only)
    prev = history[0][:, : (history[0].shape[1] // k) * k]
    prev = prev.reshape(prev.shape[0], -1, k).mean(axis=2)
    for cur in history[1:]:
        cur2 = cur[:, : (cur.shape[1] // k) * k]
        cur2 = cur2.reshape(cur2.shape[0], -1, k).mean(axis=2)
        rates.append(float(np.mean(cur2 != prev)))
        prev = cur2
    return float(np.mean(rates))


def plot_snapshots(snaps):
    rows = len(snaps)
    fig, axes = plt.subplots(rows, 3, figsize=(10, 2.5 * rows))
    if rows == 1:
        axes = np.array([axes])
    for r, (t, (s, phi, g, u)) in enumerate(sorted(snaps.items())):
        axes[r, 0].imshow(s, vmin=-1, vmax=1, cmap="bwr")
        axes[r, 0].set_title(f"s (sign) t={t}")
        axes[r, 1].imshow(phi, vmin=-1, vmax=1, cmap="PiYG")
        axes[r, 1].set_title(f"phi (phase) t={t}")
        axes[r, 2].imshow(g, vmin=-1, vmax=1, cmap="coolwarm")
        axes[r, 2].set_title(f"g (gate) t={t}")
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_stats(stats):
    t = np.arange(len(stats["chg_s"]))
    plt.figure(figsize=(10, 4))
    plt.plot(t, stats["chg_s"], label="chg s")
    plt.plot(t, stats["chg_g"], label="chg g")
    plt.plot(t, stats["act"], label="ACT")
    plt.plot(t, stats["hold"], label="HOLD")
    plt.plot(t, stats["ban"], label="BAN")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("fraction")
    plt.title("Dynamics over time")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for k in ["m4", "m7", "m9", "fatigue"]:
        plt.plot(t, stats[k], label=k)
    plt.legend()
    plt.xlabel("step")
    plt.title("Motif triggers / fatigue")
    plt.tight_layout()
    plt.show()


def plot_multiscale(history):
    ks = [1, 2, 4, 8]
    rates = [coarse_change_rate(history, k) for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, rates, marker="o")
    plt.xlabel("coarsening factor k (larger = earlier digits)")
    plt.ylabel("avg change rate at scale")
    plt.title("Change rate vs refinement depth")
    plt.tight_layout()
    plt.show()
    print("Change rates by scale:", dict(zip(ks, rates)))


def load_price_series(csv_path=None):
    if csv_path is None:
        raise FileNotFoundError("CSV path required for CA tape.")
    path = pathlib.Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    price, volume = load_prices(path)
    return price, volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/raw/stooq/btc_intraday_1s.csv", help="CSV with close column.")
    ap.add_argument("--width", type=int, default=128, help="Lag width (X-axis).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = ap.parse_args()

    price, volume = load_price_series(args.csv)
    feats = build_features(price, volume)
    snaps, stats, hist = run_ca_tape(feats, width=args.width, seed=args.seed)
    plot_snapshots(snaps)
    plot_stats(stats)
    plot_multiscale(hist)


if __name__ == "__main__":
    main()

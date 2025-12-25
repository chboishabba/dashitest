# Let's generate CA variants, compute simple "rate of change" metrics,
# and produce plots vs time, fatigue parameter, and a mock "refinement depth".

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Reuse a simplified CA with tunable parameters ---

def neighbor_counts(x):
    shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    p = np.zeros_like(x, dtype=np.int16)
    n = np.zeros_like(x, dtype=np.int16)
    for dy,dx in shifts:
        y = np.roll(np.roll(x, dy, axis=0), dx, axis=1)
        p += (y == 1)
        n += (y == -1)
    return p, n

def step_ca(G, F, A, u, fatigue_thr=6, anchor_gain=0.6):
    pF, nF = neighbor_counts(F)
    pA, nA = neighbor_counts(A)
    flow = pF - nF
    anchor = pA - nA

    # Gate update
    G2 = G.copy()
    # M9: ban when anchor bad + conflict
    conflict = np.minimum(pF, nF)
    ban = (anchor < -2) & (conflict >= 3)
    G2[ban] = -1

    # M7: fatigue rim
    rim = (G == 1) & (u >= fatigue_thr) & (anchor < 2)
    G2[rim] = 0

    # M4: anchor corridor
    G2[anchor >= 3] = 1

    # Fatigue update
    u2 = u.copy()
    engaged = (G2 == 1) & (F != 0)
    u2[engaged] = np.minimum(u2[engaged] + 1, 15)
    u2[~engaged] = np.maximum(u2[~engaged] - 1, 0)

    # Flow update
    F2 = F.copy()
    F2[G2 == -1] = 0
    score = flow + anchor_gain * anchor
    F2[(G2 == 1) & (score > 2)] = 1
    F2[(G2 == 1) & (score < -2)] = -1
    F2[(G2 != 1)] = 0

    return G2, F2, u2

def run_variant(steps=200, fatigue_thr=6, anchor_gain=0.6, seed=0):
    rng = np.random.default_rng(seed)
    H=W=64
    A = rng.choice([-1,0,1], size=(H,W), p=[0.25,0.5,0.25])
    G = np.where(A>=0,1,0)
    F = rng.choice([-1,0,1], size=(H,W), p=[0.3,0.4,0.3])
    u = np.zeros((H,W),dtype=int)

    stats = defaultdict(list)
    for t in range(steps):
        G2,F2,u2 = step_ca(G,F,A,u,fatigue_thr,anchor_gain)
        # rate of change
        stats["F_change"].append(np.mean(F2!=F))
        stats["G_change"].append(np.mean(G2!=G))
        stats["act_frac"].append(np.mean(G2==1))
        stats["fatigue_mean"].append(u2.mean())
        G,F,u = G2,F2,u2
    return stats

# --- Sweep fatigue threshold to induce stability vs chaos ---
fatigue_vals = [2,4,6,8,10]
all_stats = {}

for fthr in fatigue_vals:
    all_stats[fthr] = run_variant(fatigue_thr=fthr, anchor_gain=0.6, seed=1)

# --- Plot rate of change vs time for different fatigue ---
plt.figure(figsize=(9,4))
for fthr, st in all_stats.items():
    plt.plot(st["F_change"], label=f"fatigue_thr={fthr}")
plt.xlabel("time step")
plt.ylabel("Flow change rate")
plt.title("CA dynamics: rate of change vs time")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot average rate vs fatigue (pathological sweep) ---
avg_rates = [np.mean(all_stats[f]["F_change"]) for f in fatigue_vals]

plt.figure(figsize=(6,4))
plt.plot(fatigue_vals, avg_rates, marker="o")
plt.xlabel("fatigue threshold")
plt.ylabel("avg flow change rate")
plt.title("Stability ↔ chaos via fatigue parameter")
plt.tight_layout()
plt.show()

# --- Mock refinement depth effect ---
# We approximate "refinement depth" by downsampling the grid (coarse-to-fine)
def coarse_rate(F, k):
    # k=1 fine, k=2 coarse blocks, k=4 more coarse
    H,W = F.shape
    F2 = F.reshape(H//k,k,W//k,k).mean(axis=(1,3))
    return np.mean(np.abs(np.diff(F2,axis=0)))

depths = [1,2,4,8]
rates_by_depth = []

# use one representative run
rep = run_variant(fatigue_thr=4, anchor_gain=0.6, seed=2)
# regenerate F history for depth analysis
rng = np.random.default_rng(2)
H=W=64
A = rng.choice([-1,0,1], size=(H,W), p=[0.25,0.5,0.25])
G = np.where(A>=0,1,0)
F = rng.choice([-1,0,1], size=(H,W), p=[0.3,0.4,0.3])
u = np.zeros((H,W),dtype=int)

F_hist = []
for t in range(120):
    F_hist.append(F.copy())
    G,F,u = step_ca(G,F,A,u,4,0.6)

for k in depths:
    diffs = []
    for t in range(1,len(F_hist)):
        diffs.append(coarse_rate(F_hist[t],k))
    rates_by_depth.append(np.mean(diffs))

plt.figure(figsize=(6,4))
plt.plot(depths, rates_by_depth, marker="o")
plt.xlabel("coarsening factor (refinement depth)")
plt.ylabel("avg coarse change rate")
plt.title("Rate vs refinement depth (coarse→fine)")
plt.tight_layout()
plt.show()

rates_by_depth

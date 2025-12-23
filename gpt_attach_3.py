import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict

def neighbor_counts_ternary(x):
    # x in {0,1,2}; interpret 1 as +, 2 as -
    # counts in 3x3 including self
    shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    p = np.zeros_like(x, dtype=np.int16)
    n = np.zeros_like(x, dtype=np.int16)
    z = np.zeros_like(x, dtype=np.int16)
    for dy,dx in shifts:
        y = np.roll(np.roll(x, dy, axis=0), dx, axis=1)
        p += (y==1)
        n += (y==2)
        z += (y==0)
    return p,n,z

def step_two_layer(G,F,A,u, alpha=0.6, th=2.0, u_thr=6):
    # G in {-1,0,+1} stored as int8 with values {0,1,2} mapped: -1->2,0->0,+1->1? Let's store directly -1,0,+1.
    # F in {0,1,2} (0 neutral,1 +,2 -). A in {-1,0,+1}.
    pA,nA,_ = neighbor_counts_ternary((A==1).astype(np.int8) + 2*(A==-1).astype(np.int8))  # encode A into {0,1,2}
    # Convert counts for A: in encoded, 1 means +, 2 means -
    # But neighbor_counts_ternary expects 0/1/2; already ok
    # For pA/nA above, pA counts encoded==1, nA counts encoded==2

    pF,nF,_ = neighbor_counts_ternary(F)

    anchor_score = (pA - nA).astype(np.int16)  # range [-9,9]
    flow_score = (pF - nF).astype(np.int16)    # range [-9,9]

    # Conflict proxy: lots of both signs around you
    conflict = np.minimum(pF, nF)  # 0..9

    # --- Gate update (G) ---
    G_next = G.copy()
    M4 = np.zeros_like(G, dtype=bool)
    M7 = np.zeros_like(G, dtype=bool)
    M9 = np.zeros_like(G, dtype=bool)

    # M9: circuit breaker when anchor is negative AND conflict high AND gate currently permissive
    M9 = (anchor_score <= -2) & (conflict >= 3) & (G == 1)
    G_next[M9] = -1

    # M7: fatigue rim: if fatigued and anchor not strongly supportive, drop to HOLD(0)
    M7 = (G == 1) & (u >= u_thr) & (anchor_score < 2) & (~M9)
    G_next[M7] = 0

    # M4: anchored corridor: if anchor strongly positive, keep permissive even with moderate conflict
    M4 = (anchor_score >= 3) & (G != -1)
    G_next[M4] = 1

    # Recovery: if prohibited but anchor becomes strongly positive and conflict low, reopen
    recover = (G == -1) & (anchor_score >= 4) & (conflict <= 1)
    G_next[recover] = 1

    # --- Fatigue update ---
    u_next = u.copy()
    engaged = (G_next == 1) & (F != 0)
    u_next[engaged] = np.minimum(u_next[engaged] + 1, 15)
    u_next[~engaged] = np.maximum(u_next[~engaged] - 1, 0)

    # --- Flow update (F) ---
    F_next = F.copy()

    # Where prohibited => zeroed
    F_next[G_next == -1] = 0

    # Where HOLD => decay toward 0
    hold = (G_next == 0)
    if hold.any():
        # deterministic decay: if nonzero, halve probability-ish via rule: toggle to 0 every other step not possible without time;
        # we do: nonzero -> 0 with 0.5 probability using a hash-like mask from neighbor score
        mask = ((flow_score + anchor_score) & 1) == 0
        F_next[hold & (F!=0) & mask] = 0

    # Where ACT => update by biased majority + anchor
    act = (G_next == 1)
    if act.any():
        score = flow_score + (alpha * anchor_score)
        score = score.astype(np.float32)
        # triadic decision
        plus = score > th
        minus = score < -th
        F_next[act & plus] = 1
        F_next[act & minus] = 2
        F_next[act & ~(plus | minus)] = 0

    return G_next, F_next, A, u_next, M4, M7, M9

def run_demo(H=128,W=128,steps=160,seed=0):
    rng = np.random.default_rng(seed)
    # Anchor field: start random then smooth by majority to create blobs
    A = rng.choice([-1,0,1], size=(H,W), p=[0.25,0.5,0.25]).astype(np.int8)
    # Smooth 6 iterations
    for _ in range(6):
        # encode A into 0/1/2 for counting
        encA = (A==1).astype(np.int8) + 2*(A==-1).astype(np.int8)
        pA,nA,zA = neighbor_counts_ternary(encA)
        # majority with tie -> 0
        A2 = np.zeros_like(A)
        A2[pA > nA] = 1
        A2[nA > pA] = -1
        # keep 0 where tied or both small
        A = A2.astype(np.int8)
    # Init gate and flow
    G = np.where(A>=0, 1, 0).astype(np.int8)   # permissive in non-negative anchor, hold in negative anchor
    F = rng.integers(0,3,size=(H,W),dtype=np.int8)
    u = np.zeros((H,W), dtype=np.int8)

    stats = defaultdict(list)
    snaps = {}
    snap_steps = [0, 20, 60, 120, steps-1]

    for t in range(steps):
        if t in snap_steps:
            snaps[t] = (G.copy(), F.copy(), A.copy(), u.copy())
        G,F,A,u,M4,M7,M9 = step_two_layer(G,F,A,u)
        stats["act_frac"].append((G==1).mean())
        stats["hold_frac"].append((G==0).mean())
        stats["ban_frac"].append((G==-1).mean())
        stats["F_plus"].append((F==1).mean())
        stats["F_zero"].append((F==0).mean())
        stats["F_minus"].append((F==2).mean())
        stats["M4_trig"].append(M4.mean())
        stats["M7_trig"].append(M7.mean())
        stats["M9_trig"].append(M9.mean())
        stats["fatigue_mean"].append(u.mean())
    return snaps, stats

snaps, stats = run_demo()

# --- Plot snapshots ---
cmapA = ListedColormap(["#000000","#888888","#ffffff"])  # -1,0,1 mapped later
cmapF = ListedColormap(["#222222","#bbbbbb","#666666"])  # 0,1,2 (neutral,+,-) just grayscale-ish
cmapG = ListedColormap(["#000000","#888888","#ffffff"])  # -1,0,1

def show_grid(ax, X, kind):
    if kind=="A":
        img = (X+1)  # -1,0,1 -> 0,1,2
        ax.imshow(img, vmin=0, vmax=2)
        ax.set_title("Anchor A (-1/0/+1)")
    elif kind=="G":
        img = (X+1)
        ax.imshow(img, vmin=0, vmax=2)
        ax.set_title("Gate G (ban/hold/act)")
    elif kind=="F":
        ax.imshow(X, vmin=0, vmax=2)
        ax.set_title("Flow F (0/+/-)")
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(len(snaps), 3, figsize=(9, 2.2*len(snaps)))
for r,(t,(G,F,A,u)) in enumerate(sorted(snaps.items())):
    show_grid(axes[r,0], A, "A")
    show_grid(axes[r,1], G, "G")
    show_grid(axes[r,2], F, "F")
    axes[r,0].set_ylabel(f"t={t}", rotation=0, labelpad=25, va="center")
plt.tight_layout()
plt.show()

# --- Plot stats ---
ts = np.arange(len(stats["act_frac"]))
fig2, ax = plt.subplots(figsize=(9,4))
ax.plot(ts, stats["act_frac"], label="G=ACT")
ax.plot(ts, stats["hold_frac"], label="G=HOLD")
ax.plot(ts, stats["ban_frac"], label="G=BAN")
ax.plot(ts, stats["M4_trig"], label="M4 triggers")
ax.plot(ts, stats["M7_trig"], label="M7 triggers")
ax.plot(ts, stats["M9_trig"], label="M9 triggers")
ax.plot(ts, stats["fatigue_mean"], label="fatigue mean")
ax.set_xlabel("step")
ax.set_ylabel("fraction / mean")
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# quick numeric summary
summary = {
    "final_G_act": stats["act_frac"][-1],
    "final_G_hold": stats["hold_frac"][-1],
    "final_G_ban": stats["ban_frac"][-1],
    "avg_M4_trig": float(np.mean(stats["M4_trig"])),
    "avg_M7_trig": float(np.mean(stats["M7_trig"])),
    "avg_M9_trig": float(np.mean(stats["M9_trig"])),
    "avg_fatigue": float(np.mean(stats["fatigue_mean"])),
}
summary

import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math, os

# --- Utilities ---
def moore_counts3(S):
    """Counts of each state 0/1/2 in 3x3 Moore neighborhood incl self."""
    shifts=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    c0=np.zeros_like(S,dtype=np.int16)
    c1=np.zeros_like(S,dtype=np.int16)
    c2=np.zeros_like(S,dtype=np.int16)
    for dy,dx in shifts:
        X=np.roll(np.roll(S,dy,0),dx,1)
        c0 += (X==0)
        c1 += (X==1)
        c2 += (X==2)
    return c0,c1,c2

def moore_counts_pm(X):
    """X in {-1,0,+1}; counts of + and - in 3x3 incl self."""
    shifts=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    p=np.zeros_like(X,dtype=np.int16)
    n=np.zeros_like(X,dtype=np.int16)
    for dy,dx in shifts:
        Y=np.roll(np.roll(X,dy,0),dx,1)
        p += (Y==1)
        n += (Y==-1)
    return p,n

# --- A visibly-moving triadic CA with competing anchors + fatigue -> negative action ---
def run_moving_ca(H=160,W=240,steps=240, seed=0,
                  drive_shift=(0,1),
                  fatigue_thr=10, fatigue_gain=1,
                  conflict_thr=3, ban_thr=6,
                  noise_p=0.002):
    """
    Layers:
      A1,A2 in {-1,0,+1} = competing anchors (slowly drifting)
      G in {-1,0,+1} = permission gate
      F in {0,1,2} = cyclic 'excitable' flow to generate waves/spirals (0->1->2->0)
      u in [0..15] = fatigue
    Motif interpretations (local triggers):
      M4: anchored corridor (strong net anchor) => keep G=+1 (ACT)
      M7: fatigue overflow => G tilts negative (SELL / adverse action) rather than just HOLD
      M9: ban/shutdown islands when anchor-conflict is high and local turbulence high
    """
    rng=np.random.default_rng(seed)

    # Competing anchors as smooth blobs from random then majority smooth
    def smooth_anchor(p_pos=0.18,p_zero=0.64,p_neg=0.18, iters=5):
        A=rng.choice([-1,0,1], size=(H,W), p=[p_neg,p_zero,p_pos]).astype(np.int8)
        for _ in range(iters):
            p,n=moore_counts_pm(A)
            A2=np.zeros_like(A)
            A2[p>n]=1
            A2[n>p]=-1
            # keep 0 on ties
            A=A2.astype(np.int8)
        return A

    A1=smooth_anchor(iters=6)
    A2=smooth_anchor(iters=6)
    # Start flow in excitable model: mostly 0 with sparse 1 seeds
    F=rng.choice([0,1,2], size=(H,W), p=[0.92,0.06,0.02]).astype(np.int8)
    u=np.zeros((H,W),dtype=np.int8)
    # initial gate: act if net anchor non-negative
    net=(A1 - A2).astype(np.int8)  # competition
    G=np.where(net>=0, 1, 0).astype(np.int8)

    # stats
    stats={k:[] for k in ["chgF","chgG","act","hold","ban","fatigue","M4","M7","M9","neg_action"]}
    snaps={}
    snap_ts=[0,30,80,140,steps-1]

    for t in range(steps):
        if t in snap_ts:
            snaps[t]=(A1.copy(),A2.copy(),G.copy(),F.copy(),u.copy())

        # --- drive: drift anchors (competing fields moving like weather systems) ---
        dy,dx=drive_shift
        A1=np.roll(np.roll(A1,dy,0),dx,1)
        A2=np.roll(np.roll(A2,-dy,0),-dx,1)

        # local net anchor + conflict
        p1,n1=moore_counts_pm(A1)
        p2,n2=moore_counts_pm(A2)
        net_score=(p1-n1) - (p2-n2)                # support difference
        conflict=np.minimum(p1+n2, n1+p2)          # "both sides present" proxy

        # local turbulence proxy from F
        c0,c1,c2=moore_counts3(F)
        # turbulence high when multiple states co-exist
        turb=np.minimum(c1+c2, c0+c1) + np.minimum(c0+c2, c0+c1)

        # --- Gate update with motifs ---
        G2=G.copy()
        M9 = (conflict>=ban_thr) & (turb>=10)      # shutdown islands
        G2[M9]=-1

        M4 = (net_score>=4) & (~M9)                # corridor keeps acting
        G2[M4]=1

        # fatigue builds when acting in turbulent zones
        engaged=(G==1) & (F!=0)
        u2=u.copy()
        u2[engaged]=np.minimum(u2[engaged]+fatigue_gain, 15)
        u2[~engaged]=np.maximum(u2[~engaged]-1, 0)

        # M7: excessive fatigue causes negative action (tilt to -1 rather than hold)
        M7 = (u2>=fatigue_thr) & (G2==1) & (net_score<2) & (~M9)
        G2[M7]=-1

        # Hold zone: when anchor is ambiguous and not banned
        hold = (~M9) & (G2!=-1) & (np.abs(net_score)<2)
        G2[hold]=0

        # random flips (pathology/noise)
        if noise_p>0:
            flip = rng.random((H,W)) < noise_p
            # flip among {-1,0,1}
            choices = rng.choice([-1,0,1], size=flip.sum())
            G2[flip]=choices.astype(np.int8)

        # --- Flow update (excitable cyclic waves) ---
        F2=F.copy()
        # if banned => quench
        F2[G2==-1]=0

        # act: excitable rule with anchor bias & tie-break produces moving patterns
        act=(G2==1)
        if act.any():
            # birth/excitation when enough neighbors in state 1, modulated by net_score
            excite_thresh = 2 + (net_score<0)  # harder to excite when anchor against you
            can_excite = (F==0) & (c1>=excite_thresh)
            F2[act & can_excite]=1
            # cyclic advance
            F2[act & (F==1)] = 2
            F2[act & (F==2)] = 0
            # tie-breaking "glider-ish": if neighbors balanced, advance anyway sometimes
            balanced = (c1==c2) & (F==0) & (c1>=2)
            F2[act & balanced] = 1

        # hold: slow decay
        hold=(G2==0)
        if hold.any():
            # decay excited/refractory toward rest
            F2[hold & (F==1)] = 2
            F2[hold & (F==2)] = 0

        # track stats
        stats["chgF"].append(np.mean(F2!=F))
        stats["chgG"].append(np.mean(G2!=G))
        stats["act"].append(np.mean(G2==1))
        stats["hold"].append(np.mean(G2==0))
        stats["ban"].append(np.mean(G2==-1))
        stats["fatigue"].append(float(u2.mean()))
        stats["M4"].append(np.mean(M4))
        stats["M7"].append(np.mean(M7))
        stats["M9"].append(np.mean(M9))
        stats["neg_action"].append(np.mean(G2==-1))

        G,F,u = G2,F2,u2

    return snaps, stats

snaps, stats = run_moving_ca()

# --- Plot snapshots (A net, G, F, fatigue) ---
cmap_tern = ListedColormap(["black","gray","white"])   # -1,0,+1 mapped to 0,1,2
cmap_F = ListedColormap(["black","white","gray"])      # 0,1,2

def imshow_pm(ax, X, title):
    ax.imshow((X+1), vmin=0, vmax=2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

def imshow_F(ax, X, title):
    ax.imshow(X, vmin=0, vmax=2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

def imshow_u(ax, U, title):
    ax.imshow(U, vmin=0, vmax=15)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(len(snaps), 4, figsize=(12, 2.2*len(snaps)))
for r,(t,(A1,A2,G,F,u)) in enumerate(sorted(snaps.items())):
    net = (A1 - A2).clip(-1,1).astype(np.int8)
    imshow_pm(axes[r,0], net, "Net anchor (A1-A2)")
    imshow_pm(axes[r,1], G, "Gate G (ban/hold/act)")
    imshow_F(axes[r,2], F, "Flow F (0/1/2 waves)")
    imshow_u(axes[r,3], u, "Fatigue u (0..15)")
    axes[r,0].set_ylabel(f"t={t}", rotation=0, labelpad=25, va="center")
plt.tight_layout()
plt.show()

# --- Plot time-series stats ---
t = np.arange(len(stats["chgF"]))
plt.figure(figsize=(10,4))
plt.plot(t, stats["chgF"], label="Flow change rate")
plt.plot(t, stats["chgG"], label="Gate change rate")
plt.plot(t, stats["act"], label="ACT fraction")
plt.plot(t, stats["hold"], label="HOLD fraction")
plt.plot(t, stats["ban"], label="BAN fraction")
plt.xlabel("step")
plt.ylabel("fraction")
plt.title("Moving CA: dynamics over time")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, stats["M4"], label="M4 corridor triggers")
plt.plot(t, stats["M7"], label="M7 fatigue→negative triggers")
plt.plot(t, stats["M9"], label="M9 shutdown triggers")
plt.plot(t, stats["fatigue"], label="mean fatigue")
plt.xlabel("step")
plt.ylabel("fraction / mean")
plt.title("Motif triggers + fatigue")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# --- "Rate vs refinement depth" proxy via coarse-graining F ---
# We'll compute change rate at different coarsening factors k (bigger k = coarser, earlier digits)
def coarse_F(F, k):
    H,W=F.shape
    H2=(H//k)*k; W2=(W//k)*k
    X=F[:H2,:W2]
    # majority of 0/1/2 inside each block
    Xb=X.reshape(H2//k,k,W2//k,k)
    # count states in block
    c0=(Xb==0).sum(axis=(1,3))
    c1=(Xb==1).sum(axis=(1,3))
    c2=(Xb==2).sum(axis=(1,3))
    out=np.zeros((H2//k, W2//k), dtype=np.int8)
    out[(c1>c0) & (c1>=c2)] = 1
    out[(c2>c0) & (c2>c1)] = 2
    # else 0
    return out

# recreate a short history to compute depth curves
def get_F_history(params, T=120):
    snaps, st = run_moving_ca(steps=T, seed=1, **params)
    # run_moving_ca doesn't return full history, so rerun with capture:
    rng=np.random.default_rng(1)
    H=160; W=240
    # (repeat initialization same as run_moving_ca)
    def smooth_anchor_local(rng, p_pos=0.18,p_zero=0.64,p_neg=0.18, iters=6):
        A=rng.choice([-1,0,1], size=(H,W), p=[p_neg,p_zero,p_pos]).astype(np.int8)
        for _ in range(iters):
            p,n=moore_counts_pm(A)
            A2=np.zeros_like(A)
            A2[p>n]=1
            A2[n>p]=-1
            A=A2.astype(np.int8)
        return A
    A1=smooth_anchor_local(rng)
    A2=smooth_anchor_local(rng)
    F=rng.choice([0,1,2], size=(H,W), p=[0.92,0.06,0.02]).astype(np.int8)
    u=np.zeros((H,W),dtype=np.int8)
    net=(A1-A2).astype(np.int8)
    G=np.where(net>=0,1,0).astype(np.int8)
    hist=[]
    for tt in range(T):
        hist.append(F.copy())
        # step using same parameters (call the core loop logic by invoking run_moving_ca's dynamics inline is bulky; use a shorter variant)
        # We'll re-use run_moving_ca step by rerunning a 1-step run isn't efficient. Instead keep it simple: call run_moving_ca anew isn't possible.
        # For this plot, use a smaller grid variant that's consistent enough: use a simpler helper.
        break
    return None

# We'll compute depth curve on the captured snapshot series we already have by approximating from two snapshots isn't enough.
# Instead, generate a new run and store F history explicitly here.
def run_moving_ca_with_history(**kwargs):
    H=kwargs.pop("H",120); W=kwargs.pop("W",160)
    steps=kwargs.pop("steps",160); seed=kwargs.pop("seed",0)
    drive_shift=kwargs.pop("drive_shift",(0,1))
    fatigue_thr=kwargs.pop("fatigue_thr",10)
    fatigue_gain=kwargs.pop("fatigue_gain",1)
    ban_thr=kwargs.pop("ban_thr",6)
    noise_p=kwargs.pop("noise_p",0.002)
    anchor_gain=kwargs.pop("anchor_gain",0.6)
    rng=np.random.default_rng(seed)
    def smooth_anchor(p_pos=0.18,p_zero=0.64,p_neg=0.18, iters=5):
        A=rng.choice([-1,0,1], size=(H,W), p=[p_neg,p_zero,p_pos]).astype(np.int8)
        for _ in range(iters):
            p,n=moore_counts_pm(A)
            A2=np.zeros_like(A)
            A2[p>n]=1
            A2[n>p]=-1
            A=A2.astype(np.int8)
        return A
    A1=smooth_anchor(iters=6)
    A2=smooth_anchor(iters=6)
    F=rng.choice([0,1,2], size=(H,W), p=[0.92,0.06,0.02]).astype(np.int8)
    u=np.zeros((H,W),dtype=np.int8)
    net=(A1-A2).astype(np.int8)
    G=np.where(net>=0,1,0).astype(np.int8)
    F_hist=[]
    for t in range(steps):
        F_hist.append(F.copy())
        dy,dx=drive_shift
        A1=np.roll(np.roll(A1,dy,0),dx,1)
        A2=np.roll(np.roll(A2,-dy,0),-dx,1)
        p1,n1=moore_counts_pm(A1); p2,n2=moore_counts_pm(A2)
        net_score=(p1-n1)-(p2-n2)
        conflict=np.minimum(p1+n2, n1+p2)
        c0,c1,c2=moore_counts3(F)
        turb=np.minimum(c1+c2, c0+c1) + np.minimum(c0+c2, c0+c1)
        G2=G.copy()
        M9=(conflict>=ban_thr) & (turb>=10)
        G2[M9]=-1
        M4=(net_score>=4) & (~M9)
        G2[M4]=1
        engaged=(G==1) & (F!=0)
        u2=u.copy()
        u2[engaged]=np.minimum(u2[engaged]+fatigue_gain, 15)
        u2[~engaged]=np.maximum(u2[~engaged]-1, 0)
        M7=(u2>=fatigue_thr) & (G2==1) & (net_score<2) & (~M9)
        G2[M7]=-1
        hold=(~M9) & (G2!=-1) & (np.abs(net_score)<2)
        G2[hold]=0
        if noise_p>0:
            flip = rng.random((H,W)) < noise_p
            G2[flip]=rng.choice([-1,0,1], size=flip.sum()).astype(np.int8)
        F2=F.copy()
        F2[G2==-1]=0
        act=(G2==1)
        if act.any():
            excite_thresh = 2 + (net_score<0)
            can_excite = (F==0) & (c1>=excite_thresh)
            F2[act & can_excite]=1
            F2[act & (F==1)]=2
            F2[act & (F==2)]=0
            balanced=(c1==c2) & (F==0) & (c1>=2)
            F2[act & balanced]=1
        hold=(G2==0)
        if hold.any():
            F2[hold & (F==1)]=2
            F2[hold & (F==2)]=0
        G,F,u = G2,F2,u2
    return F_hist

F_hist = run_moving_ca_with_history(H=96,W=128,steps=160,seed=3, drive_shift=(0,1),
                                   fatigue_thr=8, ban_thr=5, noise_p=0.0015)

coarsen = [1,2,4,8]
depth_rates=[]
for k in coarsen:
    diffs=[]
    prev=coarse_F(F_hist[0],k)
    for tt in range(1,len(F_hist)):
        cur=coarse_F(F_hist[tt],k)
        diffs.append(np.mean(cur!=prev))
        prev=cur
    depth_rates.append(float(np.mean(diffs)))

plt.figure(figsize=(6,4))
plt.plot(coarsen, depth_rates, marker="o")
plt.xlabel("coarsening factor k (larger = coarser / earlier digits)")
plt.ylabel("avg change rate at that scale")
plt.title("Rate vs refinement depth (multiscale p-adic proxy)")
plt.tight_layout()
plt.show()

depth_rates

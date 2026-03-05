# Improved trading PCA visualisation
# - Real uploaded log
# - Cumulative manifold with decision overlay
# - Regime density evolution
# Matplotlib only, no explicit colors, single plot per animation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

log_path = "logs/trading_log_btc_intraday_1s.csv"
df = pd.read_csv(log_path)

# ---- Select meaningful numeric features ----
candidate_cols = [
    "ell","actionability","acceptable",
    "intent_direction","intent_target","urgency",
    "hold","fill","exposure",
    "slippage","fee","pnl","state"
]

cols = [c for c in candidate_cols if c in df.columns]

if len(cols) < 3:
    # fallback to numeric
    cols = df.select_dtypes(include=[np.number]).columns.tolist()

X = df[cols].copy()
X = X.replace([np.inf,-np.inf], np.nan).fillna(method="ffill").fillna(0.0)

# standardize
A = X.to_numpy(dtype=np.float32)
A = (A - A.mean(axis=0)) / (A.std(axis=0)+1e-9)

# PCA
pca = PCA(n_components=3)
coords = pca.fit_transform(A)
coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0)+1e-9)

# Cosine similarity (for regime recurrence)
An = A / (np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
sim = An @ An.T

# Detect trade vs hold if column exists
if "fill" in df.columns:
    trade_flag = (df["fill"].fillna(0) != 0).astype(int).to_numpy()
else:
    trade_flag = np.zeros(len(df))

n = len(coords)
frames = min(200, n)
fps = 20

# -------- Animation 1: Cumulative manifold with trade emphasis --------
fig1 = plt.figure(figsize=(7,7))
ax1 = fig1.add_subplot(111, projection="3d")

def update1(i):
    ax1.cla()
    ax1.set_title(f"Cumulative Trading Manifold (t={i+1})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")

    # All past states faint
    ax1.scatter(coords[:i+1,0], coords[:i+1,1], coords[:i+1,2], s=8, alpha=0.15)

    # Highlight current state
    ax1.scatter(coords[i,0], coords[i,1], coords[i,2], s=70)

    # If trade, draw recurrence links
    if trade_flag[i] == 1 and i > 5:
        s = sim[i,:i]
        idx = np.argsort(s)[-5:]
        for j in idx:
            ax1.plot([coords[i,0], coords[j,0]],
                     [coords[i,1], coords[j,1]],
                     [coords[i,2], coords[j,2]])
    return []

ani1 = FuncAnimation(fig1, update1, frames=frames, interval=1000/fps)
out1 = "trading_cumulative_improved.gif"
ani1.save(out1, writer=PillowWriter(fps=fps))
plt.close(fig1)


# -------- Animation 2: Regime density over time --------
# Uses sliding window PCA density projection
fig2 = plt.figure(figsize=(7,7))
ax2 = fig2.add_subplot(111)

def update2(i):
    ax2.cla()
    ax2.set_title(f"Regime Density Evolution (t={i+1})")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    pts = coords[max(0,i-200):i+1]
    ax2.scatter(pts[:,0], pts[:,1], s=10, alpha=0.25)

    ax2.scatter(coords[i,0], coords[i,1], s=80)
    return []

ani2 = FuncAnimation(fig2, update2, frames=frames, interval=1000/fps)
out2 = "trading_regime_density.gif"
ani2.save(out2, writer=PillowWriter(fps=fps))
plt.close(fig2)

out1, out2

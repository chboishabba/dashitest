# Decision Geometry Visualisation (retry)
# - Real BTC 1s trading log
# - Learn decision boundary (trade vs hold)
# - Animate decision surface in PCA space
# - Single plot, no explicit colors, no orbit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

log_path = "logs/trading_log_btc_intraday_1s.csv"
df = pd.read_csv(log_path)

# ---- Feature selection ----
candidate_cols = [
    "ell","actionability","acceptable",
    "intent_direction","intent_target","urgency",
    "hold","exposure","state"
]

cols = [c for c in candidate_cols if c in df.columns]
if len(cols) < 3:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()

X = df[cols].replace([np.inf,-np.inf], np.nan).ffill().fillna(0.0)
A = X.to_numpy(dtype=np.float32)
A = (A - A.mean(axis=0)) / (A.std(axis=0)+1e-9)

# ---- Trade flag ----
if "fill" in df.columns:
    y = (df["fill"].fillna(0) != 0).astype(int).to_numpy()
else:
    y = np.zeros(len(df))

# ---- PCA to 2D ----
pca = PCA(n_components=2)
coords = pca.fit_transform(A)
coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0)+1e-9)

# ---- Train logistic boundary ----
clf = LogisticRegression(max_iter=1000)
clf.fit(coords, y)

xmin, xmax = coords[:,0].min(), coords[:,0].max()
ymin, ymax = coords[:,1].min(), coords[:,1].max()
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 120),
                     np.linspace(ymin, ymax, 120))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:,1].reshape(xx.shape)

frames = min(200, len(coords))
fps = 20

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

def update(i):
    ax.cla()
    ax.set_title(f"Decision Geometry (t={i+1})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.contour(xx, yy, probs, levels=[0.5])

    ax.scatter(coords[:i+1,0], coords[:i+1,1], s=10, alpha=0.2)
    ax.scatter(coords[i,0], coords[i,1], s=80)

    return []

ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps)
out_path = "trading_decision_geometry.gif"
ani.save(out_path, writer=PillowWriter(fps=fps))
plt.close(fig)

out_path

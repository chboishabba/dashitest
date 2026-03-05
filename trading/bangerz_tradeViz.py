import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

def pick_numeric_cols(df: pd.DataFrame):
    # Prefer decision-related columns if present; fall back to all numeric.
    preferred = [
        "state", "acceptable",
        "ell", "actionability",
        "intent_direction", "intent_target", "urgency",
        "hold", "fill", "exposure",
        "slippage", "fee", "pnl",
        "price", "volume",
        "p_bad", "bad_flag", "stress"
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return cols

def to_feature_matrix(df: pd.DataFrame, cols, smooth=0):
    X = df[cols].copy()

    # Clean NaNs/infs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(method="ffill").fillna(0.0)

    # Optional smoothing (EMA-ish)
    if smooth > 0:
        X = X.ewm(span=smooth, adjust=False).mean()

    # Standardize
    A = X.to_numpy(dtype=np.float32)
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True) + 1e-9
    A = (A - mu) / sd
    return A

def cosine_similarity_matrix(Xn):
    # Xn should already be normalized row-wise
    return Xn @ Xn.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to logs/trading_log.csv")
    ap.add_argument("--max-frames", type=int, default=200, help="Cap frames for speed")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--smooth", type=int, default=0, help="EMA span over features (0=off)")
    ap.add_argument("--k", type=int, default=3, help="Top-k repetition links")
    ap.add_argument("--thr", type=float, default=0.95, help="Repeat threshold for motif count")
    ap.add_argument("--out-prefix", default="logs/pca_trade")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    cols = pick_numeric_cols(df)
    if len(cols) < 3:
        raise SystemExit(f"Not enough numeric columns to PCA. Found: {cols}")

    X = to_feature_matrix(df, cols, smooth=args.smooth)

    # PCA -> 3D coordinates
    pca = PCA(n_components=3)
    coords = pca.fit_transform(X).astype(np.float32)

    # Normalize coords for stable framing
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)

    # Similarities in feature space (not PCA space)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = cosine_similarity_matrix(Xn).astype(np.float32)

    n = coords.shape[0]
    frames = min(n, args.max_frames)

    # Precompute repeat edges + motif counts cumulatively
    edges = []
    motif_counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if i == 0:
            edges.append(np.array([], dtype=np.int32))
            motif_counts[i] = 0
            continue
        s = sim[i, :i].copy()
        idx = np.where(s > args.thr)[0]
        motif_counts[i] = int(len(idx))
        edges.append(idx)

    # -------- GIF 1: cumulative manifold (single 3D plot, no orbit) --------
    fig1 = plt.figure(figsize=(7, 7))
    ax = fig1.add_subplot(111, projection="3d")

    def upd1(i):
        ax.cla()
        ax.set_title(f"Cumulative PCA manifold (t={i+1}/{n})")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

        ax.scatter(coords[:i+1,0], coords[:i+1,1], coords[:i+1,2], s=8, alpha=0.25)
        ax.scatter(coords[i,0], coords[i,1], coords[i,2], s=60)

        # Top-k most similar past windows (links)
        if i > 0:
            s = sim[i, :i].copy()
            j = np.argsort(s)[-args.k:]
            for jj in j:
                ax.plot([coords[i,0], coords[jj,0]],
                        [coords[i,1], coords[jj,1]],
                        [coords[i,2], coords[jj,2]])

        # Also show "repeat count" text
        ax.text2D(0.02, 0.02, f"repeat_count(thr={args.thr})={motif_counts[i]}", transform=ax.transAxes)
        return []

    ani1 = FuncAnimation(fig1, upd1, frames=frames, interval=1000/args.fps)
    out1 = f"{args.out_prefix}_cumulative.gif"
    ani1.save(out1, writer=PillowWriter(fps=args.fps))
    plt.close(fig1)

    # -------- GIF 2: motif repeat count over time (single 2D plot) --------
    fig2 = plt.figure(figsize=(8, 4.8))
    ax2 = fig2.add_subplot(111)

    def upd2(i):
        ax2.cla()
        ax2.set_title("Repeated motif count over time")
        ax2.set_xlabel("t")
        ax2.set_ylabel(f"#repeats (sim > {args.thr})")
        ax2.plot(np.arange(i+1), motif_counts[:i+1])
        return []

    ani2 = FuncAnimation(fig2, upd2, frames=frames, interval=1000/args.fps)
    out2 = f"{args.out_prefix}_motif_count.gif"
    ani2.save(out2, writer=PillowWriter(fps=args.fps))
    plt.close(fig2)

    print("Wrote:")
    print(" ", out1)
    print(" ", out2)
    print("\nFeatures used:")
    print(cols)

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_ridge_logit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    n = X.shape[0]
    Xb = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    xtx = Xb.T @ Xb
    reg = lam * np.eye(xtx.shape[0], dtype=X.dtype)
    w = np.linalg.solve(xtx + reg, Xb.T @ y)
    return w[:-1], w[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline ell_v1 training stub (ridge + sigmoid).")
    ap.add_argument("--log", type=Path, required=True, help="Trading log CSV with acceptable.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-2, help="Ridge regularization.")
    ap.add_argument("--out", type=Path, default=Path("logs/ell_v1_weights.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    if "acceptable" not in df:
        raise SystemExit("log must contain acceptable column")
    y = pd.to_numeric(df["acceptable"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = np.clip(y, 0.0, 1.0)

    T = int(len(df))
    mm = np.memmap(args.tape, dtype=np.float32, mode="r")
    if mm.size % 8 != 0:
        raise ValueError("tape size is not divisible by 8 floats")
    records = mm.size // 8
    if records % T != 0:
        raise ValueError("tape length does not align with log length")
    S = records // T
    if args.series < 0 or args.series >= S:
        raise ValueError(f"series index {args.series} out of range (S={S})")
    tape = np.memmap(args.tape, dtype=np.float32, mode="r", shape=(S, T, 8))
    qfeat = tape[args.series, :T, :6].astype(np.float32, copy=False)
    y = y[:T]

    mean = qfeat.mean(axis=0)
    std = qfeat.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    X = (qfeat - mean) / std

    w, b = fit_ridge_logit(X, y, args.lam)
    ell = sigmoid(X @ w + b)

    acc = float(np.mean((ell >= 0.5) == (y >= 0.5)))
    mean_pos = float(np.mean(ell[y >= 0.5])) if np.any(y >= 0.5) else float("nan")
    mean_neg = float(np.mean(ell[y < 0.5])) if np.any(y < 0.5) else float("nan")

    payload = {
        "weights": w.tolist(),
        "bias": float(b),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "lambda": float(args.lam),
        "metrics": {
            "rows": int(T),
            "acc_0.5": acc,
            "mean_ell_accept": mean_pos,
            "mean_ell_reject": mean_neg,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

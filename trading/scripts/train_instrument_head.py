"""
train_instrument_head.py
------------------------
Train a simple instrument-family classifier using qfeat + market meta features.

Pseudo-labels (heuristic, no PnL):
  OPTION if (opt_mark_iv_mean >= iv_high) and (burstiness >= burst_high)
  PERP if (abs(premium_funding_rate) >= funding_thresh) and (acorr_1 >= acorr_trend)
  SPOT otherwise

Training is gated by ell (e.g., ell >= tau_on).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from trading.trading_io.prices import load_prices
    from trading.vk_qfeat import QFeatTape
except ModuleNotFoundError:
    from trading_io.prices import load_prices
    from vk_qfeat import QFeatTape


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.clip(expz.sum(axis=1, keepdims=True), 1e-12, None)


def _train_softmax(
    X: np.ndarray,
    y_idx: np.ndarray,
    num_classes: int,
    lr: float,
    epochs: int,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    W = np.zeros((d, num_classes), dtype=np.float32)
    b = np.zeros(num_classes, dtype=np.float32)
    y_onehot = np.eye(num_classes, dtype=np.float32)[y_idx]
    for _ in range(max(1, epochs)):
        logits = X @ W + b
        probs = _softmax(logits)
        grad = (probs - y_onehot) / max(1, n)
        grad_W = X.T @ grad + lam * W
        grad_b = grad.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


def _align_meta(meta_path: Path, ts_int: np.ndarray) -> pd.DataFrame:
    meta_df = pd.read_csv(meta_path)
    if "ts" not in meta_df.columns:
        raise SystemExit("meta-features CSV must contain a ts column")
    meta_df = meta_df.sort_values("ts")
    base = pd.DataFrame({"ts": ts_int})
    aligned = pd.merge_asof(base, meta_df, on="ts", direction="backward")
    return aligned.drop(columns=["ts"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a simple instrument head on qfeat + meta features.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--prices-csv", type=Path, required=True, help="Prices CSV for timestamps.")
    ap.add_argument("--meta-features", type=Path, required=True, help="Market meta feature CSV.")
    ap.add_argument("--out", type=Path, default=Path("logs/instrument_head.json"))
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--order", type=int, default=1, help="Window order for qfeat.")
    ap.add_argument("--tau-on", type=float, default=0.5, help="ell gating threshold.")
    ap.add_argument("--ell-min", type=float, default=None, help="Override ell gate threshold.")
    ap.add_argument("--iv-high", type=float, default=60.0, help="Option IV threshold.")
    ap.add_argument("--burst-high", type=float, default=1.5, help="Burstiness threshold.")
    ap.add_argument("--funding-thresh", type=float, default=0.0005, help="Funding rate threshold.")
    ap.add_argument("--acorr-trend", type=float, default=0.2, help="ACF threshold for trend.")
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3, help="L2 regularization.")
    ap.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on training rows.")
    args = ap.parse_args()

    price, _volume, ts = load_prices(args.prices_csv, return_time=True)
    ts_int = pd.to_datetime(ts).astype("int64") if ts is not None else np.arange(price.size, dtype=np.int64)

    tape = QFeatTape.from_existing(str(args.tape), rows=price.size)
    if args.series < 0 or args.series >= tape.num_series:
        raise SystemExit(f"series index {args.series} out of range (S={tape.num_series})")
    if tape.T != price.size:
        raise SystemExit(f"Tape length (T={tape.T}) != prices length (T={price.size})")

    meta = _align_meta(args.meta_features, ts_int)
    meta = meta.ffill().fillna(0.0)
    meta_cols = list(meta.columns)
    meta_arr = meta.to_numpy(dtype=np.float32)

    qfeat = tape.mm[args.series, :, :6].astype(np.float32, copy=False)
    ell = tape.mm[args.series, :, 6].astype(np.float32, copy=False)

    order = max(1, int(args.order))
    windows = np.lib.stride_tricks.sliding_window_view(qfeat, window_shape=order, axis=0)
    feat_mat = windows.reshape(-1, order * qfeat.shape[1])

    t_start = order - 1
    idx = np.arange(t_start, price.size)
    X_q = feat_mat[idx - t_start]
    X_meta = meta_arr[idx]
    X = np.concatenate([X_q, X_meta], axis=1)

    ell_gate = args.ell_min if args.ell_min is not None else args.tau_on
    mask = np.isfinite(X).all(axis=1) & np.isfinite(ell[idx]) & (ell[idx] >= ell_gate)
    if mask.sum() == 0:
        raise SystemExit("No training rows after gating/finite filters.")
    X = X[mask]

    # pseudo-labels
    burst = X_q[mask][:, 3]
    acorr = X_q[mask][:, 4]
    iv = meta.loc[idx[mask], "opt_mark_iv_mean"].to_numpy(dtype=float)
    funding = meta.loc[idx[mask], "premium_funding_rate"].to_numpy(dtype=float)

    high_iv = np.isfinite(iv) & (iv >= args.iv_high)
    high_burst = np.isfinite(burst) & (burst >= args.burst_high)
    high_funding = np.isfinite(funding) & (np.abs(funding) >= args.funding_thresh)
    trend = np.isfinite(acorr) & (acorr >= args.acorr_trend)

    labels = np.full(X.shape[0], "SPOT", dtype=object)
    labels[high_funding & trend] = "PERP"
    labels[high_iv & high_burst] = "OPTION"

    if args.max_rows is not None and X.shape[0] > args.max_rows:
        X = X[: args.max_rows]
        labels = labels[: args.max_rows]

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Xn = (X - mean) / std

    classes = ["SPOT", "PERP", "OPTION"]
    class_map = {c: i for i, c in enumerate(classes)}
    y_idx = np.vectorize(class_map.get)(labels)

    W, b = _train_softmax(
        Xn.astype(np.float32),
        y_idx.astype(np.int64),
        num_classes=len(classes),
        lr=float(args.lr),
        epochs=int(args.epochs),
        lam=float(args.lam),
    )

    logits = Xn @ W + b
    probs = _softmax(logits)
    pred = probs.argmax(axis=1)
    acc = float(np.mean(pred == y_idx))
    counts = {c: int(np.sum(labels == c)) for c in classes}

    payload = {
        "order": int(order),
        "ell_gate": float(ell_gate),
        "classes": classes,
        "feature_qfeat": ["vol_ratio", "curvature", "drawdown", "burstiness", "acorr_1", "var_ratio"],
        "feature_meta": meta_cols,
        "weights": W.tolist(),
        "bias": b.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "lambda": float(args.lam),
        "metrics": {
            "rows": int(X.shape[0]),
            "acc": acc,
            "class_counts": counts,
        },
        "heuristic": {
            "iv_high": float(args.iv_high),
            "burst_high": float(args.burst_high),
            "funding_thresh": float(args.funding_thresh),
            "acorr_trend": float(args.acorr_trend),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

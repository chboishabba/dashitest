"""
train_direction_head.py
-----------------------
Train a simple linear softmax direction head on qfeat windows.

Labels are ternary based on future log return over horizon H:
  +1 if r_t > +deadzone
   0 if |r_t| <= deadzone
  -1 if r_t < -deadzone

Training is gated by ell (e.g., ell >= tau_on).

Usage:
  PYTHONPATH=. python trading/scripts/train_direction_head.py \
    --tape logs/qfeat_btc.us_20260113_210153.memmap \
    --prices-csv data/raw/stooq/btc.us.csv \
    --out logs/dir_head_btc.us.json
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


def _compute_deadzone(
    log_returns: np.ndarray,
    horizon: int,
    deadzone: float | None,
    deadzone_mult: float | None,
    vol_window: int,
) -> np.ndarray:
    if deadzone is not None:
        return np.full_like(log_returns, fill_value=float(deadzone), dtype=float)
    if deadzone_mult is None:
        return np.zeros_like(log_returns, dtype=float)
    vol = pd.Series(log_returns).rolling(vol_window).std().to_numpy()
    return deadzone_mult * vol * math.sqrt(float(horizon))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a linear direction head on qfeat windows.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--prices-csv", type=Path, required=True, help="Prices CSV for labels.")
    ap.add_argument("--out", type=Path, default=Path("logs/dir_head.json"))
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--order", type=int, default=1, help="Window order for qfeat.")
    ap.add_argument("--horizon", type=int, default=10, help="Future return horizon in bars.")
    ap.add_argument("--deadzone", type=float, default=None, help="Absolute deadzone for labels.")
    ap.add_argument("--deadzone-mult", type=float, default=None, help="Vol-scaled deadzone multiplier.")
    ap.add_argument("--vol-window", type=int, default=50, help="Vol window for deadzone scaling.")
    ap.add_argument("--tau-on", type=float, default=0.5, help="ell gating threshold.")
    ap.add_argument("--ell-min", type=float, default=None, help="Override ell gate threshold.")
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3, help="L2 regularization.")
    ap.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on training rows.")
    args = ap.parse_args()

    price, _volume, _ts = load_prices(args.prices_csv, return_time=True)
    tape = QFeatTape.from_existing(str(args.tape))
    if args.series < 0 or args.series >= tape.num_series:
        raise SystemExit(f"series index {args.series} out of range (S={tape.num_series})")
    if tape.T != price.size:
        raise SystemExit(f"Tape length (T={tape.T}) != prices length (T={price.size})")

    qfeat = tape.mm[args.series, :, :6].astype(np.float32, copy=False)
    ell = tape.mm[args.series, :, 6].astype(np.float32, copy=False)

    order = max(1, int(args.order))
    horizon = max(1, int(args.horizon))
    log_price = np.log(np.clip(price, 1e-12, None))
    log_returns = np.diff(log_price, prepend=log_price[0])
    deadzone = _compute_deadzone(
        log_returns,
        horizon=horizon,
        deadzone=args.deadzone,
        deadzone_mult=args.deadzone_mult,
        vol_window=max(2, int(args.vol_window)),
    )

    windows = np.lib.stride_tricks.sliding_window_view(qfeat, window_shape=order, axis=0)
    t_start = order - 1
    t_end = price.size - horizon
    if t_end <= t_start:
        raise SystemExit("Not enough rows for requested order/horizon.")

    idx = np.arange(t_start, t_end)
    X = windows[: t_end - order + 1].reshape(-1, order * qfeat.shape[1])
    X = X[idx - t_start]

    future_ret = log_price[idx + horizon] - log_price[idx]
    dz = deadzone[idx]
    y = np.zeros_like(future_ret, dtype=int)
    y[future_ret > dz] = 1
    y[future_ret < -dz] = -1

    ell_gate = args.ell_min if args.ell_min is not None else args.tau_on
    mask = np.isfinite(X).all(axis=1) & np.isfinite(future_ret) & (ell[idx] >= ell_gate)
    if mask.sum() == 0:
        raise SystemExit("No training rows after gating/finite filters.")

    X = X[mask]
    y = y[mask]
    if args.max_rows is not None and X.shape[0] > args.max_rows:
        X = X[: args.max_rows]
        y = y[: args.max_rows]

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Xn = (X - mean) / std

    class_map = {-1: 0, 0: 1, 1: 2}
    y_idx = np.vectorize(class_map.get)(y)

    W, b = _train_softmax(
        Xn.astype(np.float32),
        y_idx.astype(np.int64),
        num_classes=3,
        lr=float(args.lr),
        epochs=int(args.epochs),
        lam=float(args.lam),
    )

    logits = Xn @ W + b
    probs = _softmax(logits)
    pred = probs.argmax(axis=1)
    acc = float(np.mean(pred == y_idx))
    counts = {str(k): int(np.sum(y == k)) for k in (-1, 0, 1)}

    payload = {
        "order": int(order),
        "horizon": int(horizon),
        "deadzone": float(args.deadzone) if args.deadzone is not None else None,
        "deadzone_mult": float(args.deadzone_mult) if args.deadzone_mult is not None else None,
        "vol_window": int(args.vol_window),
        "ell_gate": float(ell_gate),
        "classes": [-1, 0, 1],
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
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

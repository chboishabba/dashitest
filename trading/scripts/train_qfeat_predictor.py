from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _load_qfeat_tape(
    path: Path,
    series: int,
    limit: int | None,
    rows: int | None,
) -> np.ndarray:
    mm = np.memmap(path, dtype=np.float32, mode="r")
    if mm.size % 8 != 0:
        raise ValueError("tape size is not divisible by 8 floats")
    records = mm.size // 8
    if records == 0:
        raise ValueError("empty tape")
    if series < 0:
        raise ValueError("series index must be >= 0")
    if rows is None:
        if series > 0:
            raise ValueError("series index >0 requires --rows or --log to infer tape shape")
        # Default to a single-series tape if we cannot infer shape.
        S = 1
        T = records
    else:
        if rows <= 0:
            raise ValueError("rows must be positive")
        if records % rows != 0:
            raise ValueError("tape length does not align with rows")
        T = rows
        S = records // rows
        if series >= S:
            raise ValueError(f"series index {series} out of range (S={S})")
    tape = np.memmap(path, dtype=np.float32, mode="r", shape=(S, T, 8))
    qfeat = tape[series, :, :6].astype(np.float32, copy=False)
    if limit is not None:
        qfeat = qfeat[:limit]
    return qfeat


def _sliding_xy(qfeat: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 1:
        raise ValueError("order must be >= 1")
    if qfeat.shape[0] <= order:
        raise ValueError("not enough rows for the requested order")
    windows = np.lib.stride_tricks.sliding_window_view(qfeat, window_shape=order, axis=0)
    X = windows[:-1].reshape(-1, order * qfeat.shape[1])
    y = qfeat[order:]
    return X, y


def _robust_scale(qfeat: np.ndarray) -> np.ndarray:
    med = np.nanmedian(qfeat, axis=0)
    mad = np.nanmedian(np.abs(qfeat - med), axis=0)
    std = np.nanstd(qfeat, axis=0)
    scale = np.where(mad > 0, mad, std)
    scale = np.where(scale > 0, scale, 1.0)
    return scale.astype(np.float32)


def _fit_ridge_huber(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    delta: float,
    iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    out_dim = y.shape[1]
    Xb = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    w = np.zeros((d + 1, out_dim), dtype=X.dtype)
    weights = np.ones(n, dtype=X.dtype)
    for _ in range(max(1, iters)):
        sqrt_w = np.sqrt(weights)
        Xw = Xb * sqrt_w[:, None]
        yw = y * sqrt_w[:, None]
        xtx = Xw.T @ Xw
        xty = Xw.T @ yw
        reg = lam * np.eye(d + 1, dtype=X.dtype)
        w = np.linalg.solve(xtx + reg, xty)
        resid = y - Xb @ w
        e = np.linalg.norm(resid, axis=1)
        weights = np.where(e <= delta, 1.0, delta / np.maximum(e, 1e-12))
    return w[:-1], w[-1]


def _predict(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ w + b


def _calibrate_alpha(e: np.ndarray, target: float) -> float:
    med = float(np.nanmedian(e))
    if not math.isfinite(med) or med <= 0:
        return 1.0
    return float(math.log(target ** -1.0) / med)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a qfeat predictor and derive ell from prediction error.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--order", type=int, default=1, help="History order (AR window).")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for training.")
    ap.add_argument("--rows", type=int, default=None, help="Rows per series in tape.")
    ap.add_argument("--log", type=Path, default=None, help="Optional log CSV to infer rows.")
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-2, help="Ridge regularization.")
    ap.add_argument("--delta", type=float, default=1.0, help="Huber delta for IRLS.")
    ap.add_argument("--iters", type=int, default=5, help="IRLS iterations.")
    ap.add_argument("--alpha", type=float, default=None, help="Ell temperature (optional).")
    ap.add_argument("--alpha-target", type=float, default=0.5, help="Median ell target if alpha unset.")
    ap.add_argument("--emit-ell", type=Path, default=None, help="Optional CSV to write ell + error.")
    ap.add_argument("--out", type=Path, default=Path("logs/qfeat_predictor.json"))
    args = ap.parse_args()

    rows = args.rows
    if rows is None and args.log is not None:
        rows = _count_rows(args.log)
    qfeat = _load_qfeat_tape(args.tape, args.series, args.limit, rows)
    X, y = _sliding_xy(qfeat, args.order)
    w, b = _fit_ridge_huber(X, y, args.lam, args.delta, args.iters)

    yhat = _predict(X, w, b)
    scale = _robust_scale(y)
    resid = (y - yhat) / scale
    e = np.linalg.norm(resid, axis=1)

    alpha = args.alpha
    if alpha is None:
        alpha = _calibrate_alpha(e, args.alpha_target)
    ell = np.exp(-alpha * e)

    payload = {
        "order": int(args.order),
        "weights": w.tolist(),
        "bias": b.tolist(),
        "scale": scale.tolist(),
        "lambda": float(args.lam),
        "delta": float(args.delta),
        "iters": int(args.iters),
        "alpha": float(alpha),
        "metrics": {
            "rows": int(X.shape[0]),
            "e_mean": float(np.nanmean(e)),
            "e_p50": float(np.nanquantile(e, 0.50)),
            "ell_mean": float(np.nanmean(ell)),
            "ell_p10": float(np.nanquantile(ell, 0.10)),
            "ell_p50": float(np.nanquantile(ell, 0.50)),
            "ell_p90": float(np.nanquantile(ell, 0.90)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))

    if args.emit_ell is not None:
        args.emit_ell.parent.mkdir(parents=True, exist_ok=True)
        with args.emit_ell.open("w", encoding="utf-8") as f:
            f.write("t,ell,e\n")
            for i, (ell_i, e_i) in enumerate(zip(ell, e, strict=False)):
                f.write(f"{i},{ell_i:.8f},{e_i:.8f}\n")


if __name__ == "__main__":
    main()

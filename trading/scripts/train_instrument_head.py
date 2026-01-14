"""
train_instrument_head.py
------------------------
Train a simple instrument-family classifier using qfeat + market meta features.

Teacher uses soft scores (no PnL) to build a target distribution. Training is
gated by ell (e.g., ell >= tau_on).
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
    from utils.weights_config import (
        parse_simple_yaml,
        get_path,
        resolve_value,
        format_resolved,
        validate_never_learnable,
        validate_known_paths,
    )
except ModuleNotFoundError:
    from trading_io.prices import load_prices
    from vk_qfeat import QFeatTape
    from utils.weights_config import (
        parse_simple_yaml,
        get_path,
        resolve_value,
        format_resolved,
        validate_never_learnable,
        validate_known_paths,
    )


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.clip(expz.sum(axis=1, keepdims=True), 1e-12, None)


def _softmax_vec(scores: np.ndarray, temp: float) -> np.ndarray:
    t = max(1e-6, float(temp))
    x = scores / t
    x = x - np.max(x)
    expx = np.exp(x)
    return expx / np.clip(expx.sum(), 1e-12, None)


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.zeros_like(arr, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not math.isfinite(std) or std <= 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def _train_softmax(
    X: np.ndarray,
    y_soft: np.ndarray,
    num_classes: int,
    lr: float,
    epochs: int,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    W = np.zeros((d, num_classes), dtype=np.float32)
    b = np.zeros(num_classes, dtype=np.float32)
    for _ in range(max(1, epochs)):
        logits = X @ W + b
        probs = _softmax(logits)
        grad = (probs - y_soft) / max(1, n)
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
    ap.add_argument("--tau-on", type=float, default=None, help="ell gating threshold.")
    ap.add_argument("--ell-min", type=float, default=None, help="Override ell gate threshold.")
    ap.add_argument("--inst-temp", type=float, default=1.0, help="Instrument score temperature.")
    ap.add_argument("--inst-w-iv", type=float, default=1.4, help="Option score weight for IV.")
    ap.add_argument("--inst-w-hazard", type=float, default=1.0, help="Option score weight for hazard.")
    ap.add_argument("--inst-w-funding", type=float, default=0.6, help="Option score weight for funding.")
    ap.add_argument("--inst-w-basis", type=float, default=0.6, help="Option score weight for basis.")
    ap.add_argument("--inst-w-opt-oi", type=float, default=0.5, help="Option score weight for options OI.")
    ap.add_argument("--inst-w-opt-vol", type=float, default=0.3, help="Option score weight for options volume.")
    ap.add_argument("--inst-w-opt-imb", type=float, default=0.4, help="Option score weight for put/call imbalance.")
    ap.add_argument("--inst-w-leg", type=float, default=0.5, help="Option score weight for ell margin.")
    ap.add_argument("--perp-w-dirconf", type=float, default=1.2, help="Perp score weight for direction confidence.")
    ap.add_argument("--perp-w-carry", type=float, default=0.8, help="Perp score weight for funding carry.")
    ap.add_argument("--perp-w-oi", type=float, default=0.6, help="Perp score weight for OI value.")
    ap.add_argument("--perp-w-iv", type=float, default=1.0, help="Perp score weight for IV.")
    ap.add_argument("--perp-w-hazard", type=float, default=0.8, help="Perp score weight for hazard.")
    ap.add_argument("--perp-w-leg", type=float, default=0.4, help="Perp score weight for ell margin.")
    ap.add_argument("--spot-base", type=float, default=0.3, help="Spot base prior.")
    ap.add_argument("--spot-w-funding", type=float, default=0.8, help="Spot score weight for funding.")
    ap.add_argument("--spot-w-basis", type=float, default=0.8, help="Spot score weight for basis.")
    ap.add_argument("--spot-w-hazard", type=float, default=0.6, help="Spot score weight for hazard.")
    ap.add_argument("--hazard-a", type=float, default=1.0, help="Hazard weight for burstiness.")
    ap.add_argument("--hazard-b", type=float, default=1.0, help="Hazard weight for curvature.")
    ap.add_argument("--hazard-c", type=float, default=1.0, help="Hazard weight for vol_ratio.")
    ap.add_argument("--horizon", type=int, default=None, help="Utility horizon in bars.")
    ap.add_argument("--deadzone", type=float, default=None, help="Return deadzone for direction labels.")
    ap.add_argument("--utility-temp", type=float, default=None, help="Utility softmax temperature.")
    ap.add_argument("--teacher-mix", type=float, default=None, help="Mix realized utility vs heuristic scores.")
    ap.add_argument("--funding-lambda", type=float, default=None, help="Funding cost multiplier for perp utility.")
    ap.add_argument("--option-iv-penalty", type=float, default=None, help="IV penalty for option utility.")
    ap.add_argument("--option-slip", type=float, default=None, help="Flat slippage penalty for option utility.")
    ap.add_argument("--weights", type=Path, default=Path("weights.yaml"), help="Optional weights.yaml")
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3, help="L2 regularization.")
    ap.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap on training rows.")
    ap.add_argument("--dump-schema", action="store_true", help="Print resolved weight sources.")
    args = ap.parse_args()

    weights = {}
    if args.weights is not None and args.weights.exists():
        weights = parse_simple_yaml(args.weights)
        validate_known_paths(weights)
        validate_never_learnable(weights)

    resolved: dict[str, tuple[object, str]] = {}
    args.tau_on = resolve_value(args.tau_on, weights, ["thresholds", "tau_on"], 0.5, "thresholds.tau_on", resolved)
    args.horizon = int(
        resolve_value(args.horizon, weights, ["teachers", "instrument", "horizon_bars"], 10, "teachers.instrument.horizon_bars", resolved)
    )
    args.deadzone = float(
        resolve_value(args.deadzone, weights, ["teachers", "instrument", "deadzone"], 0.0, "teachers.instrument.deadzone", resolved)
    )
    args.utility_temp = float(
        resolve_value(args.utility_temp, weights, ["teachers", "instrument", "utility_temp"], 1.0, "teachers.instrument.utility_temp", resolved)
    )
    args.teacher_mix = float(
        resolve_value(args.teacher_mix, weights, ["teachers", "instrument", "teacher_mix"], 0.5, "teachers.instrument.teacher_mix", resolved)
    )
    args.funding_lambda = float(
        resolve_value(args.funding_lambda, weights, ["teachers", "instrument", "funding_lambda"], 1.0, "teachers.instrument.funding_lambda", resolved)
    )
    args.option_iv_penalty = float(
        resolve_value(args.option_iv_penalty, weights, ["teachers", "instrument", "option_iv_penalty"], 0.01, "teachers.instrument.option_iv_penalty", resolved)
    )
    args.option_slip = float(
        resolve_value(args.option_slip, weights, ["teachers", "instrument", "option_slip"], 0.0, "teachers.instrument.option_slip", resolved)
    )

    if args.dump_schema:
        dump = format_resolved(resolved)
        if dump:
            print(dump)

    price, _volume, ts = load_prices(args.prices_csv, return_time=True)
    ts_int = pd.to_datetime(ts).astype("int64") if ts is not None else np.arange(price.size, dtype=np.int64)

    log_price = np.log(np.clip(price, 1e-12, None))
    horizon = max(1, int(args.horizon))
    future_ret = np.full(price.shape[0], np.nan, dtype=float)
    if price.size > horizon:
        future_ret[: -horizon] = log_price[horizon:] - log_price[:-horizon]

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

    # teacher scores (soft labels)
    burst = X_q[mask][:, 3]
    curvature = X_q[mask][:, 1]
    vol_ratio = X_q[mask][:, 0]
    acorr = X_q[mask][:, 4]
    hazard = args.hazard_a * burst + args.hazard_b * curvature + args.hazard_c * vol_ratio
    z_hazard = _zscore(hazard)
    z_trend = _zscore(np.abs(acorr))
    z_ell_margin = _zscore(ell[idx][mask] - args.tau_on)

    iv = meta.loc[idx[mask], "opt_mark_iv_p50"].to_numpy(dtype=float)
    if np.isnan(iv).all():
        iv = meta.loc[idx[mask], "opt_mark_iv_mean"].to_numpy(dtype=float)
    iv_chg = np.diff(iv, prepend=iv[0]) if iv.size else np.zeros_like(iv)
    z_iv = _zscore(iv)
    z_iv_chg = _zscore(iv_chg)

    funding = meta.loc[idx[mask], "premium_funding_rate"].to_numpy(dtype=float)
    z_funding_abs = _zscore(np.abs(funding))
    z_funding_signed = _zscore(funding)

    if "premium_mark_price" in meta.columns and "premium_index_price" in meta.columns:
        basis = meta.loc[idx[mask], "premium_mark_price"].to_numpy(dtype=float) - meta.loc[idx[mask], "premium_index_price"].to_numpy(dtype=float)
    else:
        basis = np.zeros_like(iv)
    z_basis = _zscore(np.abs(basis))

    oi_val = meta.loc[idx[mask], "oi_sum_open_interest_value"].to_numpy(dtype=float)
    oi_chg = np.diff(oi_val, prepend=oi_val[0]) if oi_val.size else np.zeros_like(oi_val)
    z_oi_chg = _zscore(oi_chg)
    z_oi_val = _zscore(oi_val)

    opt_oi = meta.loc[idx[mask], "opt_open_interest_sum"].to_numpy(dtype=float)
    opt_vol = meta.loc[idx[mask], "opt_volume_sum"].to_numpy(dtype=float)
    z_opt_oi = _zscore(opt_oi)
    z_opt_vol = _zscore(opt_vol)

    opt_count = meta.loc[idx[mask], "opt_count"].to_numpy(dtype=float)
    opt_call = meta.loc[idx[mask], "opt_call_count"].to_numpy(dtype=float)
    opt_put = meta.loc[idx[mask], "opt_put_count"].to_numpy(dtype=float)
    denom = np.where(opt_count != 0, opt_count, 1.0)
    opt_imb = (opt_put - opt_call) / denom
    z_opt_imb = _zscore(opt_imb)

    s_opt = (
        args.inst_w_iv * z_iv
        + args.inst_w_hazard * z_hazard
        + args.inst_w_funding * z_funding_abs
        + args.inst_w_basis * z_basis
        + args.inst_w_opt_oi * z_opt_oi
        + args.inst_w_opt_vol * z_opt_vol
        + args.inst_w_opt_imb * z_opt_imb
        + args.inst_w_leg * z_ell_margin
    )
    dir_sign = np.sign(acorr)
    carry = -funding * dir_sign
    z_carry = _zscore(carry)
    s_perp = (
        args.perp_w_dirconf * z_trend
        + args.perp_w_carry * z_carry
        + args.perp_w_oi * z_oi_val
        - args.perp_w_iv * z_iv
        - args.perp_w_hazard * z_hazard
        + args.perp_w_leg * z_ell_margin
    )
    s_spot = args.spot_base - args.spot_w_funding * z_funding_abs - args.spot_w_basis * z_basis - args.spot_w_hazard * z_hazard
    scores = np.stack([s_spot, s_perp, s_opt], axis=1)
    y_heur = np.array([_softmax_vec(row, args.inst_temp) for row in scores], dtype=np.float32)

    r = future_ret[idx][mask]
    dir_label = np.sign(r)
    dir_label[np.abs(r) <= float(args.deadzone)] = 0.0
    u_spot = dir_label * r
    u_perp = u_spot - float(args.funding_lambda) * funding * dir_label
    iv_penalty = float(args.option_iv_penalty) * (iv / 100.0)
    u_opt = np.abs(r) - iv_penalty - float(args.option_slip)
    u_scores = np.stack([_zscore(u_spot), _zscore(u_perp), _zscore(u_opt)], axis=1)
    u_soft = np.array([_softmax_vec(row, args.utility_temp) for row in u_scores], dtype=np.float32)

    mix = float(args.teacher_mix)
    y_soft = mix * u_soft + (1.0 - mix) * y_heur

    if args.max_rows is not None and X.shape[0] > args.max_rows:
        X = X[: args.max_rows]
        y_soft = y_soft[: args.max_rows]

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Xn = (X - mean) / std

    classes = ["SPOT", "PERP", "OPTION"]

    W, b = _train_softmax(
        Xn.astype(np.float32),
        y_soft.astype(np.float32),
        num_classes=len(classes),
        lr=float(args.lr),
        epochs=int(args.epochs),
        lam=float(args.lam),
    )

    logits = Xn @ W + b
    probs = _softmax(logits)
    pred = probs.argmax(axis=1)
    hard = np.argmax(y_soft, axis=1)
    acc = float(np.mean(pred == hard))
    counts = {c: int(np.sum(hard == i)) for i, c in enumerate(classes)}

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
        "teacher": {
            "inst_temp": float(args.inst_temp),
            "inst_w_iv": float(args.inst_w_iv),
            "inst_w_hazard": float(args.inst_w_hazard),
            "inst_w_funding": float(args.inst_w_funding),
            "inst_w_basis": float(args.inst_w_basis),
            "inst_w_opt_oi": float(args.inst_w_opt_oi),
            "inst_w_opt_vol": float(args.inst_w_opt_vol),
            "inst_w_opt_imb": float(args.inst_w_opt_imb),
            "inst_w_leg": float(args.inst_w_leg),
            "perp_w_dirconf": float(args.perp_w_dirconf),
            "perp_w_carry": float(args.perp_w_carry),
            "perp_w_oi": float(args.perp_w_oi),
            "perp_w_iv": float(args.perp_w_iv),
            "perp_w_hazard": float(args.perp_w_hazard),
            "perp_w_leg": float(args.perp_w_leg),
            "spot_base": float(args.spot_base),
            "spot_w_funding": float(args.spot_w_funding),
            "spot_w_basis": float(args.spot_w_basis),
            "spot_w_hazard": float(args.spot_w_hazard),
            "hazard_a": float(args.hazard_a),
            "hazard_b": float(args.hazard_b),
            "hazard_c": float(args.hazard_c),
            "horizon": int(args.horizon),
            "deadzone": float(args.deadzone),
            "utility_temp": float(args.utility_temp),
            "teacher_mix": float(args.teacher_mix),
            "funding_lambda": float(args.funding_lambda),
            "option_iv_penalty": float(args.option_iv_penalty),
            "option_slip": float(args.option_slip),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

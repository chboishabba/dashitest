"""
train_per_ontology.py
---------------------
Train per-ontology preference weights using a pairwise rank loss.

This script is read-only with respect to ontology, legitimacy, veto, and geometry.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from trading.trading_io.prices import load_prices
    from trading.vk_qfeat import QFeatTape
    from utils.forward_proxy import forward_return_proxy
    from utils.rank_loss import pairwise_rank_loss
    from utils.weights_config import (
        parse_simple_yaml,
        get_path,
        clamp_learnable_vector,
        normalize_vector,
    )
except ModuleNotFoundError:
    from trading_io.prices import load_prices
    from vk_qfeat import QFeatTape
    from utils.forward_proxy import forward_return_proxy
    from utils.rank_loss import pairwise_rank_loss
    from utils.weights_config import (
        parse_simple_yaml,
        get_path,
        clamp_learnable_vector,
        normalize_vector,
    )


def _cfg_get(cfg: dict, path: list[str], default: object) -> object:
    return get_path(cfg, path, default)


def _normalize(values: list[float]) -> np.ndarray:
    normed = normalize_vector(values)
    return np.array(normed, dtype=np.float32)


def _init_weights() -> dict[str, dict[str, np.ndarray]]:
    weights: dict[str, dict[str, np.ndarray]] = {}
    weights["T"] = {
        "score_weights": _normalize([1, 1, 1, 1, 1, 1]),
        "instrument_weights": _normalize([1.0, 1.0, 0.5]),
        "opt_tenor_weights": _normalize([0.5, 0.8, 1.0, 0.6, 0.3]),
        "opt_mny_weights": _normalize([0.3, 0.6, 1.0, 0.6, 0.2]),
        "size_weights": _normalize([1, 1, 1, 1]),
    }
    weights["R"] = {
        "score_weights": _normalize([1, 1, 1, 1, 1, 1]),
        "instrument_weights": _normalize([0.8, 0.8, 1.0]),
        "opt_tenor_weights": _normalize([0.3, 0.6, 1.0, 0.8, 0.4]),
        "opt_mny_weights": _normalize([0.2, 0.5, 1.0, 0.7, 0.3]),
        "size_weights": _normalize([1, 1, 1, 1]),
    }
    weights["H"] = {
        "score_weights": _normalize([1, 1, 1, 1, 1, 1]),
        "instrument_weights": _normalize([0.2, 0.2, 0.1]),
        "opt_tenor_weights": _normalize([0.1, 0.1, 0.1, 0.1, 0.1]),
        "opt_mny_weights": _normalize([0.1, 0.1, 0.1, 0.1, 0.1]),
        "size_weights": _normalize([1, 1, 1, 1]),
    }
    return weights


def _clamp_and_normalize(name: str, vec: np.ndarray) -> np.ndarray:
    clamped = clamp_learnable_vector(vec.tolist(), name)
    return np.array(normalize_vector(clamped), dtype=np.float32)


def _safe_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        row = arr[i]
        if not np.isfinite(row).all() or float(np.sum(row)) <= 0.0:
            out[i] = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float64)
        else:
            out[i] = row / float(np.sum(row))
    return out.astype(np.float32)


def _log_probs(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.log(np.clip(arr, eps, None)).astype(np.float32)


def _entropy_grad(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = float(np.sum(vec))
    if not math.isfinite(s) or s <= 0:
        return np.zeros_like(vec, dtype=np.float32)
    p = vec / s
    logp = np.log(np.clip(p, eps, None))
    a = float(np.sum(vec * (logp + 1.0)))
    grad = (logp + 1.0) / s - a / (s * s)
    return grad.astype(np.float32)


def _extract_probs(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = df[cols].to_numpy(dtype=np.float32, copy=True)
    return _safe_probs(arr)


def _select_rows(
    df: pd.DataFrame,
    ont: str,
    conf_min: float,
    target: np.ndarray,
) -> np.ndarray:
    mask = df["ontology_k"].astype(str).to_numpy() == ont
    if "ontology_confidence" in df.columns:
        conf = df["ontology_confidence"].to_numpy(dtype=np.float32)
        mask = mask & (conf >= conf_min)
    idx = df["i"].to_numpy(dtype=int)
    mask = mask & np.isfinite(target[idx])
    return np.where(mask)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train per-ontology preference weights.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--proposal-log", type=Path, required=True, help="Proposal log CSV.")
    ap.add_argument("--prices-csv", type=Path, required=True, help="Prices CSV for forward proxy.")
    ap.add_argument("--config", type=Path, required=True, help="Training config YAML.")
    ap.add_argument("--out", type=Path, required=True, help="Output weights JSON.")
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed.")
    args = ap.parse_args()

    cfg = parse_simple_yaml(args.config)
    epochs = int(_cfg_get(cfg, ["training", "epochs"], 8))
    batch_size = int(_cfg_get(cfg, ["training", "batch_size"], 256))
    lr = float(_cfg_get(cfg, ["training", "learning_rate"], 5.0e-4))
    grad_clip = float(_cfg_get(cfg, ["training", "grad_clip_norm"], 1.0))
    conf_min = float(_cfg_get(cfg, ["training", "confidence_min"], 0.60))

    l2_reg = float(_cfg_get(cfg, ["loss", "l2_reg"], 1.0e-3))
    entropy_reg = float(_cfg_get(cfg, ["loss", "entropy_reg"], 1.0e-3))

    horizon = int(_cfg_get(cfg, ["forward_proxy", "horizon"], 8))
    clip_return = float(_cfg_get(cfg, ["forward_proxy", "clip_return"], 0.01))
    use_log_return = bool(_cfg_get(cfg, ["forward_proxy", "use_log_return"], True))

    ontologies = _cfg_get(cfg, ["ontologies"], ["T", "R", "H"])
    learnable_cfg = _cfg_get(cfg, ["learnable"], {})
    log_every = int(_cfg_get(cfg, ["logging", "log_every"], 200))
    dump_weights = bool(_cfg_get(cfg, ["logging", "dump_weights"], True))
    dump_grad_norms = bool(_cfg_get(cfg, ["logging", "dump_grad_norms"], True))

    learnable = {
        "score_weights": bool(learnable_cfg.get("score_weights", True)),
        "instrument_weights": bool(learnable_cfg.get("instrument_weights", True)),
        "opt_tenor_weights": bool(learnable_cfg.get("opt_tenor_weights", False)),
        "opt_mny_weights": bool(learnable_cfg.get("opt_mny_weights", False)),
        "size_weights": bool(learnable_cfg.get("size_weights", False)),
    }

    price, _volume, _ts = load_prices(args.prices_csv, return_time=True)
    forward = forward_return_proxy(
        price,
        horizon=horizon,
        clip_return=clip_return,
        use_log_return=use_log_return,
    )

    tape = QFeatTape.from_existing(str(args.tape), rows=price.size)
    if args.series < 0 or args.series >= tape.num_series:
        raise SystemExit(f"series index {args.series} out of range (S={tape.num_series})")
    if tape.T != price.size:
        raise SystemExit(f"Tape length (T={tape.T}) != prices length (T={price.size})")
    qfeat = tape.mm[args.series, :, :6].astype(np.float32, copy=False)
    q_mean = np.nanmean(qfeat, axis=0)
    q_std = np.nanstd(qfeat, axis=0)
    q_std = np.where(q_std <= 0, 1.0, q_std)

    df = pd.read_csv(args.proposal_log)
    if "i" not in df.columns or "ontology_k" not in df.columns:
        raise SystemExit("proposal log must include columns: i, ontology_k")
    idx = df["i"].to_numpy(dtype=int)
    if idx.max() >= qfeat.shape[0]:
        raise SystemExit("proposal log index exceeds tape length")

    score_feat = (qfeat[idx] - q_mean) / q_std
    inst_probs = _extract_probs(df, ["p_spot", "p_perp", "p_option"])
    size_probs = _extract_probs(df, ["p_size_0", "p_size_0_5", "p_size_1", "p_size_2"])
    tenor_probs = _extract_probs(
        df,
        [
            "p_opt_e_1_3",
            "p_opt_e_4_7",
            "p_opt_e_8_21",
            "p_opt_e_22_60",
            "p_opt_e_61_180",
        ],
    )
    mny_probs = _extract_probs(
        df,
        [
            "p_opt_m_deep_itm",
            "p_opt_m_itm",
            "p_opt_m_atm",
            "p_opt_m_otm",
            "p_opt_m_deep_otm",
        ],
    )
    inst_logp = _log_probs(inst_probs)
    size_logp = _log_probs(size_probs)
    tenor_logp = _log_probs(tenor_probs)
    mny_logp = _log_probs(mny_probs)

    rng = np.random.default_rng(int(args.seed))
    weights = _init_weights()
    init_weights = copy.deepcopy(weights)

    step = 0
    for epoch in range(max(1, epochs)):
        for ont in ontologies:
            if ont not in weights:
                continue
            rows = _select_rows(df, ont, conf_min, forward)
            if rows.size < batch_size:
                continue
            batch_idx = rng.choice(rows, size=batch_size, replace=False)

            s_feat = score_feat[batch_idx]
            i_feat = inst_logp[batch_idx]
            z_feat = size_logp[batch_idx]
            t_feat = tenor_logp[batch_idx]
            m_feat = mny_logp[batch_idx]
            targets = forward[idx[batch_idx]]

            w = weights[ont]
            scores = (
                s_feat @ w["score_weights"]
                + i_feat @ w["instrument_weights"]
                + z_feat @ w["size_weights"]
                + t_feat @ w["opt_tenor_weights"]
                + m_feat @ w["opt_mny_weights"]
            ).astype(np.float32)

            loss, grad_scores, used = pairwise_rank_loss(scores, targets, rng)
            if used == 0:
                continue

            grads = {
                "score_weights": (s_feat.T @ grad_scores) / float(batch_size),
                "instrument_weights": (i_feat.T @ grad_scores) / float(batch_size),
                "size_weights": (z_feat.T @ grad_scores) / float(batch_size),
                "opt_tenor_weights": (t_feat.T @ grad_scores) / float(batch_size),
                "opt_mny_weights": (m_feat.T @ grad_scores) / float(batch_size),
            }

            for name, grad in grads.items():
                grad = grad.astype(np.float32)
                grad += np.float32(2.0 * l2_reg) * (w[name] - init_weights[ont][name])
                grad += np.float32(entropy_reg) * _entropy_grad(w[name])
                if not learnable.get(name, False):
                    continue
                norm = float(np.linalg.norm(grad))
                if math.isfinite(norm) and norm > grad_clip and norm > 0:
                    grad = grad * (grad_clip / norm)
                w[name] = w[name] - np.float32(lr) * grad
                w[name] = _clamp_and_normalize(name, w[name])

            step += 1
            if log_every > 0 and step % log_every == 0:
                msg = f"epoch={epoch} ont={ont} loss={loss:.6f} pairs={used}"
                if dump_grad_norms:
                    norms = {k: float(np.linalg.norm(v)) for k, v in grads.items()}
                    msg += f" grad_norms={norms}"
                print(msg)

        if dump_weights:
            snapshot = {k: {n: v.tolist() for n, v in w.items()} for k, w in weights.items()}
            print(f"epoch={epoch} weights={json.dumps(snapshot, separators=(',', ':'))}")

    out_payload = {
        "config": cfg,
        "weights": {k: {n: v.tolist() for n, v in w.items()} for k, w in weights.items()},
        "init_weights": {k: {n: v.tolist() for n, v in w.items()} for k, w in init_weights.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

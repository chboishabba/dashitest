# scripts/train_size_per_ontology.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.forward_proxy import forward_return_proxy
from utils.weights_config import normalize_vector


def _load_prices(path: Path, close_col: str | None = None) -> np.ndarray:
    df = pd.read_csv(path)
    candidate_cols = [close_col] if close_col else []
    candidate_cols += ["close", "Close", "Zamkniecie", "Zamknie"]
    for col in candidate_cols:
        if col and col in df.columns:
            return df[col].astype(float).to_numpy()
    raise SystemExit(f"Cannot find a close column in {path.name}")


def _compute_forward(prices: np.ndarray, horizon: int, clip: float) -> np.ndarray:
    return forward_return_proxy(
        prices, horizon=horizon, clip_return=clip, use_log_return=True
    )


def _summarize_ontology(df: pd.DataFrame, forward: np.ndarray, column: str) -> dict:
    stats: dict[str, dict] = {}
    if df.empty:
        return stats

    if column not in df.columns:
        return stats
    for size_bin, group in df.groupby(column):
        idx = group["i"].astype(int).to_numpy()
        valid = (idx >= 0) & (idx < len(forward))
        idx = idx[valid]
        vals = forward[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q75, q25 = np.percentile(vals, [75, 25])
        stats[str(size_bin)] = {
            "count": int(vals.size),
            "median": float(np.median(vals)),
            "iqr": float(q75 - q25),
            "mean_clipped": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    return stats


def _bin_index(value: float | str | int, max_bins: int) -> int | None:
    if pd.isna(value):
        return None
    try:
        idx = int(float(value))
    except (ValueError, TypeError):
        return None
    if 0 <= idx < max_bins:
        return idx
    return None


def _pairwise_rank_step(weights: np.ndarray, pos: int, neg: int, lr: float) -> np.ndarray:
    delta = -(weights[pos] - weights[neg])
    sigma = 1.0 / (1.0 + np.exp(-delta))
    weights[pos] += lr * sigma
    weights[neg] -= lr * sigma
    return weights


def _prepare_training_rows(
    df: pd.DataFrame, forward: np.ndarray, horizon: int
) -> pd.DataFrame:
    would_act = df["would_act"] if "would_act" in df else pd.Series(False, index=df.index)
    act_mask = (
        (df["action"] == "ACT")
        | would_act.fillna(False).astype(bool)
        | (df.get("state", 0) == 1)
    )
    rows = df[act_mask].copy()
    if rows.empty:
        return rows
    rows["i"] = rows["i"].astype(int)
    valid = (rows["i"] >= 0) & (rows["i"] + horizon < len(forward))
    rows = rows[valid].copy()
    idx = rows["i"].to_numpy()
    rows["proxy"] = forward[idx]
    return rows


def _train_ontology(
    weights: np.ndarray,
    rows: pd.DataFrame,
    horizon: int,
    lr: float,
    epochs: int,
    column: str,
) -> np.ndarray:
    max_bins = len(weights)
    rows = rows.copy()
    if column not in rows.columns:
        return weights
    rows["bin_idx"] = rows[column].map(lambda val: _bin_index(val, max_bins))
    rows = rows.dropna(subset=["bin_idx", "proxy"])
    if rows.empty:
        return weights
    rows["bin_idx"] = rows["bin_idx"].astype(int)

    for _ in range(epochs):
        medians = rows.groupby("bin_idx")["proxy"].median()
        if medians.size < 2:
            break
        bins = sorted(medians.index.tolist())
        for i in range(len(bins)):
            for j in range(i + 1, len(bins)):
                bin_i = bins[i]
                bin_j = bins[j]
                median_i = medians.loc[bin_i]
                median_j = medians.loc[bin_j]
                if median_i == median_j:
                    continue
                pos, neg = (bin_i, bin_j) if median_i > median_j else (bin_j, bin_i)
                weights = _pairwise_rank_step(weights, pos, neg, lr)
        weights = np.clip(weights, 0.05, 0.80)
        weights = np.array(normalize_vector(weights.tolist()), dtype=np.float32)

    return weights


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-4 size diagnostics + trainer.")
    ap.add_argument("--proposal-log", type=Path, required=True)
    ap.add_argument("--prices-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init-weights", type=Path, default=None)
    ap.add_argument("--weights-out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--clip", type=float, default=0.02)
    ap.add_argument("--size-column", type=str, default="size_pred", help="Proposal column that represents size bins.")
    ap.add_argument("--close-col", type=str, default=None)
    args = ap.parse_args()

    if not args.proposal_log.exists():
        raise SystemExit(f"Missing proposal log: {args.proposal_log}")

    df = pd.read_csv(args.proposal_log)
    required_cols = {"ontology_k", "i", args.size_column}
    for col in required_cols:
        if col not in df.columns:
            raise SystemExit(f"proposal log missing '{col}' column")

    prices = _load_prices(args.prices_csv, close_col=args.close_col)
    forward = _compute_forward(prices, args.horizon, args.clip)

    diagnostics: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for ont in ("T", "R", "H"):
        subset = df[df["ontology_k"] == ont]
        diagnostics[ont] = _summarize_ontology(subset, forward, args.size_column)
        counts[ont] = int(subset[args.size_column].notna().sum())

    payload: dict[str, object] = {
        "horizon": args.horizon,
        "clip": args.clip,
        "counts": counts,
        "diagnostics": diagnostics,
    }

    weights: dict | None = None
    if args.init_weights:
        if not args.init_weights.exists():
            raise SystemExit(f"Missing init weights: {args.init_weights}")
        weights_payload = json.loads(args.init_weights.read_text())
        weights = weights_payload.get("weights", weights_payload)

        for ont in ("T", "R"):
            if ont not in weights or "size_weights" not in weights[ont]:
                raise SystemExit(f"init weights missing {ont}.size_weights")
            arr = np.array(weights[ont]["size_weights"], dtype=np.float32)
            ont_rows = _prepare_training_rows(
                df[df["ontology_k"] == ont], forward, args.horizon
            )
            arr = _train_ontology(
                arr,
                ont_rows,
                args.horizon,
                args.lr,
                args.epochs,
                args.size_column,
            )
            weights[ont]["size_weights"] = arr.tolist()

        hazard_len = len(weights["H"]["size_weights"])
        weights["H"]["size_weights"] = [1.0 / hazard_len] * hazard_len

        payload["weights"] = weights

        if args.weights_out:
            args.weights_out.parent.mkdir(parents=True, exist_ok=True)
            args.weights_out.write_text(json.dumps({"weights": weights}, indent=2))
            print(f"Wrote {args.weights_out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

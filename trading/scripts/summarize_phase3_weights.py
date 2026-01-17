"""
summarize_phase3_weights.py
---------------------------
Summarize per-ontology weights across multiple tapes and optionally aggregate.
"""

from __future__ import annotations

import argparse
import json
import numbers
from pathlib import Path

import numpy as np

try:
    from utils.weights_config import clamp_learnable_vector, normalize_vector
except ModuleNotFoundError:
    from weights_config import clamp_learnable_vector, normalize_vector


def validate_weights_schema(weights_in, *, name="<weights>") -> None:
    if not isinstance(weights_in, dict):
        raise ValueError(f"{name}: root must be dict")

    weights = weights_in.get("weights", weights_in)
    for ont in ("T", "R", "H"):
        if ont not in weights:
            raise ValueError(f"{name}: missing ontology '{ont}'")

        block = weights[ont]
        if not isinstance(block, dict):
            raise ValueError(f"{name}.{ont}: must be dict")

        for key, val in block.items():
            if not isinstance(val, list):
                raise ValueError(
                    f"{name}.{ont}.{key}: expected list, got {type(val).__name__}"
                )
            if len(val) == 0:
                raise ValueError(f"{name}.{ont}.{key}: empty vector")

            for i, x in enumerate(val):
                if not isinstance(x, numbers.Real):
                    raise ValueError(
                        f"{name}.{ont}.{key}[{i}]: expected float, got {type(x).__name__}"
                    )


def validate_consistency(weight_list: list[dict[str, dict[str, np.ndarray]]]) -> None:
    if not weight_list:
        return
    ref = weight_list[0]
    for w in weight_list[1:]:
        for ont in ("T", "R", "H"):
            for key in ref[ont]:
                if key not in w[ont]:
                    raise ValueError(f"Missing {ont}.{key} in one run")
                if len(ref[ont][key]) != len(w[ont][key]):
                    raise ValueError(
                        f"Length mismatch for {ont}.{key}: "
                        f"{len(ref[ont][key])} vs {len(w[ont][key])}"
                    )


def _defaults() -> dict[str, dict[str, np.ndarray]]:
    def _norm(vals: list[float]) -> np.ndarray:
        return np.array(normalize_vector(vals), dtype=np.float32)

    return {
        "T": {
            "score_weights": _norm([1, 1, 1, 1, 1, 1]),
            "instrument_weights": _norm([1.0, 1.0, 0.5]),
            "opt_tenor_weights": _norm([0.5, 0.8, 1.0, 0.6, 0.3]),
            "opt_mny_weights": _norm([0.3, 0.6, 1.0, 0.6, 0.2]),
            "size_weights": _norm([1, 1, 1, 1]),
        },
        "R": {
            "score_weights": _norm([1, 1, 1, 1, 1, 1]),
            "instrument_weights": _norm([0.8, 0.8, 1.0]),
            "opt_tenor_weights": _norm([0.3, 0.6, 1.0, 0.8, 0.4]),
            "opt_mny_weights": _norm([0.2, 0.5, 1.0, 0.7, 0.3]),
            "size_weights": _norm([1, 1, 1, 1]),
        },
        "H": {
            "score_weights": _norm([1, 1, 1, 1, 1, 1]),
            "instrument_weights": _norm([0.2, 0.2, 0.1]),
            "opt_tenor_weights": _norm([0.1, 0.1, 0.1, 0.1, 0.1]),
            "opt_mny_weights": _norm([0.1, 0.1, 0.1, 0.1, 0.1]),
            "size_weights": _norm([1, 1, 1, 1]),
        },
    }


def _weights_from_payload(payload: dict) -> dict[str, dict[str, np.ndarray]]:
    weights = payload.get("weights", payload)
    out: dict[str, dict[str, np.ndarray]] = {}
    for ont, group in weights.items():
        out[ont] = {}
        for name, vec in group.items():
            out[ont][name] = np.array(vec, dtype=np.float32)
    return out


def _load_weights(path: Path) -> dict[str, dict[str, np.ndarray]]:
    payload = json.loads(path.read_text())
    return _weights_from_payload(payload)


def _aggregate(
    stacks: dict[str, dict[str, list[np.ndarray]]],
    default: dict[str, dict[str, np.ndarray]],
    lam: float,
) -> dict[str, dict[str, np.ndarray]]:
    agg: dict[str, dict[str, np.ndarray]] = {}
    for ont, groups in stacks.items():
        agg[ont] = {}
        for name, vecs in groups.items():
            mat = np.stack(vecs, axis=0)
            med = np.median(mat, axis=0)
            base = default[ont][name]
            mixed = (1.0 - lam) * base + lam * med
            mixed = np.array(clamp_learnable_vector(mixed.tolist(), name), dtype=np.float32)
            mixed = np.array(normalize_vector(mixed.tolist()), dtype=np.float32)
            agg[ont][name] = mixed
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize and aggregate Phase-3 weights.")
    ap.add_argument("--weights", type=Path, required=True, help="Weights file or directory.")
    ap.add_argument("--out", type=Path, default=None, help="Write aggregated weights JSON.")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.25, help="Blend factor vs defaults.")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.weights.is_dir():
        paths = sorted(args.weights.glob("*.json"))
    else:
        paths = [args.weights]
    if not paths:
        raise SystemExit("No weight files found.")

    default = _defaults()

    payloads: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text())
        validate_weights_schema(payload, name=str(path))
        payloads.append(payload)

    weight_dicts = [_weights_from_payload(payload) for payload in payloads]
    validate_consistency(weight_dicts)

    stacks: dict[str, dict[str, list[np.ndarray]]] = {}
    for weights in weight_dicts:
        for ont, group in weights.items():
            stacks.setdefault(ont, {})
            for name, vec in group.items():
                stacks[ont].setdefault(name, []).append(vec)

    summary = {}
    for ont, group in stacks.items():
        summary[ont] = {}
        for name, vecs in group.items():
            mat = np.stack(vecs, axis=0)
            base = default[ont][name]
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            min_v = mat.min(axis=0)
            max_v = mat.max(axis=0)
            max_abs_delta = float(np.max(np.abs(mean - base)))
            summary[ont][name] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "min": min_v.tolist(),
                "max": max_v.tolist(),
                "max_abs_delta_vs_default": max_abs_delta,
                "num_tapes": int(mat.shape[0]),
            }

    agg = _aggregate(stacks, default, float(args.lam))
    payload = {
        "sources": [str(p) for p in paths],
        "lambda": float(args.lam),
        "summary": summary,
        "aggregated_weights": {k: {n: v.tolist() for n, v in w.items()} for k, w in agg.items()},
    }

    print(json.dumps(payload, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

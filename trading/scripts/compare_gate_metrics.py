from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compute_gate_metrics import compute_metrics


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = np.sort(a)
    b = np.sort(b)
    data_all = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, data_all, side="right") / a.size
    cdf_b = np.searchsorted(b, data_all, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _load_ell(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return pd.to_numeric(df.get("ell", np.nan), errors="coerce").to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare gate metrics across two logs.")
    ap.add_argument("--log-a", type=Path, required=True, help="First trading log CSV.")
    ap.add_argument("--log-b", type=Path, required=True, help="Second trading log CSV.")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    df_a = pd.read_csv(args.log_a)
    df_b = pd.read_csv(args.log_b)
    metrics_a = compute_metrics(df_a)
    metrics_b = compute_metrics(df_b)

    ell_a = _load_ell(args.log_a)
    ell_b = _load_ell(args.log_b)
    ks = _ks_distance(ell_a, ell_b)

    delta = {
        "hold_pct": float(metrics_b["hold_pct"] - metrics_a["hold_pct"]),
        "act_pct": float(metrics_b["act_pct"] - metrics_a["act_pct"]),
        "flip_rate": float(metrics_b["flip_rate"] - metrics_a["flip_rate"]),
        "ell_mean": float(metrics_b["ell_mean"] - metrics_a["ell_mean"]),
        "ell_p10": float(metrics_b["ell_p10"] - metrics_a["ell_p10"]),
        "ell_p50": float(metrics_b["ell_p50"] - metrics_a["ell_p50"]),
        "ell_p90": float(metrics_b["ell_p90"] - metrics_a["ell_p90"]),
        "ks_ell": ks,
    }

    payload = {
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "delta": delta,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

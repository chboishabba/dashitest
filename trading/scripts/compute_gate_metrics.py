from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=int)
    lengths = []
    count = 0
    for v in mask:
        if v:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return np.asarray(lengths, dtype=int)


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    direction = pd.to_numeric(df.get("intent_direction", df.get("action")), errors="coerce").fillna(0).to_numpy()
    ell = pd.to_numeric(df.get("ell", np.nan), errors="coerce").to_numpy()
    act = direction != 0
    hold = direction == 0

    flip_rate = float(np.mean(direction[1:] != direction[:-1])) if direction.size > 1 else float("nan")
    act_runs = _run_lengths(act)
    hold_runs = _run_lengths(hold)

    metrics = {
        "rows": int(len(df)),
        "hold_pct": float(np.mean(hold)) if len(df) else float("nan"),
        "act_pct": float(np.mean(act)) if len(df) else float("nan"),
        "flip_rate": flip_rate,
        "act_runs_mean": float(np.mean(act_runs)) if act_runs.size else float("nan"),
        "act_runs_median": float(np.median(act_runs)) if act_runs.size else float("nan"),
        "hold_runs_mean": float(np.mean(hold_runs)) if hold_runs.size else float("nan"),
        "hold_runs_median": float(np.median(hold_runs)) if hold_runs.size else float("nan"),
        "ell_mean": float(np.nanmean(ell)) if ell.size else float("nan"),
        "ell_p10": float(np.nanquantile(ell, 0.10)) if ell.size else float("nan"),
        "ell_p50": float(np.nanquantile(ell, 0.50)) if ell.size else float("nan"),
        "ell_p90": float(np.nanquantile(ell, 0.90)) if ell.size else float("nan"),
    }

    if "exposure" in df:
        exposure = pd.to_numeric(df["exposure"], errors="coerce").to_numpy()
        metrics["exposure_mean"] = float(np.nanmean(np.abs(exposure)))
    if "intent_target" in df:
        target = pd.to_numeric(df["intent_target"], errors="coerce").to_numpy()
        metrics["intent_target_mean"] = float(np.nanmean(np.abs(target)))

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute gating metrics from a trading log CSV.")
    ap.add_argument("--log", type=Path, required=True, help="Path to trading_log CSV.")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    metrics = compute_metrics(df)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

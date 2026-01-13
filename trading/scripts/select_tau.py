from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compute_gate_metrics import compute_metrics


def _parse_tau_from_name(path: Path) -> tuple[float | None, float | None]:
    matches = re.findall(r"(-?\d+(?:\.\d+)?)", path.stem)
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    return None, None


def _load_from_logs(pattern: str) -> pd.DataFrame:
    rows = []
    for log_path in sorted(Path().glob(pattern)):
        df = pd.read_csv(log_path)
        metrics = compute_metrics(df)
        tau_on = df.get("tau_on")
        tau_off = df.get("tau_off")
        if tau_on is not None and tau_off is not None:
            tau_on = float(pd.to_numeric(tau_on, errors="coerce").iloc[-1])
            tau_off = float(pd.to_numeric(tau_off, errors="coerce").iloc[-1])
        else:
            tau_on, tau_off = _parse_tau_from_name(log_path)
        metrics["tau_on"] = tau_on
        metrics["tau_off"] = tau_off
        metrics["log_path"] = str(log_path)
        rows.append(metrics)
    if not rows:
        raise SystemExit(f"No logs matched {pattern!r}")
    return pd.DataFrame(rows)


def _score_row(
    row: pd.Series,
    hold_target: float,
    w_hold: float,
    w_flip: float,
    w_runs: float,
) -> float:
    hold = float(row.get("hold_pct", np.nan))
    flip = float(row.get("flip_rate", np.nan))
    hold_runs = float(row.get("hold_runs_mean", np.nan))
    hold_runs = hold_runs if np.isfinite(hold_runs) else 0.0
    return w_hold * abs(hold - hold_target) + w_flip * flip - w_runs * hold_runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Select tau_on/tau_off from gate metrics.")
    ap.add_argument("--sweep-csv", type=Path, default=None, help="CSV with tau metrics.")
    ap.add_argument("--logs-glob", type=str, default=None, help="Glob for log CSVs.")
    ap.add_argument("--hold-min", type=float, default=0.4, help="Minimum hold percentage.")
    ap.add_argument("--hold-max", type=float, default=0.6, help="Maximum hold percentage.")
    ap.add_argument("--hold-target", type=float, default=0.5, help="Target hold percentage.")
    ap.add_argument("--flip-max", type=float, default=0.2, help="Maximum flip rate.")
    ap.add_argument("--w-hold", type=float, default=10.0, help="Hold deviation weight.")
    ap.add_argument("--w-flip", type=float, default=2.0, help="Flip rate weight.")
    ap.add_argument("--w-runs", type=float, default=0.1, help="Hold runs weight.")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    if args.sweep_csv is None and args.logs_glob is None:
        args.sweep_csv = Path("logs/tau_sweep.csv")

    if args.sweep_csv is not None:
        if not args.sweep_csv.exists():
            raise SystemExit(f"sweep csv not found: {args.sweep_csv}")
        df = pd.read_csv(args.sweep_csv)
    else:
        df = _load_from_logs(args.logs_glob)

    required = {"hold_pct", "flip_rate", "tau_on", "tau_off"}
    if not required.issubset(set(df.columns)):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise SystemExit(f"missing required columns: {missing}")

    df = df.copy()
    df["score"] = df.apply(
        _score_row,
        axis=1,
        hold_target=args.hold_target,
        w_hold=args.w_hold,
        w_flip=args.w_flip,
        w_runs=args.w_runs,
    )
    mask = (
        (df["flip_rate"] <= args.flip_max)
        & (df["hold_pct"] >= args.hold_min)
        & (df["hold_pct"] <= args.hold_max)
    )
    filtered = df[mask]
    if filtered.empty:
        best = df.sort_values("score").iloc[0]
        note = "no candidate met constraints; selected best score overall"
    else:
        best = filtered.sort_values("score").iloc[0]
        note = "selected best score within constraints"

    payload = {
        "tau_on": float(best["tau_on"]),
        "tau_off": float(best["tau_off"]),
        "score": float(best["score"]),
        "hold_pct": float(best["hold_pct"]),
        "flip_rate": float(best["flip_rate"]),
        "act_pct": float(best.get("act_pct", float("nan"))),
        "hold_runs_mean": float(best.get("hold_runs_mean", float("nan"))),
        "note": note,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

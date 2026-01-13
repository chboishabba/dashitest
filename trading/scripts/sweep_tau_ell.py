from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def apply_gate(state: np.ndarray, ell: np.ndarray, tau_on: float, tau_off: float) -> np.ndarray:
    n = min(state.size, ell.size)
    out = np.zeros(n, dtype=int)
    is_holding = False
    for i in range(n):
        conf = float(ell[i])
        s = int(state[i])
        hold_by_conf = False
        if not is_holding:
            if conf < tau_on:
                hold_by_conf = True
        else:
            if conf < tau_off:
                hold_by_conf = True
        if s == 0 or hold_by_conf or conf <= 0.0:
            out[i] = 0
            is_holding = True
        else:
            out[i] = s
            is_holding = False
    return out


def run_lengths(mask: np.ndarray) -> np.ndarray:
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


def compute_metrics(direction: np.ndarray) -> dict[str, float]:
    act = direction != 0
    hold = direction == 0
    flip_rate = float(np.mean(direction[1:] != direction[:-1])) if direction.size > 1 else float("nan")
    act_runs = run_lengths(act)
    hold_runs = run_lengths(hold)
    return {
        "hold_pct": float(np.mean(hold)) if direction.size else float("nan"),
        "act_pct": float(np.mean(act)) if direction.size else float("nan"),
        "flip_rate": flip_rate,
        "act_runs_mean": float(np.mean(act_runs)) if act_runs.size else float("nan"),
        "hold_runs_mean": float(np.mean(hold_runs)) if hold_runs.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep tau_on/tau_off over ell + state log.")
    ap.add_argument("--log", type=Path, required=True, help="Trading log CSV with ell + state.")
    ap.add_argument("--tau-on", type=float, nargs="+", required=True, help="Entry thresholds to sweep.")
    ap.add_argument("--tau-off", type=float, nargs="+", required=True, help="Exit thresholds to sweep.")
    ap.add_argument("--out", type=Path, default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    if "ell" not in df or "state" not in df:
        raise SystemExit("log must contain ell and state columns")
    ell = pd.to_numeric(df["ell"], errors="coerce").fillna(0.0).to_numpy()
    state = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int).to_numpy()

    rows = []
    for tau_on in args.tau_on:
        for tau_off in args.tau_off:
            if tau_on < tau_off:
                continue
            direction = apply_gate(state, ell, tau_on=tau_on, tau_off=tau_off)
            metrics = compute_metrics(direction)
            rows.append({**metrics, "tau_on": tau_on, "tau_off": tau_off})

    out_df = pd.DataFrame(rows).sort_values(["tau_on", "tau_off"]).reset_index(drop=True)
    print(out_df.to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

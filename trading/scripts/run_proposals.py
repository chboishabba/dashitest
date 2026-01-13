"""
run_proposals.py
----------------
Generate direction proposals and PnL-veto decisions without execution.

Outputs a proposal log (CSV) with per-bar diagnostics:
  ts, i, price, ell, actionability, margin, acceptable,
  dir_pred, p_long, p_flat, p_short,
  delta_pnl (future log return proxy), delta_pnl_signed,
  veto, veto_reason, would_act
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from trading.regime import RegimeSpec, check_regime
    from trading.signals.triadic import compute_triadic_state
    from trading.trading_io.prices import load_prices
    from trading.vk_qfeat import QFeatTape
except ModuleNotFoundError:
    from regime import RegimeSpec, check_regime
    from signals.triadic import compute_triadic_state
    from trading_io.prices import load_prices
    from vk_qfeat import QFeatTape


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.clip(expz.sum(axis=1, keepdims=True), 1e-12, None)


def _infer_margin(states: np.ndarray, prices: np.ndarray, spec: RegimeSpec) -> np.ndarray:
    runs = np.zeros(len(states), dtype=int)
    run = 0
    for i, s in enumerate(states):
        if s == 0:
            run = 0
        else:
            if i > 0 and s == states[i - 1]:
                run += 1
            else:
                run = 1
        runs[i] = run

    flips = np.abs(np.diff(states, prepend=states[0])) > 0
    fr = np.zeros_like(states, dtype=float)
    w = max(1, spec.window)
    for i in range(len(states)):
        lo = max(0, i - w + 1)
        span = max(1, i - lo)
        fr[i] = flips[lo:i].sum() / span if span > 0 else 0.0

    rets = np.diff(prices, prepend=prices[0])
    vols = pd.Series(rets).rolling(spec.window).std().to_numpy()

    big = 1e9
    margins = np.full_like(states, fill_value=big, dtype=float)
    if spec.min_run_length is not None:
        margins = np.minimum(margins, runs - spec.min_run_length)
    if spec.max_flip_rate is not None:
        margins = np.minimum(margins, spec.max_flip_rate - fr)
    if spec.max_vol is not None:
        margins = np.minimum(margins, spec.max_vol - vols)
    return margins


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.n < 2:
            return float("nan")
        return math.sqrt(self.m2 / (self.n - 1))


def _ell_bin(ell: float, bins: int) -> int:
    if not math.isfinite(ell):
        return -1
    idx = int(math.floor(ell * bins))
    if idx < 0:
        return 0
    if idx >= bins:
        return bins - 1
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate direction proposals and veto decisions.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--prices-csv", type=Path, required=True, help="Prices CSV for bars.")
    ap.add_argument("--dir-model", type=Path, required=True, help="Direction head JSON.")
    ap.add_argument("--proposal-log", type=Path, required=True, help="Output proposal CSV.")
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--tau-on", type=float, default=0.5, help="ell hysteresis on.")
    ap.add_argument("--tau-off", type=float, default=0.49, help="ell hysteresis off.")
    ap.add_argument("--ell-bins", type=int, default=10, help="Number of ell bins for veto stats.")
    ap.add_argument("--kappa", type=float, default=0.0, help="Veto threshold: mean - kappa*std <= 0.")
    ap.add_argument("--min-samples", type=int, default=50, help="Min samples before veto activates.")
    ap.add_argument("--horizon", type=int, default=None, help="Override horizon for delta_pnl.")
    ap.add_argument("--min-run-length", type=int, default=3, help="RegimeSpec min_run_length.")
    ap.add_argument("--max-flip-rate", type=float, default=None, help="RegimeSpec max_flip_rate.")
    ap.add_argument("--max-vol", type=float, default=None, help="RegimeSpec max_vol.")
    ap.add_argument("--window", type=int, default=50, help="RegimeSpec window.")
    args = ap.parse_args()

    if args.tau_on < args.tau_off:
        raise SystemExit("Require tau-on >= tau-off for hysteresis.")

    price, volume, ts = load_prices(args.prices_csv, return_time=True)
    tape = QFeatTape.from_existing(str(args.tape))
    if args.series < 0 or args.series >= tape.num_series:
        raise SystemExit(f"series index {args.series} out of range (S={tape.num_series})")
    if tape.T != price.size:
        raise SystemExit(f"Tape length (T={tape.T}) != prices length (T={price.size})")

    state = compute_triadic_state(price)
    ts_int = pd.to_datetime(ts).astype("int64") if ts is not None else np.arange(price.size, dtype=np.int64)
    price = price.astype(float)

    model = json.loads(args.dir_model.read_text())
    order = int(model["order"])
    classes = model["classes"]
    W = np.array(model["weights"], dtype=np.float32)
    b = np.array(model["bias"], dtype=np.float32)
    mean = np.array(model["mean"], dtype=np.float32)
    std = np.array(model["std"], dtype=np.float32)
    horizon = int(args.horizon) if args.horizon is not None else int(model.get("horizon", 10))

    qfeat = tape.mm[args.series, :, :6].astype(np.float32, copy=False)
    ell = tape.mm[args.series, :, 6].astype(np.float32, copy=False)

    log_price = np.log(np.clip(price, 1e-12, None))
    future_ret = np.full(price.shape[0], np.nan, dtype=float)
    if price.size > horizon:
        future_ret[: -horizon] = log_price[horizon:] - log_price[:-horizon]

    spec = RegimeSpec(
        min_run_length=args.min_run_length,
        max_flip_rate=args.max_flip_rate,
        max_vol=args.max_vol,
        window=args.window,
    )
    rets = np.diff(price, prepend=price[0])
    vols = pd.Series(rets).rolling(spec.window).std().to_numpy()
    acceptable = np.asarray(check_regime(state, vols, spec), dtype=bool)
    margin = _infer_margin(state, price, spec)

    windows = np.lib.stride_tricks.sliding_window_view(qfeat, window_shape=order, axis=0)
    feat_mat = windows.reshape(-1, order * qfeat.shape[1])

    stats: dict[tuple[int, int], RunningStats] = {}
    rows = []
    is_holding = False

    for t in range(price.size):
        row = {
            "i": int(t),
            "ts": int(ts_int[t]),
            "price": float(price[t]),
            "state": int(state[t]),
            "ell": float(ell[t]),
            "actionability": float(ell[t]),
            "margin": float(margin[t]) if math.isfinite(margin[t]) else float("nan"),
            "acceptable": bool(acceptable[t]),
        }

        gate_open = True
        if not is_holding:
            if ell[t] < args.tau_on:
                gate_open = False
        else:
            if ell[t] < args.tau_off:
                gate_open = False
        if ell[t] <= 0.0 or not math.isfinite(ell[t]):
            gate_open = False

        if t < order - 1 or t >= price.size - horizon or not gate_open:
            row.update(
                {
                    "dir_pred": 0,
                    "p_long": float("nan"),
                    "p_flat": float("nan"),
                    "p_short": float("nan"),
                    "delta_pnl": float(future_ret[t]),
                    "delta_pnl_signed": float("nan"),
                    "veto": 0,
                    "veto_reason": "gate_closed" if not gate_open else "insufficient_history",
                    "would_act": "HOLD",
                    "action": 0,
                    "hold": 1,
                }
            )
            is_holding = True
            rows.append(row)
            continue

        x = feat_mat[t - (order - 1)]
        x = (x - mean) / std
        logits = x @ W + b
        probs = _softmax(logits.reshape(1, -1)).ravel()
        class_idx = int(np.argmax(probs))
        dir_pred = int(classes[class_idx])

        delta_pnl = float(future_ret[t])
        delta_signed = float(delta_pnl * dir_pred) if math.isfinite(delta_pnl) else float("nan")

        ell_bin = _ell_bin(float(ell[t]), args.ell_bins)
        key = (dir_pred, ell_bin)
        if key not in stats:
            stats[key] = RunningStats()
        if dir_pred != 0:
            stats[key].update(delta_signed)

        veto = 0
        veto_reason = ""
        st = stats[key]
        if dir_pred == 0:
            veto_reason = "flat"
        elif st.n < args.min_samples:
            veto = 1
            veto_reason = "insufficient_samples"
        else:
            std = st.std()
            thresh = st.mean - args.kappa * (std if math.isfinite(std) else 0.0)
            if thresh <= 0.0:
                veto = 1
                veto_reason = "mean_minus_kstd<0"

        if dir_pred == 0 or veto == 1:
            would_act = "HOLD"
            action = 0
            is_holding = True
        else:
            would_act = "ACT_LONG" if dir_pred > 0 else "ACT_SHORT"
            action = int(dir_pred)
            is_holding = False

        row.update(
            {
                "dir_pred": dir_pred,
                "p_long": float(probs[classes.index(1)]) if 1 in classes else float("nan"),
                "p_flat": float(probs[classes.index(0)]) if 0 in classes else float("nan"),
                "p_short": float(probs[classes.index(-1)]) if -1 in classes else float("nan"),
                "delta_pnl": delta_pnl,
                "delta_pnl_signed": delta_signed,
                "veto": int(veto),
                "veto_reason": veto_reason,
                "would_act": would_act,
                "action": action,
                "hold": int(action == 0),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    args.proposal_log.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.proposal_log, index=False)
    print(f"Wrote {args.proposal_log} ({len(df)} rows)")


if __name__ == "__main__":
    main()

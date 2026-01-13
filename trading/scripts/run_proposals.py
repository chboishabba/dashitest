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


@dataclass
class BucketBuffer:
    size: int
    values: np.ndarray
    index: int = 0
    count: int = 0

    @classmethod
    def create(cls, size: int) -> "BucketBuffer":
        return cls(size=size, values=np.zeros(size, dtype=float))

    def add(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self.values[self.index] = x
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self) -> np.ndarray:
        return self.values[: self.count]


def _ell_bin(ell: float, bins: int) -> int:
    if not math.isfinite(ell):
        return -1
    idx = int(math.floor(ell * bins))
    if idx < 0:
        return 0
    if idx >= bins:
        return bins - 1
    return idx


def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not math.isfinite(std) or std <= 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def _softmax_vec(scores: np.ndarray, temp: float) -> np.ndarray:
    t = max(1e-6, float(temp))
    x = scores / t
    x = x - np.max(x)
    expx = np.exp(x)
    return expx / np.clip(expx.sum(), 1e-12, None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate direction proposals and veto decisions.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--prices-csv", type=Path, required=True, help="Prices CSV for bars.")
    ap.add_argument("--dir-model", type=Path, required=True, help="Direction head JSON.")
    ap.add_argument("--proposal-log", type=Path, required=True, help="Output proposal CSV.")
    ap.add_argument("--meta-features", type=Path, default=None, help="Optional market meta feature CSV.")
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--tau-on", type=float, default=0.5, help="ell hysteresis on.")
    ap.add_argument("--tau-off", type=float, default=0.49, help="ell hysteresis off.")
    ap.add_argument("--ell-bins", type=int, default=10, help="Number of ell bins for veto stats.")
    ap.add_argument("--veto-mode", type=str, default="cvar", choices=["meanstd", "cvar"])
    ap.add_argument("--kappa", type=float, default=0.0, help="Mean/std veto: mean - kappa*std <= 0.")
    ap.add_argument("--cvar-alpha", type=float, default=0.10, help="CVaR tail alpha.")
    ap.add_argument("--epsilon", type=float, default=0.0, help="CVaR veto threshold.")
    ap.add_argument("--veto-min-samples", type=int, default=50, help="Min samples before veto activates.")
    ap.add_argument("--veto-buffer", type=int, default=1024, help="Per-bucket buffer size.")
    ap.add_argument("--veto-cooldown", type=int, default=25, help="Cooldown bars after veto.")
    ap.add_argument("--hazard-veto", action="store_true", help="Enable hazard veto.")
    ap.add_argument("--hazard-threshold", type=float, default=2.0, help="Hazard veto threshold.")
    ap.add_argument("--hazard-a", type=float, default=1.0, help="Hazard weight for burstiness.")
    ap.add_argument("--hazard-b", type=float, default=1.0, help="Hazard weight for curvature.")
    ap.add_argument("--hazard-c", type=float, default=1.0, help="Hazard weight for vol_ratio.")
    ap.add_argument("--inst-temp", type=float, default=1.0, help="Instrument score temperature.")
    ap.add_argument("--inst-w-iv", type=float, default=1.0, help="Option score weight for IV.")
    ap.add_argument("--inst-w-iv-chg", type=float, default=0.5, help="Option score weight for IV change.")
    ap.add_argument("--inst-w-hazard", type=float, default=0.8, help="Option score weight for hazard.")
    ap.add_argument("--inst-w-funding", type=float, default=0.4, help="Option score weight for funding.")
    ap.add_argument("--inst-w-basis", type=float, default=0.4, help="Option score weight for basis.")
    ap.add_argument("--inst-w-ell", type=float, default=0.5, help="Option score weight for ell margin.")
    ap.add_argument("--perp-w-trend", type=float, default=0.8, help="Perp score weight for trend proxy.")
    ap.add_argument("--perp-w-iv", type=float, default=0.6, help="Perp score weight for IV.")
    ap.add_argument("--perp-w-hazard", type=float, default=0.6, help="Perp score weight for hazard.")
    ap.add_argument("--perp-w-carry", type=float, default=0.5, help="Perp score weight for funding carry.")
    ap.add_argument("--perp-w-oi", type=float, default=0.4, help="Perp score weight for OI change.")
    ap.add_argument("--spot-w-funding", type=float, default=0.3, help="Spot score weight for funding.")
    ap.add_argument("--spot-w-basis", type=float, default=0.3, help="Spot score weight for basis.")
    ap.add_argument("--spot-w-hazard", type=float, default=0.4, help="Spot score weight for hazard.")
    ap.add_argument("--exp-temp", type=float, default=1.0, help="Exposure score temperature.")
    ap.add_argument("--horizon", type=int, default=None, help="Override horizon for delta_pnl.")
    ap.add_argument("--min-run-length", type=int, default=3, help="RegimeSpec min_run_length.")
    ap.add_argument("--max-flip-rate", type=float, default=None, help="RegimeSpec max_flip_rate.")
    ap.add_argument("--max-vol", type=float, default=None, help="RegimeSpec max_vol.")
    ap.add_argument("--window", type=int, default=50, help="RegimeSpec window.")
    args = ap.parse_args()

    if args.tau_on < args.tau_off:
        raise SystemExit("Require tau-on >= tau-off for hysteresis.")

    price, volume, ts = load_prices(args.prices_csv, return_time=True)
    tape = QFeatTape.from_existing(str(args.tape), rows=price.size)
    if args.series < 0 or args.series >= tape.num_series:
        raise SystemExit(f"series index {args.series} out of range (S={tape.num_series})")
    if tape.T != price.size:
        raise SystemExit(f"Tape length (T={tape.T}) != prices length (T={price.size})")

    state = compute_triadic_state(price)
    ts_int = pd.to_datetime(ts).astype("int64") if ts is not None else np.arange(price.size, dtype=np.int64)
    price = price.astype(float)

    meta_cols = []
    meta_values = None
    if args.meta_features is not None:
        meta_df = pd.read_csv(args.meta_features)
        if "ts" not in meta_df.columns:
            raise SystemExit("meta-features CSV must contain a ts column")
        meta_df = meta_df.sort_values("ts")
        base = pd.DataFrame({"ts": ts_int})
        aligned = pd.merge_asof(base, meta_df, on="ts", direction="backward")
        aligned = aligned.drop(columns=["ts"])
        meta_cols = list(aligned.columns)
        meta_values = aligned.to_numpy(dtype=float)

    if meta_values is not None:
        meta_map = {name: meta_values[:, idx] for idx, name in enumerate(meta_cols)}
        iv = meta_map.get("opt_mark_iv_p50", meta_map.get("opt_mark_iv_mean"))
        basis = None
        if "premium_mark_price" in meta_map and "premium_index_price" in meta_map:
            basis = meta_map["premium_mark_price"] - meta_map["premium_index_price"]
        funding = meta_map.get("premium_funding_rate")
        oi_val = meta_map.get("oi_sum_open_interest_value")
        iv_chg = np.diff(iv, prepend=iv[0]) if iv is not None else None
        oi_chg = np.diff(oi_val, prepend=oi_val[0]) if oi_val is not None else None
        z_iv = _zscore(iv) if iv is not None else None
        z_iv_chg = _zscore(iv_chg) if iv_chg is not None else None
        z_basis = _zscore(np.abs(basis)) if basis is not None else None
        z_funding = _zscore(np.abs(funding)) if funding is not None else None
        z_oi_chg = _zscore(oi_chg) if oi_chg is not None else None
    else:
        iv = iv_chg = basis = funding = oi_chg = None
        z_iv = z_iv_chg = z_basis = z_funding = z_oi_chg = None

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
    z_hazard = _zscore(
        args.hazard_a * qfeat[:, 3] + args.hazard_b * qfeat[:, 1] + args.hazard_c * qfeat[:, 0]
    )
    z_trend = _zscore(np.abs(qfeat[:, 4]))
    z_ell_margin = _zscore(ell - args.tau_on)

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

    stats: dict[tuple[int, int, str], RunningStats] = {}
    buffers: dict[tuple[int, int, str], BucketBuffer] = {}
    rows = []
    is_holding = False
    cooldown = 0

    for t in range(price.size):
        qrow = qfeat[t]
        hazard = (
            args.hazard_a * float(qrow[3])
            + args.hazard_b * float(qrow[1])
            + args.hazard_c * float(qrow[0])
        )
        row = {
            "i": int(t),
            "ts": int(ts_int[t]),
            "price": float(price[t]),
            "state": int(state[t]),
            "ell": float(ell[t]),
            "actionability": float(ell[t]),
            "margin": float(margin[t]) if math.isfinite(margin[t]) else float("nan"),
            "acceptable": bool(acceptable[t]),
            "hazard": float(hazard),
        }
        if meta_values is not None:
            for idx, col in enumerate(meta_cols):
                row[col] = float(meta_values[t, idx]) if math.isfinite(meta_values[t, idx]) else float("nan")

        gate_open = True
        if not is_holding:
            if ell[t] < args.tau_on:
                gate_open = False
        else:
            if ell[t] < args.tau_off:
                gate_open = False
        if ell[t] <= 0.0 or not math.isfinite(ell[t]):
            gate_open = False

        if cooldown > 0:
            cooldown -= 1
            row.update(
                {
                    "dir_pred": 0,
                    "p_long": float("nan"),
                    "p_flat": float("nan"),
                    "p_short": float("nan"),
                    "delta_pnl": float(future_ret[t]),
                    "delta_pnl_signed": float("nan"),
                    "veto": 1,
                    "veto_reason": "cooldown",
                    "would_act": "HOLD",
                    "action": 0,
                    "hold": 1,
                    "bucket_n": 0,
                    "bucket_q": float("nan"),
                    "bucket_cvar": float("nan"),
                    "instrument_pred": "flat",
                    "p_spot": float("nan"),
                    "p_perp": float("nan"),
                    "p_option": float("nan"),
                    "exposure_pred": "flat",
                    "p_exp_small": float("nan"),
                    "p_exp_med": float("nan"),
                    "score_best": float("nan"),
                    "score_second": float("nan"),
                    "score_margin": float("nan"),
                }
            )
            is_holding = True
            rows.append(row)
            continue

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
                    "bucket_n": 0,
                    "bucket_q": float("nan"),
                    "bucket_cvar": float("nan"),
                    "instrument_pred": "flat",
                    "p_spot": float("nan"),
                    "p_perp": float("nan"),
                    "p_option": float("nan"),
                    "exposure_pred": "flat",
                    "p_exp_small": float("nan"),
                    "p_exp_med": float("nan"),
                    "score_best": float("nan"),
                    "score_second": float("nan"),
                    "score_margin": float("nan"),
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
        if key not in buffers:
            buffers[key] = BucketBuffer.create(int(args.veto_buffer))
        if dir_pred != 0:
            stats[key].update(delta_signed)
            buffers[key].add(delta_signed)

        veto = 0
        veto_reason = ""
        st = stats[key]
        buf = buffers[key]
        bucket_q = float("nan")
        bucket_cvar = float("nan")

        if dir_pred == 0:
            veto_reason = "flat"
        elif args.hazard_veto and hazard > args.hazard_threshold:
            veto = 1
            veto_reason = "hazard"
        elif st.n < args.veto_min_samples:
            veto = 1
            veto_reason = "insufficient_samples"
        else:
            if args.veto_mode == "meanstd":
                std = st.std()
                thresh = st.mean - args.kappa * (std if math.isfinite(std) else 0.0)
                if thresh <= 0.0:
                    veto = 1
                    veto_reason = "mean_minus_kstd<0"
            else:
                alpha = float(args.cvar_alpha)
                values = buf.sample()
                if values.size:
                    k = max(0, int(math.floor(alpha * values.size)) - 1)
                    bucket_q = float(np.partition(values, k)[k]) if values.size > 1 else float(values[0])
                    tail = values[values <= bucket_q]
                    bucket_cvar = float(np.mean(tail)) if tail.size else float("nan")
                if math.isfinite(bucket_cvar) and bucket_cvar < -float(args.epsilon):
                    veto = 1
                    veto_reason = "cvar"

        if dir_pred == 0 or veto == 1:
            would_act = "HOLD"
            action = 0
            is_holding = True
            if veto == 1 and veto_reason not in {"insufficient_samples", "flat"}:
                cooldown = max(cooldown, int(args.veto_cooldown))
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
                "bucket_n": int(st.n),
                "bucket_q": bucket_q,
                "bucket_cvar": bucket_cvar,
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    args.proposal_log.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.proposal_log, index=False)
    print(f"Wrote {args.proposal_log} ({len(df)} rows)")


if __name__ == "__main__":
    main()

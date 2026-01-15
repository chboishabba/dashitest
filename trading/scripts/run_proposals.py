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
from typing import Optional

import numpy as np
import pandas as pd

from options.chain import (
    OptionChain,
    RealOptionChain,
    SyntheticOptionChain,
    SyntheticOptionsConfig,
)
from utils.weights_config import (
    parse_simple_yaml,
    get_path,
    resolve_value,
    format_resolved,
    validate_never_learnable,
    validate_known_paths,
    clamp_learnable_vector,
)

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


def _parse_csv_numbers(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [float(p) for p in parts]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(float(p)) for p in _parse_csv_numbers(raw)]


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
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.zeros_like(arr, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not math.isfinite(std) or std <= 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(float(p)) for p in _parse_csv_numbers(raw)]


def _parse_csv_strings(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _softmax_vec(scores: np.ndarray, temp: float) -> np.ndarray:
    t = max(1e-6, float(temp))
    x = scores / t
    x = x - np.max(x)
    expx = np.exp(x)
    return expx / np.clip(expx.sum(), 1e-12, None)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _close(delta: float, scale: float, eps: float = 1e-12) -> float:
    denom = np.float32(scale) + np.float32(eps)
    return float(np.exp(-np.abs(np.float32(delta)) / denom))


def _shock(delta: float, scale: float, eps: float = 1e-12) -> float:
    denom = np.float32(scale) + np.float32(eps)
    return float(np.exp(-np.maximum(np.float32(delta), np.float32(0.0)) / denom))


def _median_abs_delta(arr: np.ndarray, floor: float = 1e-6) -> float:
    diffs = np.diff(arr, prepend=arr[0]).astype(np.float32, copy=False)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float(floor)
    med = float(np.nanmedian(np.abs(diffs)))
    return max(med, float(floor))


def _clamp01(val: float) -> float:
    return float(min(max(val, 0.0), 1.0))


def _ontology_support(
    qrow: np.ndarray,
    *,
    means: np.ndarray,
    stds: np.ndarray,
    temp: float,
) -> tuple[np.ndarray, str, float]:
    z = (qrow[:6] - means) / np.clip(stds, 1e-6, None)
    v, c, d, b, a, r = z
    A = (qrow[4] + 1.0) * 0.5
    V = float(_sigmoid(v))
    C = float(_sigmoid(c))
    D = float(_sigmoid(d))
    B = float(_sigmoid(b))
    R = float(_sigmoid(r))

    u_t = 1.2 * A + 0.8 * R - 1.0 * B - 0.6 * D
    u_r = 1.0 * C + 0.6 * V + 0.6 * (1.0 - A) - 1.0 * B - 0.6 * D
    u_h = 1.4 * B + 1.0 * D + 0.4 * (1.0 - A)
    probs = _softmax_vec(np.array([u_t, u_r, u_h], dtype=float), temp)
    idx = int(np.argmax(probs))
    ont = ("T", "R", "H")[idx]
    return probs, ont, float(probs[idx])


class OntologyGate:
    def __init__(
        self,
        *,
        theta_on: float,
        theta_off: float,
        ton: int,
        toff: int,
        margin: float,
        start: str = "H",
    ) -> None:
        self.k = start
        self.streak_on = 0
        self.streak_off = 0
        self.theta_on = float(theta_on)
        self.theta_off = float(theta_off)
        self.ton = int(ton)
        self.toff = int(toff)
        self.margin = float(margin)

    def step(self, probs: np.ndarray, action_state: str) -> tuple[str, bool]:
        p = {"T": float(probs[0]), "R": float(probs[1]), "H": float(probs[2])}
        best = max(p, key=p.get)
        cur = self.k
        switched = False

        if action_state != "HOLD":
            return cur, switched

        if best == cur:
            self.streak_on = 0
            self.streak_off = 0
            return cur, switched

        if p[best] >= self.theta_on and (p[best] - p[cur]) >= self.margin:
            self.streak_on += 1
        else:
            self.streak_on = 0

        if p[cur] <= self.theta_off:
            self.streak_off += 1
        else:
            self.streak_off = 0

        if self.streak_on >= self.ton:
            self.k = best
            self.streak_on = 0
            self.streak_off = 0
            switched = True

        return self.k, switched


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate direction proposals and veto decisions.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--prices-csv", type=Path, required=True, help="Prices CSV for bars.")
    ap.add_argument("--dir-model", type=Path, required=True, help="Direction head JSON.")
    ap.add_argument("--inst-model", type=Path, default=None, help="Instrument head JSON (optional).")
    ap.add_argument("--use-inst-model", action="store_true", help="Use instrument head model output.")
    ap.add_argument("--proposal-log", type=Path, required=True, help="Output proposal CSV.")
    ap.add_argument("--meta-features", type=Path, default=None, help="Optional market meta feature CSV.")
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--tau-on", type=float, default=None, help="ell hysteresis on.")
    ap.add_argument("--tau-off", type=float, default=None, help="ell hysteresis off.")
    ap.add_argument("--weights", type=Path, default=Path("weights.yaml"), help="Optional weights.yaml")
    ap.add_argument("--ell-bins", type=int, default=10, help="Number of ell bins for veto stats.")
    ap.add_argument("--veto-mode", type=str, default="cvar", choices=["meanstd", "cvar"])
    ap.add_argument("--kappa", type=float, default=0.0, help="Mean/std veto: mean - kappa*std <= 0.")
    ap.add_argument("--cvar-alpha", type=float, default=None, help="CVaR tail alpha.")
    ap.add_argument("--epsilon", type=float, default=0.0, help="CVaR veto threshold.")
    ap.add_argument("--veto-min-samples", type=int, default=None, help="Min samples before veto activates.")
    ap.add_argument("--veto-buffer", type=int, default=None, help="Per-bucket buffer size.")
    ap.add_argument("--veto-cooldown", type=int, default=None, help="Cooldown bars after veto.")
    ap.add_argument("--hazard-veto", action="store_true", help="Enable hazard veto.")
    ap.add_argument("--hazard-threshold", type=float, default=None, help="Hazard veto threshold.")
    ap.add_argument("--hazard-a", type=float, default=1.0, help="Hazard weight for burstiness.")
    ap.add_argument("--hazard-b", type=float, default=1.0, help="Hazard weight for curvature.")
    ap.add_argument("--hazard-c", type=float, default=1.0, help="Hazard weight for vol_ratio.")
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
    ap.add_argument("--size-temp", type=float, default=1.0, help="Size score temperature.")
    ap.add_argument("--pnl-weight-k", type=float, default=0.5, help="PnL weight strength for size teacher.")
    ap.add_argument("--pnl-weight-min", type=float, default=None, help="Min PnL weight.")
    ap.add_argument("--pnl-weight-max", type=float, default=None, help="Max PnL weight.")
    ap.add_argument("--pnl-weight-eps", type=float, default=1e-6, help="PnL weight epsilon.")
    ap.add_argument("--score-margin-min", type=float, default=None, help="Min score margin to accept proposal.")
    ap.add_argument("--opt-temp", type=float, default=1.0, help="Option head temperature.")
    ap.add_argument("--opt-type-w-imb", type=float, default=0.4, help="Option type weight for put/call imbalance.")
    ap.add_argument("--opt-tenor-w-hazard", type=float, default=0.8, help="Option tenor weight for hazard.")
    ap.add_argument("--opt-tenor-w-iv", type=float, default=0.6, help="Option tenor weight for IV.")
    ap.add_argument("--opt-mny-w-hazard", type=float, default=0.8, help="Option mny weight for hazard.")
    ap.add_argument("--opt-mny-w-iv", type=float, default=0.6, help="Option mny weight for IV.")
    ap.add_argument("--horizon", type=int, default=None, help="Override horizon for delta_pnl.")
    ap.add_argument("--options-source", type=str, choices=["none", "synthetic", "real"], default="none", help="Option candidate source.")
    ap.add_argument("--options-db", type=Path, default=None, help="Parquet file for real option surface.")
    ap.add_argument("--options-real-fallback", action="store_true", help="Fall back to synthetic when real data missing.")
    ap.add_argument("--options-sigma0", type=float, default=0.60, help="Synthetic options base IV.")
    ap.add_argument("--options-sigma-min", type=float, default=0.20, help="Synthetic options min IV.")
    ap.add_argument("--options-sigma-max", type=float, default=1.50, help="Synthetic options max IV.")
    ap.add_argument("--options-k-v", type=float, default=0.50, help="Synthetic options vol sensitivity.")
    ap.add_argument("--options-k-b", type=float, default=0.10, help="Synthetic options burst sensitivity.")
    ap.add_argument("--options-tenor-days", type=str, default="2,5,14,41,120", help="Comma-separated tenor days.")
    ap.add_argument("--options-tenor-bins", type=str, default="e_1_3,e_4_7,e_8_21,e_22_60,e_61_180", help="Canonical tenor labels.")
    ap.add_argument("--options-mny-bins", type=str, default="m_deep_itm,m_itm,m_atm,m_otm,m_deep_otm", help="Canonical moneyness labels.")
    ap.add_argument("--options-mny-deltas", type=str, default="-0.2,-0.1,0.0,0.1,0.2", help="Moneyness deltas for synthetic options.")
    ap.add_argument("--options-mny-cutoffs", type=str, default="-0.15,-0.05,0.05,0.15", help="Cutoffs used for real options binning.")
    ap.add_argument("--options-max-staleness-ms", type=int, default=1800000, help="Max age (ms) for real surface snapshots.")
    ap.add_argument("--options-min-candidates", type=int, default=20, help="Min option candidates required when using real surface.")
    ap.add_argument("--min-run-length", type=int, default=3, help="RegimeSpec min_run_length.")
    ap.add_argument("--max-flip-rate", type=float, default=None, help="RegimeSpec max_flip_rate.")
    ap.add_argument("--max-vol", type=float, default=None, help="RegimeSpec max_vol.")
    ap.add_argument("--window", type=int, default=50, help="RegimeSpec window.")
    ap.add_argument("--ont-temp", type=float, default=0.8, help="Ontology softmax temperature.")
    ap.add_argument("--ont-theta-on", type=float, default=0.60, help="Ontology gate theta on.")
    ap.add_argument("--ont-theta-off", type=float, default=0.45, help="Ontology gate theta off.")
    ap.add_argument("--ont-ton", type=int, default=8, help="Ontology gate on persistence.")
    ap.add_argument("--ont-toff", type=int, default=8, help="Ontology gate off persistence.")
    ap.add_argument("--ont-margin", type=float, default=0.08, help="Ontology gate margin.")
    ap.add_argument(
        "--use-ontology-legitimacy",
        action="store_true",
        help="Use ontology-modulated ell_eff for ACT/HOLD gating.",
    )
    ap.add_argument("--use-learned-weights", type=Path, default=None, help="Per-ontology weights JSON.")
    ap.add_argument("--use-score-weights", action="store_true", help="Apply per-ontology score_weights.")
    ap.add_argument("--score-weight-beta", type=float, default=0.1, help="Score weight scale (beta).")
    ap.add_argument("--score-weight-clip", type=float, default=0.05, help="Score weight delta clip.")
    ap.add_argument("--dump-schema", action="store_true", help="Print resolved weight sources.")
    ap.add_argument("--size-probe", action="store_true", help="Compute diagnostic size_probe_bin.")
    ap.add_argument("--size-probe-scale", type=float, default=100.0, help="Scale applied to |delta_pnl| when probing size.")
    args = ap.parse_args()

    weights = {}
    if args.weights is not None and args.weights.exists():
        weights = parse_simple_yaml(args.weights)
        validate_known_paths(weights)
        validate_never_learnable(weights)

    # Apply precedence: defaults -> YAML -> explicit CLI
    resolved: dict[str, tuple[object, str]] = {}
    args.tau_on = resolve_value(args.tau_on, weights, ["thresholds", "tau_on"], 0.5, "thresholds.tau_on", resolved)
    args.tau_off = resolve_value(args.tau_off, weights, ["thresholds", "tau_off"], 0.49, "thresholds.tau_off", resolved)
    args.score_margin_min = resolve_value(
        args.score_margin_min,
        weights,
        ["competition", "margin_gate", "score_margin_min"],
        0.0,
        "competition.margin_gate.score_margin_min",
        resolved,
    )
    args.hazard_threshold = resolve_value(
        args.hazard_threshold, weights, ["veto", "hazard_threshold"], 2.0, "veto.hazard_threshold", resolved
    )
    args.cvar_alpha = resolve_value(args.cvar_alpha, weights, ["veto", "alpha"], 0.10, "veto.alpha", resolved)
    args.veto_min_samples = int(
        resolve_value(args.veto_min_samples, weights, ["veto", "min_samples"], 50, "veto.min_samples", resolved)
    )
    args.veto_buffer = int(
        resolve_value(args.veto_buffer, weights, ["veto", "buffer"], 1024, "veto.buffer", resolved)
    )
    args.veto_cooldown = int(
        resolve_value(args.veto_cooldown, weights, ["veto", "cooldown"], 25, "veto.cooldown", resolved)
    )
    args.pnl_weight_min = resolve_value(
        args.pnl_weight_min, weights, ["pnl", "reweight_min"], 0.5, "pnl.reweight_min", resolved
    )
    args.pnl_weight_max = resolve_value(
        args.pnl_weight_max, weights, ["pnl", "reweight_max"], 2.0, "pnl.reweight_max", resolved
    )

    if args.dump_schema:
        dump = format_resolved(resolved)
        if dump:
            print(dump)

    if args.tau_on < args.tau_off:
        raise SystemExit("Require tau-on >= tau-off for hysteresis.")

    tenor_bins = tuple(_parse_csv_strings(args.options_tenor_bins))
    tenor_days = tuple(_parse_csv_ints(args.options_tenor_days))
    mny_bins = tuple(_parse_csv_strings(args.options_mny_bins))
    mny_deltas = tuple(_parse_csv_numbers(args.options_mny_deltas))
    mny_cutoffs = tuple(_parse_csv_numbers(args.options_mny_cutoffs))
    if len(tenor_bins) != len(tenor_days):
        raise SystemExit("tenor_bins and tenor_days must have equal length.")
    if len(mny_bins) != len(mny_deltas):
        raise SystemExit("mny_bins and mny_deltas must have equal length.")
    if len(mny_cutoffs) != len(mny_bins) - 1:
        raise SystemExit("mny_cutoffs length must equal len(mny_bins)-1.")
    synthetic_config = SyntheticOptionsConfig(
        sigma0=float(args.options_sigma0),
        sigma_min=float(args.options_sigma_min),
        sigma_max=float(args.options_sigma_max),
        k_v=float(args.options_k_v),
        k_b=float(args.options_k_b),
        tenor_bins=tenor_bins,
        tenor_days=tenor_days,
        mny_bins=mny_bins,
        mny_deltas=mny_deltas,
    )
    option_chain: Optional[OptionChain] = None
    if args.options_source != "none":
        synthetic_chain = SyntheticOptionChain(synthetic_config)
        real_chain = None
        if args.options_source == "real":
            if args.options_db is None:
                raise SystemExit("--options-db is required when --options-source real")
            real_chain = RealOptionChain(
                args.options_db,
                stale_ms=float(args.options_max_staleness_ms),
                tenor_bins=tenor_bins,
                tenor_days=tenor_days,
                mny_bins=mny_bins,
                mny_cutoffs=mny_cutoffs,
            )
        option_chain = OptionChain(
            synthetic=synthetic_chain,
            real=real_chain,
            fallback_to_synthetic=args.options_real_fallback,
        )

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
        opt_oi = meta_map.get("opt_open_interest_sum")
        opt_vol = meta_map.get("opt_volume_sum")
        opt_count = meta_map.get("opt_count")
        opt_call = meta_map.get("opt_call_count")
        opt_put = meta_map.get("opt_put_count")
        iv_chg = np.diff(iv, prepend=iv[0]) if iv is not None else None
        oi_chg = np.diff(oi_val, prepend=oi_val[0]) if oi_val is not None else None
        z_iv = _zscore(iv) if iv is not None else None
        z_iv_chg = _zscore(iv_chg) if iv_chg is not None else None
        z_basis = _zscore(np.abs(basis)) if basis is not None else None
        z_funding_abs = _zscore(np.abs(funding)) if funding is not None else None
        z_funding_signed = _zscore(funding) if funding is not None else None
        z_oi_chg = _zscore(oi_chg) if oi_chg is not None else None
        z_oi_val = _zscore(oi_val) if oi_val is not None else None
        z_opt_oi = _zscore(opt_oi) if opt_oi is not None else None
        z_opt_vol = _zscore(opt_vol) if opt_vol is not None else None
        if opt_count is not None and opt_call is not None and opt_put is not None:
            denom = np.where(opt_count != 0, opt_count, 1.0)
            opt_imb = (opt_put - opt_call) / denom
            z_opt_imb = _zscore(opt_imb)
        else:
            z_opt_imb = None
    else:
        iv = iv_chg = basis = funding = oi_chg = None
        opt_oi = opt_vol = opt_count = opt_call = opt_put = None
        z_iv = z_iv_chg = z_basis = z_funding_abs = z_funding_signed = z_oi_chg = None
        z_opt_oi = z_opt_vol = z_opt_imb = None
        z_oi_val = None

    model = json.loads(args.dir_model.read_text())
    order = int(model["order"])
    classes = model["classes"]
    W = np.array(model["weights"], dtype=np.float32)
    b = np.array(model["bias"], dtype=np.float32)
    mean = np.array(model["mean"], dtype=np.float32)
    std = np.array(model["std"], dtype=np.float32)
    horizon = int(args.horizon) if args.horizon is not None else int(model.get("horizon", 10))

    inst_model = None
    inst_order = None
    inst_W = inst_b = inst_mean = inst_std = None
    inst_meta_cols = None
    if args.inst_model is not None:
        inst_model = json.loads(args.inst_model.read_text())
        inst_order = int(inst_model.get("order", 1))
        inst_W = np.array(inst_model["weights"], dtype=np.float32)
        inst_b = np.array(inst_model["bias"], dtype=np.float32)
        inst_mean = np.array(inst_model["mean"], dtype=np.float32)
        inst_std = np.array(inst_model["std"], dtype=np.float32)
        inst_meta_cols = inst_model.get("feature_meta", [])
        if meta_values is None:
            raise SystemExit("--inst-model requires --meta-features for input alignment")
        if inst_meta_cols:
            missing = [c for c in inst_meta_cols if c not in meta_cols]
            if missing:
                raise SystemExit(f"meta-features missing columns required by inst model: {missing}")

    qfeat = tape.mm[args.series, :, :6].astype(np.float32, copy=False)
    ell = tape.mm[args.series, :, 6].astype(np.float32, copy=False)
    alpha = float(0.7)
    a0 = float(0.1)
    s_dv = _median_abs_delta(qfeat[:, 0])
    s_dc = _median_abs_delta(qfeat[:, 1])
    s_dd = _median_abs_delta(qfeat[:, 2])
    s_db = _median_abs_delta(qfeat[:, 3])
    s_da = _median_abs_delta(qfeat[:, 4])
    s_dr = _median_abs_delta(qfeat[:, 5])
    delta_v = np.diff(qfeat[:, 0], prepend=qfeat[0, 0]).astype(np.float32, copy=False)
    delta_c = np.diff(qfeat[:, 1], prepend=qfeat[0, 1]).astype(np.float32, copy=False)
    delta_d = np.diff(qfeat[:, 2], prepend=qfeat[0, 2]).astype(np.float32, copy=False)
    delta_b = np.diff(qfeat[:, 3], prepend=qfeat[0, 3]).astype(np.float32, copy=False)
    delta_a = np.diff(qfeat[:, 4], prepend=qfeat[0, 4]).astype(np.float32, copy=False)
    delta_r = np.diff(qfeat[:, 5], prepend=qfeat[0, 5]).astype(np.float32, copy=False)
    scales_payload = {
        "s_dv": s_dv,
        "s_dc": s_dc,
        "s_dd": s_dd,
        "s_db": s_db,
        "s_da": s_da,
        "s_dr": s_dr,
    }
    scales_path = args.proposal_log.with_suffix(".ontology_scales.json")
    scales_path.parent.mkdir(parents=True, exist_ok=True)
    scales_path.write_text(json.dumps(scales_payload, indent=2))
    hazard_array = (
        args.hazard_a * qfeat[:, 3] + args.hazard_b * qfeat[:, 1] + args.hazard_c * qfeat[:, 0]
    )
    z_hazard = _zscore(hazard_array)
    z_trend = _zscore(np.abs(qfeat[:, 4]))
    z_ell_margin = _zscore(ell - args.tau_on)

    log_price = np.log(np.clip(price, 1e-12, None))
    future_ret = np.full(price.shape[0], np.nan, dtype=float)
    if price.size > horizon:
        future_ret[: -horizon] = log_price[horizon:] - log_price[:-horizon]

    log_returns = np.zeros_like(price)
    if price.size > 1:
        log_returns[1:] = log_price[1:] - log_price[:-1]
    s_r = float(max(np.nanmedian(np.abs(log_returns[1:])), 1e-6)) if price.size > 1 else 1e-6
    risk_ref = float(np.median(np.clip(np.maximum(hazard_array, 0.0), 0.0, None)))
    risk_ref = max(risk_ref, 1e-6)

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
    persist_ema = 0.0
    persist_alpha = 2.0 / (8.0 + 1.0)

    windows = np.lib.stride_tricks.sliding_window_view(qfeat, window_shape=order, axis=0)
    feat_mat = windows.reshape(-1, order * qfeat.shape[1])
    inst_feat_mat = None
    if inst_model is not None:
        windows_inst = np.lib.stride_tricks.sliding_window_view(qfeat, window_shape=inst_order, axis=0)
        inst_feat_mat = windows_inst.reshape(-1, inst_order * qfeat.shape[1])

    qfeat_stats = np.nan_to_num(qfeat[:, :6], nan=0.0, posinf=0.0, neginf=0.0)
    qfeat_means = np.nanmean(qfeat_stats, axis=0)
    qfeat_stds = np.nanstd(qfeat_stats, axis=0)
    ontology_gate = OntologyGate(
        theta_on=args.ont_theta_on,
        theta_off=args.ont_theta_off,
        ton=args.ont_ton,
        toff=args.ont_toff,
        margin=args.ont_margin,
    )
    default_weights = {
        "T": {
            "score_weights": np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            "instrument_weights": np.array([1.0, 1.0, 0.5], dtype=np.float32),
            "opt_tenor_weights": np.array([0.5, 0.8, 1.0, 0.6, 0.3], dtype=np.float32),
            "opt_mny_weights": np.array([0.3, 0.6, 1.0, 0.6, 0.2], dtype=np.float32),
            "size_weights": np.array([1, 1, 1, 1], dtype=np.float32),
        },
        "R": {
            "score_weights": np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            "instrument_weights": np.array([0.8, 0.8, 1.0], dtype=np.float32),
            "opt_tenor_weights": np.array([0.3, 0.6, 1.0, 0.8, 0.4], dtype=np.float32),
            "opt_mny_weights": np.array([0.2, 0.5, 1.0, 0.7, 0.3], dtype=np.float32),
            "size_weights": np.array([1, 1, 1, 1], dtype=np.float32),
        },
        "H": {
            "score_weights": np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            "instrument_weights": np.array([0.2, 0.2, 0.1], dtype=np.float32),
            "opt_tenor_weights": np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
            "opt_mny_weights": np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
            "size_weights": np.array([1, 1, 1, 1], dtype=np.float32),
        },
    }
    learned_weights = None
    if args.use_learned_weights is not None and args.use_learned_weights.exists():
        payload = json.loads(args.use_learned_weights.read_text())
        learned_weights = payload.get("weights", payload)

    def _weight_vec(ont: str, name: str, size: int) -> np.ndarray:
        base = default_weights.get(ont, default_weights["H"]).get(name)
        if base is None or base.size != size:
            base = np.ones(size, dtype=np.float32)
        vec = base.astype(np.float32, copy=True)
        if learned_weights is not None and ont in learned_weights:
            raw = learned_weights[ont].get(name)
            if raw is not None and len(raw) == size:
                vec = np.array(raw, dtype=np.float32)
        vec = np.array(clamp_learnable_vector(vec.tolist(), name), dtype=np.float32)
        total = float(np.sum(vec))
        if total > 0 and math.isfinite(total):
            vec = vec / total
        return vec

    def _apply_weights(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if probs.size != weights.size:
            return probs
        weighted = probs * weights
        total = float(np.sum(weighted))
        if total <= 0 or not math.isfinite(total):
            return probs
        return weighted / total

    def _score_delta(w_score: np.ndarray, features: np.ndarray, beta: float, clip: float) -> float:
        if w_score.size != features.size:
            return 0.0
        delta = float(beta) * float(np.dot(w_score, features))
        if not math.isfinite(delta):
            return 0.0
        return float(np.clip(delta, -abs(clip), abs(clip)))

    stats: dict[tuple[int, int, str], RunningStats] = {}
    buffers: dict[tuple[int, int, str], BucketBuffer] = {}
    rows = []
    is_holding = False
    cooldown = 0

    for t in range(price.size):
        qrow = qfeat[t]
        option_candidates = []
        if option_chain is not None:
            option_candidates = option_chain.generate(
                t,
                float(price[t]),
                qrow,
                int(ts_int[t]),
            )
            if args.options_source == "real" and len(option_candidates) < int(args.options_min_candidates):
                raise SystemExit(
                    f"Not enough real option candidates at t={t}: {len(option_candidates)} < {args.options_min_candidates}"
                )
        option_count = len(option_candidates)
        option_source = option_candidates[0].source if option_candidates else "none"
        ont_probs, ont_raw, ont_conf = _ontology_support(
            qrow,
            means=qfeat_means,
            stds=qfeat_stds,
            temp=args.ont_temp,
        )
        action_state = "HOLD" if is_holding else "ACT"
        ontology_k, ontology_switched = ontology_gate.step(ont_probs, action_state)
        w_score = _weight_vec(ontology_k, "score_weights", 6)
        w_inst = _weight_vec(ontology_k, "instrument_weights", 3)
        w_size = _weight_vec(ontology_k, "size_weights", 4)
        w_tenor = _weight_vec(ontology_k, "opt_tenor_weights", 5)
        w_mny = _weight_vec(ontology_k, "opt_mny_weights", 5)
        hazard = float(hazard_array[t])
        risk_value = max(0.0, float(hazard))
        delta_pnl = float(future_ret[t])
        ell_t = (
            _close(delta_a[t], s_da)
            * _close(delta_r[t], s_dr)
            * _close(delta_v[t], s_dv)
            * _shock(delta_b[t], s_db)
            * _shock(delta_d[t], s_dd)
            * _shock(a0 - float(qrow[4]), s_da)
        )
        ell_r = (
            _close(delta_c[t], s_dc)
            * _close(delta_v[t], s_dv)
            * _close(delta_r[t], s_dr)
            * _shock(delta_b[t], s_db)
            * _shock(delta_d[t], s_dd)
            * _shock(float(qrow[4]), s_da)
        )
        ell_h = (
            _close(delta_b[t], s_db)
            * _close(delta_d[t], s_dd)
            * _shock(delta_b[t], s_db)
            * _shock(delta_d[t], s_dd)
            * _close(delta_v[t], s_dv)
        )
        if ontology_k == "T":
            ell_k = ell_t
        elif ontology_k == "R":
            ell_k = ell_r
        else:
            ell_k = ell_h
        ell_eff = float(ell[t]) * (alpha + (1.0 - alpha) * float(ell_k))
        ell_base = float(ell[t])
        ell_for_gate = float(ell_eff) if args.use_ontology_legitimacy else ell_base
        row = {
            "i": int(t),
            "ts": int(ts_int[t]),
            "price": float(price[t]),
            "state": int(state[t]),
            "ell": ell_base,
            "ell_T": float(ell_t),
            "ell_R": float(ell_r),
            "ell_H": float(ell_h),
            "ell_k": float(ell_k),
            "ell_eff": float(ell_eff),
            "alpha": float(alpha),
            "s_dv": float(s_dv),
            "s_dc": float(s_dc),
            "s_dd": float(s_dd),
            "s_db": float(s_db),
            "s_da": float(s_da),
            "s_dr": float(s_dr),
            "actionability": ell_base,
            "margin": float(margin[t]) if math.isfinite(margin[t]) else float("nan"),
            "acceptable": bool(acceptable[t]),
            "hazard": float(hazard),
            "ontology_raw": ont_raw,
            "ontology_k": ontology_k,
            "ontology_switched": int(ontology_switched),
            "ontology_confidence": ont_conf,
            "option_count": option_count,
            "option_source": option_source,
            "p_ont_t": float(ont_probs[0]),
            "p_ont_r": float(ont_probs[1]),
            "p_ont_h": float(ont_probs[2]),
        }
        if meta_values is not None:
            for idx, col in enumerate(meta_cols):
                row[col] = float(meta_values[t, idx]) if math.isfinite(meta_values[t, idx]) else float("nan")

        gate_open = True
        if not is_holding:
            if ell_for_gate < args.tau_on:
                gate_open = False
        else:
            if ell_for_gate < args.tau_off:
                gate_open = False
        if ell_for_gate <= 0.0 or not math.isfinite(ell_for_gate):
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
                    "p_spot_teacher": float("nan"),
                    "p_perp_teacher": float("nan"),
                    "p_option_teacher": float("nan"),
                    "inst_source": "none",
                    "inst_confidence": float("nan"),
                    "opt_type_pred": "none",
                    "opt_tenor_pred": "none",
                    "opt_mny_pred": "none",
                    "p_opt_call": float("nan"),
                    "p_opt_put": float("nan"),
                    "p_opt_e_1_3": float("nan"),
                    "p_opt_e_4_7": float("nan"),
                    "p_opt_e_8_21": float("nan"),
                    "p_opt_e_22_60": float("nan"),
                    "p_opt_e_61_180": float("nan"),
                    "p_opt_m_deep_itm": float("nan"),
                    "p_opt_m_itm": float("nan"),
                    "p_opt_m_atm": float("nan"),
                    "p_opt_m_otm": float("nan"),
                    "p_opt_m_deep_otm": float("nan"),
                    "size_pred": "0",
                    "size_pred_v2": "0",
                    "size_probe_bin": "0",
                    "size_pred_v2": "0",
                    "size_probe_bin": "0",
                    "p_size_0": float("nan"),
                    "p_size_0_5": float("nan"),
                    "p_size_1": float("nan"),
                    "p_size_2": float("nan"),
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
                    "p_spot_teacher": float("nan"),
                    "p_perp_teacher": float("nan"),
                    "p_option_teacher": float("nan"),
                    "inst_source": "none",
                    "inst_confidence": float("nan"),
                    "opt_type_pred": "none",
                    "opt_tenor_pred": "none",
                    "opt_mny_pred": "none",
                    "p_opt_call": float("nan"),
                    "p_opt_put": float("nan"),
                    "p_opt_e_1_3": float("nan"),
                    "p_opt_e_4_7": float("nan"),
                    "p_opt_e_8_21": float("nan"),
                    "p_opt_e_22_60": float("nan"),
                    "p_opt_e_61_180": float("nan"),
                    "p_opt_m_deep_itm": float("nan"),
                    "p_opt_m_itm": float("nan"),
                    "p_opt_m_atm": float("nan"),
                    "p_opt_m_otm": float("nan"),
                    "p_opt_m_deep_otm": float("nan"),
                    "size_pred": "0",
                    "p_size_0": float("nan"),
                    "p_size_0_5": float("nan"),
                    "p_size_1": float("nan"),
                    "p_size_2": float("nan"),
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
        ell_bin = _ell_bin(float(ell[t]), args.ell_bins)

        inst_scores = np.zeros(3, dtype=float)
        if meta_values is not None and dir_pred != 0:
            s_opt = 0.0
            s_opt += args.inst_w_iv * (z_iv[t] if z_iv is not None else 0.0)
            s_opt += args.inst_w_hazard * float(z_hazard[t])
            s_opt += args.inst_w_funding * (z_funding_abs[t] if z_funding_abs is not None else 0.0)
            s_opt += args.inst_w_basis * (z_basis[t] if z_basis is not None else 0.0)
            s_opt += args.inst_w_opt_oi * (z_opt_oi[t] if z_opt_oi is not None else 0.0)
            s_opt += args.inst_w_opt_vol * (z_opt_vol[t] if z_opt_vol is not None else 0.0)
            s_opt += args.inst_w_opt_imb * (z_opt_imb[t] if z_opt_imb is not None else 0.0)
            s_opt += args.inst_w_leg * float(ell[t] - args.tau_on)

            fund_carry_z = 0.0
            if z_funding_signed is not None:
                fund_carry_z = -float(z_funding_signed[t]) * float(dir_pred)

            s_perp = 0.0
            s_perp += args.perp_w_dirconf * float(max(probs[classes.index(1)], probs[classes.index(-1)]))
            s_perp += args.perp_w_carry * fund_carry_z
            s_perp += args.perp_w_oi * (z_oi_val[t] if z_oi_val is not None else 0.0)
            s_perp -= args.perp_w_iv * (z_iv[t] if z_iv is not None else 0.0)
            s_perp -= args.perp_w_hazard * float(z_hazard[t])
            s_perp += args.perp_w_leg * float(ell[t] - args.tau_on)

            s_spot = args.spot_base
            s_spot -= args.spot_w_funding * (z_funding_abs[t] if z_funding_abs is not None else 0.0)
            s_spot -= args.spot_w_basis * (z_basis[t] if z_basis is not None else 0.0)
            s_spot -= args.spot_w_hazard * float(z_hazard[t])
            inst_scores = np.array([s_spot, s_perp, s_opt], dtype=float)

        inst_labels = ["spot", "perp", "option"]
        inst_probs_teacher = _softmax_vec(inst_scores, args.inst_temp) if dir_pred != 0 else np.array([1.0, 0.0, 0.0])
        inst_probs = inst_probs_teacher
        inst_source = "teacher"
        if args.use_inst_model and inst_model is not None and inst_feat_mat is not None and dir_pred != 0 and t >= inst_order - 1:
            x_inst = inst_feat_mat[t - (inst_order - 1)]
            meta_row = meta_values[t]
            if inst_meta_cols:
                meta_map = {name: meta_row[idx] for idx, name in enumerate(meta_cols)}
                meta_vec = np.array([meta_map.get(c, 0.0) for c in inst_meta_cols], dtype=np.float32)
            else:
                meta_vec = meta_row.astype(np.float32, copy=False)
            x_inst = np.concatenate([x_inst.astype(np.float32, copy=False), meta_vec], axis=0)
            x_inst = (x_inst - inst_mean) / inst_std
            logits_inst = x_inst @ inst_W + inst_b
            inst_probs = _softmax_vec(logits_inst, 1.0)
            inst_source = "model"
        if args.use_learned_weights is not None and learned_weights is not None and dir_pred != 0:
            inst_probs = _apply_weights(inst_probs, w_inst)
        inst_choice = inst_labels[int(np.argmax(inst_probs))]
        dir_conf = float(max(probs[classes.index(1)], probs[classes.index(-1)])) if dir_pred != 0 else 0.0
        dir_trend = float(dir_conf) * (1.0 if dir_pred > 0 else -1.0) if dir_pred != 0 else 0.0
        persist_ema = persist_alpha * dir_trend + (1.0 - persist_alpha) * persist_ema
        persist = _clamp01((persist_ema + 1.0) * 0.5)
        impulse01 = _clamp01((abs(log_returns[t]) / (s_r + 1e-9)) / 3.0)
        z = _clamp01(0.65 * impulse01 + 0.35 * persist)
        risk01 = _clamp01(risk_value / (risk_ref + 1e-9))
        z_eff = _clamp01(z * (1.0 - 0.6 * risk01))
        if dir_pred == 0:
            size_pred_v2 = "0"
        elif z_eff < 0.25:
            size_pred_v2 = "0"
        elif z_eff < 0.75:
            size_pred_v2 = "0.5"
        else:
            size_pred_v2 = "1"
        size_centers = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
        size_labels = ["0", "0.5", "1", "2"]
        size_score = z_eff if dir_pred != 0 else 0.0
        size_logits = -np.abs(size_score - size_centers)
        size_probs = _softmax_vec(size_logits, args.size_temp) if dir_pred != 0 else np.array([1.0, 0.0, 0.0, 0.0])
        if args.use_learned_weights is not None and learned_weights is not None and dir_pred != 0:
            size_probs = _apply_weights(size_probs, w_size)
        size_pred = size_labels[int(np.argmax(size_probs))] if dir_pred != 0 else "0"
        size_probe_bin = "0"
        if args.size_probe and dir_pred != 0:
            probe_score = float(min(max(abs(delta_pnl) * args.size_probe_scale, 0.0), 2.0))
            probe_logits = -np.abs(probe_score - size_centers)
            probe_probs = _softmax_vec(probe_logits, args.size_temp)
            size_probe_bin = size_labels[int(np.argmax(probe_probs))]

        final_size_pred = size_pred_v2 if dir_pred != 0 else "0"

        inst_confidence = float(np.sort(inst_probs)[-1] - np.sort(inst_probs)[-2]) if dir_pred != 0 else float("nan")

        if dir_pred == 0:
            instrument_pred = "flat"
            exposure_pred = "flat"
            best_score = float("nan")
            second_score = float("nan")
            score_margin = float("nan")
            best_score_eff = float("nan")
            second_score_eff = float("nan")
            score_margin_eff = float("nan")
        else:
            cand_scores = []
            cand_meta = []
            cand_scores_eff = []
            p_dir = max(float(probs[classes.index(dir_pred)]), 1e-6)
            logp_dir = math.log(p_dir)
            ell_margin = max(0.0, float(ell[t] - args.tau_on))
            for i_s, inst in enumerate(inst_labels):
                for i_m, size in enumerate(size_labels):
                    score = 0.0
                    p_inst = max(float(inst_probs[i_s]), 1e-6)
                    p_size = max(float(size_probs[i_m]), 1e-6)
                    logp_inst = math.log(p_inst)
                    logp_size = math.log(p_size)
                    score += logp_dir + logp_inst + 2.0 * logp_size
                    score += 0.8 * ell_margin
                    score -= 0.7 * float(hazard)
                    carry = 0.0
                    if inst == "perp" and funding is not None and math.isfinite(float(funding[t])):
                        carry = 0.6 * (-float(funding[t]) * float(dir_pred))
                        score += carry
                    # tail penalty from current bucket if available
                    pen = 0.0
                    key_c = (dir_pred, ell_bin, inst)
                    if key_c in buffers:
                        vals = buffers[key_c].sample()
                        if vals.size:
                            k = max(0, int(math.floor(float(args.cvar_alpha) * vals.size)) - 1)
                            q = float(np.partition(vals, k)[k]) if vals.size > 1 else float(vals[0])
                            tail = vals[vals <= q]
                            cvar = float(np.mean(tail)) if tail.size else float("nan")
                            if math.isfinite(cvar):
                                pen = max(0.0, -cvar - float(args.epsilon))
                    score -= 1.5 * pen
                    cand_scores.append(score)
                    if args.use_score_weights:
                        features = np.array(
                            [logp_dir, logp_inst, logp_size, ell_margin, -float(hazard), carry],
                            dtype=np.float32,
                        )
                        delta = _score_delta(w_score, features, args.score_weight_beta, args.score_weight_clip)
                        cand_scores_eff.append(score + delta)
                    else:
                        cand_scores_eff.append(score)
                    cand_meta.append((inst, size))
            cand_scores = np.array(cand_scores, dtype=float)
            cand_scores_eff = np.array(cand_scores_eff, dtype=float)
            best_idx = int(np.argmax(cand_scores_eff))
            best_score = float(np.sort(cand_scores)[::-1][0]) if cand_scores.size else float("nan")
            sorted_scores = np.sort(cand_scores)[::-1]
            second_score = float(sorted_scores[1]) if sorted_scores.size > 1 else float("nan")
            score_margin = best_score - second_score if math.isfinite(second_score) else float("nan")
            sorted_scores_eff = np.sort(cand_scores_eff)[::-1]
            best_score_eff = float(sorted_scores_eff[0]) if sorted_scores_eff.size else float("nan")
            second_score_eff = float(sorted_scores_eff[1]) if sorted_scores_eff.size > 1 else float("nan")
            score_margin_eff = best_score_eff - second_score_eff if math.isfinite(second_score_eff) else float("nan")
            instrument_pred, size_pred = cand_meta[best_idx]

        opt_type_pred = "none"
        opt_tenor_pred = "none"
        opt_mny_pred = "none"
        opt_type_probs = np.array([float("nan"), float("nan")])
        opt_tenor_probs = np.array([float("nan")] * 5)
        opt_mny_probs = np.array([float("nan")] * 5)
        if instrument_pred == "option" and dir_pred != 0:
            opt_type_labels = ["call", "put"]
            opt_type_score = np.array([1.0, 1.0], dtype=float)
            opt_type_score[0] += 0.5 if dir_pred > 0 else -0.5
            opt_type_score[1] += 0.5 if dir_pred < 0 else -0.5
            if z_opt_imb is not None:
                opt_type_score[0] -= args.opt_type_w_imb * float(z_opt_imb[t])
                opt_type_score[1] += args.opt_type_w_imb * float(z_opt_imb[t])
            opt_type_probs = _softmax_vec(opt_type_score, args.opt_temp)
            opt_type_pred = opt_type_labels[int(np.argmax(opt_type_probs))]

            tenor_bins = np.array([2.0, 5.5, 14.5, 41.0, 120.0], dtype=float)
            tenor_labels = ["e_1_3", "e_4_7", "e_8_21", "e_22_60", "e_61_180"]
            tenor_scores = -0.02 * tenor_bins
            tenor_scores += args.opt_tenor_w_hazard * float(z_hazard[t])
            tenor_scores -= args.opt_tenor_w_iv * (z_iv[t] if z_iv is not None else 0.0)
            opt_tenor_probs = _softmax_vec(tenor_scores, args.opt_temp)
            if args.use_learned_weights is not None and learned_weights is not None:
                opt_tenor_probs = _apply_weights(opt_tenor_probs, w_tenor)
            opt_tenor_pred = tenor_labels[int(np.argmax(opt_tenor_probs))]

            opt_mny_labels = ["m_deep_itm", "m_itm", "m_atm", "m_otm", "m_deep_otm"]
            mny_scores = np.zeros(5, dtype=float)
            mny_scores[0] += args.opt_mny_w_hazard * float(z_hazard[t])
            mny_scores[1] += 0.2 - 0.6 * args.opt_mny_w_iv * (z_iv[t] if z_iv is not None else 0.0)
            mny_scores[2] += 0.1 - 0.4 * args.opt_mny_w_iv * (z_iv[t] if z_iv is not None else 0.0)
            mny_scores[3] += 0.2 - 0.6 * args.opt_mny_w_iv * (z_iv[t] if z_iv is not None else 0.0)
            mny_scores[4] += 0.05 + 0.2 * args.opt_mny_w_hazard * float(z_hazard[t])
            opt_mny_probs = _softmax_vec(mny_scores, args.opt_temp)
            if args.use_learned_weights is not None and learned_weights is not None:
                opt_mny_probs = _apply_weights(opt_mny_probs, w_mny)
            opt_mny_pred = opt_mny_labels[int(np.argmax(opt_mny_probs))]

        delta_signed = float(delta_pnl * dir_pred) if math.isfinite(delta_pnl) else float("nan")
        key = (dir_pred, ell_bin, instrument_pred)
        if key not in stats:
            stats[key] = RunningStats()
        if key not in buffers:
            buffers[key] = BucketBuffer.create(int(args.veto_buffer))

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
        if veto == 0 and math.isfinite(score_margin) and score_margin < args.score_margin_min:
            veto = 1
            veto_reason = "low_margin"

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
        if dir_pred != 0:
            stats[key].update(delta_signed)
            buffers[key].add(delta_signed)

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
                "instrument_pred": instrument_pred,
                "p_spot": float(inst_probs[0]),
                "p_perp": float(inst_probs[1]),
                "p_option": float(inst_probs[2]),
                "p_spot_teacher": float(inst_probs_teacher[0]),
                "p_perp_teacher": float(inst_probs_teacher[1]),
                "p_option_teacher": float(inst_probs_teacher[2]),
                "inst_source": inst_source,
                "inst_confidence": inst_confidence,
                "opt_type_pred": opt_type_pred,
                "opt_tenor_pred": opt_tenor_pred,
                "opt_mny_pred": opt_mny_pred,
                "p_opt_call": float(opt_type_probs[0]),
                "p_opt_put": float(opt_type_probs[1]),
                "p_opt_e_1_3": float(opt_tenor_probs[0]),
                "p_opt_e_4_7": float(opt_tenor_probs[1]),
                "p_opt_e_8_21": float(opt_tenor_probs[2]),
                "p_opt_e_22_60": float(opt_tenor_probs[3]),
                "p_opt_e_61_180": float(opt_tenor_probs[4]),
                "p_opt_m_deep_itm": float(opt_mny_probs[0]),
                "p_opt_m_itm": float(opt_mny_probs[1]),
                "p_opt_m_atm": float(opt_mny_probs[2]),
                "p_opt_m_otm": float(opt_mny_probs[3]),
                "p_opt_m_deep_otm": float(opt_mny_probs[4]),
                "size_pred": final_size_pred,
                "size_pred_v2": size_pred_v2,
                "size_probe_bin": size_probe_bin,
                "p_size_0": float(size_probs[0]),
                "p_size_0_5": float(size_probs[1]),
                "p_size_1": float(size_probs[2]),
                "p_size_2": float(size_probs[3]),
                "score_best": best_score,
                "score_second": second_score,
                "score_margin": score_margin,
                "score_best_eff": best_score_eff,
                "score_second_eff": second_score_eff,
                "score_margin_eff": score_margin_eff,
            }
        )
        if args.options_source == "real":
            required_opt_cols = [
                "opt_type_pred",
                "opt_tenor_pred",
                "opt_mny_pred",
                "p_opt_call",
                "p_opt_put",
                "p_opt_e_1_3",
                "p_opt_e_4_7",
                "p_opt_e_8_21",
                "p_opt_e_22_60",
                "p_opt_e_61_180",
                "p_opt_m_deep_itm",
                "p_opt_m_itm",
                "p_opt_m_atm",
                "p_opt_m_otm",
                "p_opt_m_deep_otm",
            ]
            missing = [col for col in required_opt_cols if col not in row]
            if missing:
                raise SystemExit(f"Missing option columns in proposals when using real surface: {missing}")
        rows.append(row)

    df = pd.DataFrame(rows)
    args.proposal_log.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.proposal_log, index=False)
    print(f"Wrote {args.proposal_log} ({len(df)} rows)")


if __name__ == "__main__":
    main()

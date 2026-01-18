#!/usr/bin/env python3
"""
Phase-07 eigen/boundary check (stub).

Formal intent:
- Operates on the action channel (decisions/exposure deltas) and observations (returns),
  producing a boundary/asymmetry certificate b_t and aggregate rho_A(horizon).
- Does not infer labels and does not alter the kernel K or quotient state. This is a
  certificate used by supervisory predicates (Phase-7/8/9).

Definitions (canonical, as per compactified context):
    edge_t = x_{t-1} * Delta m_t
    cost_t = (h_t + f_t + slip_t) * m_t * |Delta x_t|
    rho_A  = sum edge / (sum cost + eps)

Robustness:
- Evaluate rho_A across horizons (decimation / batching).
- Evaluate sensitivity under small perturbations of cost model parameters (+/- eps_cost).

Classification (coarse):
    - false: rho_A > 0 but fails robustness/persistence
    - boundary-unstable: rho_A > 0 but flips under small perturbations or horizon changes
    - boundary-stable: rho_A clears threshold, persists, and remains positive under perturbations

NOTE:
- This file is a stub: alignment/parsing is minimal, and some heuristics are placeholders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DecisionRec:
    ts: float  # seconds (unix) or monotone; only used for ordering
    x: float   # target exposure in [-1, +1] or [0, 1] with signed direction encoded separately
    d: int     # direction in {-1, 0, +1}
    m: int     # activity mask in {0, 1}
    symbol: str = ""
    phase6_open: Optional[bool] = None


@dataclass(frozen=True)
class ReturnRec:
    ts: float
    r: float
    symbol: str = ""


@dataclass(frozen=True)
class CostModel:
    half_spread: float
    fee: float
    slip: float
    eps: float = 1e-9


@dataclass(frozen=True)
class RhoResult:
    horizon_s: int
    rho_A: float
    sum_edge: float
    sum_cost: float
    n_support: int
    activity_rate: float
    robust_pass: bool
    class_label: str  # {false, boundary-unstable, boundary-stable}


def pi_supp(d: int) -> int:
    return 1 if d != 0 else 0


def _coerce_ts(value: object) -> float:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return 0.0
    # Live daemon emits ms timestamps; normalize to seconds.
    if ts > 1e11:
        ts /= 1000.0
    return ts


def _parse_phase6_gate(raw: object) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def compute_deltas(decisions: List[DecisionRec]) -> Tuple[List[float], List[int], List[float]]:
    """
    Returns:
        dx_t: |Delta x_t| aligned at t
        dm_t: Delta m_t aligned at t
        x_prev: x_{t-1} aligned at t
    """
    if len(decisions) < 2:
        return [], [], []

    dx: List[float] = []
    dm: List[int] = []
    x_prev: List[float] = []

    for i in range(1, len(decisions)):
        x0 = decisions[i - 1].x
        x1 = decisions[i].x
        m0 = decisions[i - 1].m
        m1 = decisions[i].m
        dx.append(abs(x1 - x0))
        dm.append(int(m1 - m0))
        x_prev.append(x0)
    return dx, dm, x_prev


def rho_A_for_stream(
    decisions: List[DecisionRec],
    returns: List[ReturnRec],
    cost: CostModel,
) -> Tuple[float, float, float, int, float]:
    """
    Placeholder alignment:
    - Assumes decisions and returns are already aligned by ts ordering and same symbol.
    - For the stub: uses decisions deltas and ignores returns in rho_A definition
      (edge = x_{t-1} * Delta m_t). Returns are unused beyond future extensions.

    Returns:
        rho_A, sum_edge, sum_cost, n_support, activity_rate
    """
    _ = returns
    dx, dm, x_prev = compute_deltas(decisions)
    if not dx:
        return 0.0, 0.0, 0.0, 0, 0.0

    sum_edge = 0.0
    sum_cost = 0.0
    n_support = 0

    for i in range(len(dx)):
        edge_t = x_prev[i] * float(dm[i])
        m_t = decisions[i + 1].m
        cost_t = (cost.half_spread + cost.fee + cost.slip) * float(m_t) * dx[i]

        sum_edge += edge_t
        sum_cost += cost_t
        n_support += int(m_t)

    rho = sum_edge / (sum_cost + cost.eps)
    activity_rate = n_support / max(1, len(dx))
    return rho, sum_edge, sum_cost, n_support, activity_rate


def decimate_by_horizon(decisions: List[DecisionRec], horizon_s: int) -> List[DecisionRec]:
    """
    Stub decimation: take one record every ~horizon_s seconds.
    Assumes ts is seconds. If ts isn't seconds, replace with index-based grouping.
    """
    if not decisions:
        return decisions
    out = [decisions[0]]
    last_t = decisions[0].ts
    for rec in decisions[1:]:
        if (rec.ts - last_t) >= horizon_s:
            out.append(rec)
            last_t = rec.ts
    return out


def robust_check(
    decisions: List[DecisionRec],
    returns: List[ReturnRec],
    base_cost: CostModel,
    rho_thresh: float,
    persist_min: int,
    eps_cost_frac: float,
) -> Tuple[bool, str]:
    """
    Robustness gates (stub):
    1) rho_A >= threshold
    2) stability under +/- eps_cost perturbations of (h, f, slip)
    3) persistence: require n_support >= persist_min (proxy for non-empty support)
    """
    rho0, _, _, n_support, _ = rho_A_for_stream(decisions, returns, base_cost)
    if n_support < persist_min:
        return False, "sparse_support"

    if rho0 < rho_thresh:
        return False, "below_thresh"

    def perturbed(sign: float) -> CostModel:
        factor = 1.0 + sign * eps_cost_frac
        return CostModel(
            half_spread=base_cost.half_spread * factor,
            fee=base_cost.fee * factor,
            slip=base_cost.slip * factor,
            eps=base_cost.eps,
        )

    rho_plus, *_ = rho_A_for_stream(decisions, returns, perturbed(+1.0))
    rho_minus, *_ = rho_A_for_stream(decisions, returns, perturbed(-1.0))

    if rho_plus < rho_thresh or rho_minus < rho_thresh:
        return False, "cost_sensitivity"

    return True, "ok"


def classify(rho: float, robust_pass: bool) -> str:
    if rho <= 0:
        return "false"
    if not robust_pass:
        return "boundary-unstable"
    return "boundary-stable"


def load_decisions_ndjson(path: Path, symbol: Optional[str]) -> List[DecisionRec]:
    out: List[DecisionRec] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sym = obj.get("symbol", "")
            if symbol and sym and sym != symbol:
                continue
            ts_raw = obj.get("timestamp", obj.get("ts_ms", obj.get("ts", 0.0)))
            ts = _coerce_ts(ts_raw)
            d = int(obj.get("direction", obj.get("d", obj.get("action", 0))))
            x = float(obj.get("target_exposure", obj.get("x", obj.get("target", 0.0))))
            if d != 0 and x >= 0.0:
                x = abs(x) * float(d)
            m = obj.get("m")
            if m is None:
                m = pi_supp(d)
            phase6_gate = _parse_phase6_gate(obj.get("phase6_gate"))
            phase6 = None
            if phase6_gate:
                phase6 = phase6_gate.get("open")
            if phase6 is None:
                phase6 = obj.get("phase6_open")
            out.append(DecisionRec(ts=ts, x=x, d=d, m=int(m), symbol=sym, phase6_open=phase6))
    out.sort(key=lambda r: r.ts)
    return out


def load_returns_ndjson(path: Path, symbol: Optional[str]) -> List[ReturnRec]:
    out: List[ReturnRec] = []
    last_price: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sym = obj.get("symbol", "")
            if symbol and sym and sym != symbol:
                continue
            ts_raw = obj.get("timestamp", obj.get("ts_ms", obj.get("ts", 0.0)))
            ts = _coerce_ts(ts_raw)
            r_val = None
            for key in ("r", "return", "ret", "price_ret"):
                if key in obj:
                    r_val = obj.get(key)
                    break
            if r_val is None and "delta_mid" in obj:
                r_val = obj.get("delta_mid")
            if r_val is None:
                price = None
                for key in ("close", "mid", "price"):
                    if key in obj:
                        price = obj.get(key)
                        break
                if price is not None:
                    try:
                        price_f = float(price)
                    except (TypeError, ValueError):
                        price_f = None
                    if price_f is not None:
                        prev = last_price.get(sym)
                        last_price[sym] = price_f
                        if prev and prev != 0:
                            r_val = (price_f / prev) - 1.0
            if r_val is None:
                continue
            try:
                r = float(r_val)
            except (TypeError, ValueError):
                continue
            out.append(ReturnRec(ts=ts, r=r, symbol=sym))
    out.sort(key=lambda r: r.ts)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase-07 eigen/boundary check (stub).")
    ap.add_argument("--decisions-ndjson", required=True, help="NDJSON decisions stream")
    ap.add_argument("--returns-ndjson", default="", help="Optional NDJSON returns stream")
    ap.add_argument("--symbol", default="", help="Filter symbol (optional)")

    ap.add_argument("--half-spread", type=float, default=0.0003)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0)

    ap.add_argument("--rho-thresh", type=float, default=1.0)
    ap.add_argument("--persist-min", type=int, default=25)
    ap.add_argument("--eps-cost-frac", type=float, default=0.05)

    ap.add_argument("--horizons", default="10,30,60,120,300")
    ap.add_argument("--out-json", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.strip() or None

    decisions = load_decisions_ndjson(Path(args.decisions_ndjson), symbol)
    returns: List[ReturnRec] = []
    if args.returns_ndjson:
        returns = load_returns_ndjson(Path(args.returns_ndjson), symbol)

    cost = CostModel(args.half_spread, args.fee, args.slip)
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]

    results: List[RhoResult] = []

    for h in horizons:
        dec_h = decimate_by_horizon(decisions, h)
        rho, sum_edge, sum_cost, n_support, activity_rate = rho_A_for_stream(dec_h, returns, cost)
        robust_pass, _reason = robust_check(
            dec_h,
            returns,
            cost,
            rho_thresh=args.rho_thresh,
            persist_min=args.persist_min,
            eps_cost_frac=args.eps_cost_frac,
        )
        label = classify(rho, robust_pass)

        results.append(
            RhoResult(
                horizon_s=h,
                rho_A=rho,
                sum_edge=sum_edge,
                sum_cost=sum_cost,
                n_support=n_support,
                activity_rate=activity_rate,
                robust_pass=robust_pass,
                class_label=label,
            )
        )

    payload = {
        "symbol": symbol or "",
        "cost_model": asdict(cost),
        "params": {
            "rho_thresh": args.rho_thresh,
            "persist_min": args.persist_min,
            "eps_cost_frac": args.eps_cost_frac,
            "horizons": horizons,
        },
        "results": [asdict(r) for r in results],
        "best": max(results, key=lambda r: r.rho_A, default=None) and asdict(max(results, key=lambda r: r.rho_A)),
    }

    output = json.dumps(payload, indent=2)
    print(output)
    if args.out_json:
        Path(args.out_json).write_text(output, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

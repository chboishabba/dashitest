#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _clean_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {key: _clean_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(val) for val in value]
    return value


def dump_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_clean_json(row), ensure_ascii=True, allow_nan=False))
            fh.write("\n")


def safe_get(d: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def coerce_ts_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        if x > 1e11:
            return x / 1000.0
        return x
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return None
    return None


@dataclass
class Friction:
    fee_rate: float
    half_spread: float
    slippage: float

    @property
    def kappa(self) -> float:
        return float(self.fee_rate + self.half_spread + self.slippage)


def compute_returns_from_price(price: np.ndarray, horizon_steps: int) -> np.ndarray:
    r = np.full_like(price, np.nan, dtype=np.float64)
    h = int(horizon_steps)
    if h <= 0:
        return r
    p0 = price[:-h]
    p1 = price[h:]
    r[:-h] = (p1 - p0) / np.maximum(p0, 1e-12)
    return r


def aggregate_returns(returns: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return returns.copy()
    n = (len(returns) // k) * k
    if n <= 0:
        return np.array([], dtype=returns.dtype)
    r = returns[:n].reshape(-1, k)
    compounded = np.prod(1.0 + r, axis=1) - 1.0
    return compounded.astype(np.float64)


def decimate_by_k(x: np.ndarray, k: int, how: str = "last") -> np.ndarray:
    if k <= 1:
        return x.copy()
    n = (len(x) // k) * k
    if n <= 0:
        return np.array([], dtype=x.dtype)
    x2 = x[:n].reshape(-1, k)
    if how == "last":
        return x2[:, -1]
    if how == "mean":
        return np.nanmean(x2, axis=1)
    raise ValueError(f"unknown decimation mode: {how}")


def edge_cost_density(
    x: np.ndarray,
    m: np.ndarray,
    r_h: np.ndarray,
    kappa: float,
    edge_mode: str,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    if len(x) < 3:
        return float("nan"), 0.0, 0.0

    x_prev = x[:-1]
    dx = np.diff(x)
    m_t = m[1:]
    if edge_mode == "delta_m":
        dm = np.diff(m.astype(np.float64))
        edge = x_prev * dm
    elif edge_mode == "return":
        edge = x_prev * r_h[1:]
        edge = np.nan_to_num(edge, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError(f"unknown edge_mode: {edge_mode}")

    cost = (kappa * m_t * np.abs(dx)).astype(np.float64)
    s_edge = float(np.sum(edge))
    s_cost = float(np.sum(cost))
    rho = s_edge / (s_cost + eps)
    return rho, s_edge, s_cost


def noise_floor_null(
    x: np.ndarray,
    m: np.ndarray,
    r_h: np.ndarray,
    kappa: float,
    edge_mode: str,
    n_shuffles: int,
    rng: random.Random,
) -> float:
    if n_shuffles <= 0:
        return float("nan")
    vals: list[float] = []
    for _ in range(n_shuffles):
        if edge_mode == "delta_m":
            perm = list(m)
            rng.shuffle(perm)
            m_perm = np.array(perm, dtype=np.float64)
            rho, _, _ = edge_cost_density(x, m_perm, r_h, kappa, edge_mode=edge_mode)
        else:
            perm = list(r_h)
            rng.shuffle(perm)
            r_perm = np.array(perm, dtype=np.float64)
            rho, _, _ = edge_cost_density(x, m, r_perm, kappa, edge_mode=edge_mode)
        if math.isfinite(rho):
            vals.append(rho)
    if not vals:
        return float("nan")
    return float(np.median(vals))


def robust_eps_check(
    x: np.ndarray,
    m: np.ndarray,
    r_h: np.ndarray,
    kappa: float,
    edge_mode: str,
    eps_cost_frac: float,
) -> tuple[bool, float]:
    base_rho, _, base_cost = edge_cost_density(x, m, r_h, kappa, edge_mode=edge_mode)
    if not math.isfinite(base_rho) or base_cost <= 0:
        return False, float("nan")

    deltas: list[float] = []
    robust = True
    for sgn in (-1.0, 1.0):
        k2 = kappa * (1.0 + sgn * eps_cost_frac)
        rho2, _, _ = edge_cost_density(x, m, r_h, k2, edge_mode=edge_mode)
        if math.isfinite(rho2):
            deltas.append(abs(rho2 - base_rho))
            if (base_rho > 0) != (rho2 > 0):
                robust = False
    if not deltas:
        return False, float("nan")
    return robust, float(max(deltas))


def extract_series(
    rows: list[dict[str, Any]],
    symbol: str,
    x_path: str,
    d_path: str,
    m_path: Optional[str],
    price_path: Optional[str],
    returns_path: Optional[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ts_list: list[float] = []
    x_list: list[float] = []
    d_list: list[float] = []
    m_list: list[float] = []
    p_list: list[float] = []
    r_list: list[float] = []

    for row in rows:
        if str(row.get("symbol") or "") != symbol:
            continue
        ts = coerce_ts_seconds(row.get("ts") or row.get("t") or row.get("timestamp"))
        if ts is None:
            continue
        x_val = safe_get(row, x_path, None)
        d_val = safe_get(row, d_path, None)
        if x_val is None or d_val is None:
            continue
        try:
            x = float(x_val)
            d = float(d_val)
        except (TypeError, ValueError):
            continue
        if m_path:
            m_val = safe_get(row, m_path, None)
            try:
                m = float(m_val)
            except (TypeError, ValueError):
                m = float(abs(d) > 0)
        else:
            m = float(abs(d) > 0)

        ts_list.append(ts)
        x_list.append(x)
        d_list.append(d)
        m_list.append(m)

        if price_path:
            pv = safe_get(row, price_path, float("nan"))
            try:
                p_list.append(float(pv))
            except (TypeError, ValueError):
                p_list.append(float("nan"))
        else:
            p_list.append(float("nan"))

        if returns_path:
            rv = safe_get(row, returns_path, float("nan"))
            try:
                r_list.append(float(rv))
            except (TypeError, ValueError):
                r_list.append(float("nan"))
        else:
            r_list.append(float("nan"))

    if not ts_list:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    ts_arr = np.asarray(ts_list, dtype=np.float64)
    x_arr = np.asarray(x_list, dtype=np.float64)
    d_arr = np.asarray(d_list, dtype=np.float64)
    m_arr = np.asarray(m_list, dtype=np.float64)
    p_arr = np.asarray(p_list, dtype=np.float64)
    r_arr = np.asarray(r_list, dtype=np.float64)

    order = np.argsort(ts_arr)
    return (
        ts_arr[order],
        x_arr[order],
        d_arr[order],
        m_arr[order],
        p_arr[order],
        r_arr[order],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-7.3 horizon sweep emitter (NDJSON).")
    ap.add_argument("--tower-log", required=True, help="Tower NDJSON (per-tick projection rows).")
    ap.add_argument("--symbol", required=True, help="Symbol to sweep (exact match).")
    ap.add_argument("--out", required=True, help="Output NDJSON path for sweep results.")

    ap.add_argument("--base-dt-s", type=float, default=1.0, help="Base timestep seconds.")
    ap.add_argument("--taus", type=str, default="10,30,60,120,300", help="Comma-separated horizons in seconds.")

    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--half-spread", type=float, default=0.0003)
    ap.add_argument("--slippage", type=float, default=0.0)

    ap.add_argument(
        "--edge-mode",
        choices=["delta_m", "return"],
        default="delta_m",
        help="Edge definition: delta_m matches COMPACTIFIED_CONTEXT.",
    )
    ap.add_argument("--eps-cost-frac", type=float, default=0.05, help="Cost perturbation fraction for robustness.")
    ap.add_argument("--n-shuffles", type=int, default=25, help="Null-model shuffles per tau.")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--x-path", type=str, default="P9.action.x", help="Dotted path to signed exposure x_t.")
    ap.add_argument("--d-path", type=str, default="P9.action.d", help="Dotted path to direction d_t.")
    ap.add_argument("--m-path", type=str, default="", help="Optional dotted path to support m_t.")
    ap.add_argument(
        "--price-path",
        type=str,
        default="price",
        help="Dotted path to price (used for edge-mode=return). Use '' to disable.",
    )
    ap.add_argument(
        "--returns-path",
        type=str,
        default="",
        help="Optional dotted path to per-step returns (used for edge-mode=return).",
    )

    args = ap.parse_args()

    tower_path = Path(args.tower_log)
    out_path = Path(args.out)
    symbol = args.symbol

    taus = [int(x.strip()) for x in args.taus.split(",") if x.strip()]
    taus = [t for t in taus if t > 0]
    if not taus:
        raise SystemExit("No valid taus provided.")

    fr = Friction(fee_rate=args.fee_rate, half_spread=args.half_spread, slippage=args.slippage)
    rng = random.Random(args.seed)

    rows = load_ndjson(tower_path)
    if not rows:
        raise SystemExit(f"No rows found in {tower_path}")

    m_path = args.m_path.strip() if args.m_path is not None else ""
    price_path = args.price_path.strip() if args.price_path is not None else ""
    returns_path = args.returns_path.strip() if args.returns_path is not None else ""
    m_path = m_path or None
    price_path = price_path or None
    returns_path = returns_path or None

    ts, x, d, m, price, returns = extract_series(
        rows, symbol, args.x_path, args.d_path, m_path, price_path, returns_path
    )
    if len(ts) < 10:
        raise SystemExit(f"Not enough rows for symbol {symbol} in {tower_path}")

    run_id = rows[0].get("run_id") or "unknown"
    sweep_rows: list[dict[str, Any]] = []

    for tau_s in taus:
        k = max(1, int(round(tau_s / float(args.base_dt_s))))
        x_h = decimate_by_k(x, k, how="last")
        m_h = decimate_by_k(m, k, how="last")
        r_h = np.zeros_like(x_h, dtype=np.float64)

        if args.edge_mode == "return":
            if returns_path:
                r_h = aggregate_returns(returns, k)
            elif price_path and np.isfinite(price).any():
                p_h = decimate_by_k(price, k, how="last")
                r_h = compute_returns_from_price(p_h, horizon_steps=1)
            else:
                raise SystemExit("edge-mode=return requires --price-path or --returns-path.")

        rho, s_edge, s_cost = edge_cost_density(x_h, m_h, r_h, fr.kappa, edge_mode=args.edge_mode)
        rho_null = noise_floor_null(
            x_h,
            m_h,
            r_h,
            fr.kappa,
            edge_mode=args.edge_mode,
            n_shuffles=args.n_shuffles,
            rng=rng,
        )
        robust, robust_eps = robust_eps_check(
            x_h, m_h, r_h, fr.kappa, edge_mode=args.edge_mode, eps_cost_frac=float(args.eps_cost_frac)
        )

        delta = (rho - rho_null) if (math.isfinite(rho) and math.isfinite(rho_null)) else float("nan")
        noise_ratio = (
            rho / (rho_null + 1e-12)
            if (math.isfinite(rho) and math.isfinite(rho_null))
            else float("nan")
        )
        net_positive = bool(math.isfinite(delta) and (delta > 1e-6) and robust)
        if not math.isfinite(rho) or rho <= 0:
            class_label = "false"
        elif not robust:
            class_label = "boundary-unstable"
        elif not net_positive:
            class_label = "boundary-unstable"
        else:
            class_label = "boundary-stable"

        sweep_rows.append(
            {
                "ts": float(ts[-1]),
                "symbol": symbol,
                "run_id": run_id,
                "phase": "7.3",
                "tau_s": int(tau_s),
                "k": int(k),
                "base_dt_s": float(args.base_dt_s),
                "edge_mode": args.edge_mode,
                "kappa": fr.kappa,
                "rho_A": float(rho) if math.isfinite(rho) else None,
                "rho_A_null": float(rho_null) if math.isfinite(rho_null) else None,
                "rho_null": float(rho_null) if math.isfinite(rho_null) else None,
                "delta_rho_A": float(delta) if math.isfinite(delta) else None,
                "delta_rho": float(delta) if math.isfinite(delta) else None,
                "noise_ratio": float(noise_ratio) if math.isfinite(noise_ratio) else None,
                "sum_edge": float(s_edge),
                "sum_cost": float(s_cost),
                "robust": bool(robust),
                "robust_pass": bool(robust),
                "robust_eps": float(robust_eps) if math.isfinite(robust_eps) else None,
                "net_positive": bool(net_positive),
                "class_label": class_label,
                "n_rows": int(len(x_h)),
            }
        )

    dump_ndjson(out_path, sweep_rows)
    print(f"[phase7.3] wrote {len(sweep_rows)} sweep rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

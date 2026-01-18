from __future__ import annotations

import argparse
import json
import math
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


def coerce_ts_seconds(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        x = float(value)
        if x > 1e11:
            return x / 1000.0
        return x
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        try:
            return float(text)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return fallback
    return fallback


@dataclass
class NearMiss:
    symbol: str
    t0: float
    t1: float
    cls: str
    rank: float
    evidence: dict[str, Any]


def _mean(values: list[Optional[float]]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _window_indices(ts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return np.where((ts >= t0) & (ts <= t1))[0]


def rank_near_misses(
    tower_rows: list[dict[str, Any]],
    window_s: float = 300.0,
    stride_s: float = 60.0,
    min_points: int = 50,
    symbol_filter: Optional[str] = None,
) -> list[NearMiss]:
    if not tower_rows:
        return []

    symbols = [str(r.get("symbol") or "") for r in tower_rows]
    if symbol_filter is None:
        symbol_filter = next((s for s in symbols if s), "default")

    rows = [r for r in tower_rows if str(r.get("symbol") or "") == symbol_filter]
    if not rows:
        return []

    ts = np.array([coerce_ts_seconds(r.get("ts") or r.get("t"), i) for i, r in enumerate(rows)], dtype=np.float64)

    A7 = [safe_get(r, "P7.A7", None) for r in rows]
    A8 = [safe_get(r, "P8.A8", None) for r in rows]
    A9 = [safe_get(r, "P9.A9", None) for r in rows]

    m8_pre = [safe_get(r, "M8.precheck", None) for r in rows]
    m8_open = [safe_get(r, "M8.open", None) for r in rows]
    components = [safe_get(r, "M8.components", {}) or {} for r in rows]

    def cflag(i: int, name: str) -> Optional[float]:
        v = components[i].get(name)
        if v is None:
            return None
        return 1.0 if bool(v) else 0.0

    t_min, t_max = float(np.min(ts)), float(np.max(ts))
    out: list[NearMiss] = []
    t0 = t_min

    while t0 + window_s <= t_max:
        t1 = t0 + window_s
        idx = _window_indices(ts, t0, t1)
        if idx.size < min_points:
            t0 += stride_s
            continue

        a7m = _mean([A7[i] for i in idx])
        a8m = _mean([A8[i] for i in idx])
        a9m = _mean([A9[i] for i in idx])
        pre_rate = _mean([(1.0 if m8_pre[i] else 0.0) for i in idx if m8_pre[i] is not None])

        boundary = _mean([cflag(i, "boundary_stable") for i in idx])
        ready = _mean([cflag(i, "phase8_ready") for i in idx])
        actfix = _mean([cflag(i, "actuator_mode_fixed") for i in idx])
        ledger = _mean([cflag(i, "ledger_ready") for i in idx])
        horiz = _mean([cflag(i, "horizon_certified") for i in idx])

        if math.isfinite(a7m) and a7m >= 0.4 and (not math.isfinite(a8m) or a8m < 1.0):
            rank = float(a7m * (1.0 - (a8m if math.isfinite(a8m) else 0.0)))
            out.append(
                NearMiss(
                    symbol=symbol_filter,
                    t0=t0,
                    t1=t1,
                    cls="boundary_ready_near_miss",
                    rank=rank,
                    evidence={"A7_mean": a7m, "A8_mean": a8m, "P8_ready": ready},
                )
            )

        if math.isfinite(pre_rate) and pre_rate > 0.2 and (not math.isfinite(horiz) or horiz < 1.0):
            missing = 1.0 - (horiz if math.isfinite(horiz) else 0.0)
            rank = float(pre_rate * missing)
            out.append(
                NearMiss(
                    symbol=symbol_filter,
                    t0=t0,
                    t1=t1,
                    cls="m8_precheck_missing_horizon",
                    rank=rank,
                    evidence={
                        "pre_rate": pre_rate,
                        "components": {
                            "boundary": boundary,
                            "phase8_ready": ready,
                            "actuator_fixed": actfix,
                            "ledger_ready": ledger,
                            "horizon_certified": horiz,
                        },
                    },
                )
            )

        open_rate = _mean([(1.0 if m8_open[i] else 0.0) for i in idx if m8_open[i] is not None])
        if math.isfinite(open_rate) and open_rate > 0.2 and math.isfinite(a9m) and a9m < 1.0:
            rank = float(open_rate * (1.0 - a9m))
            out.append(
                NearMiss(
                    symbol=symbol_filter,
                    t0=t0,
                    t1=t1,
                    cls="witness_blocked",
                    rank=rank,
                    evidence={"M8_open_rate": open_rate, "A9_mean": a9m},
                )
            )

        t0 += stride_s

    out.sort(key=lambda nm: nm.rank, reverse=True)
    return out


def _to_row(nm: NearMiss) -> dict[str, Any]:
    return {
        "symbol": nm.symbol,
        "t0": nm.t0,
        "t1": nm.t1,
        "cls": nm.cls,
        "rank": nm.rank,
        "evidence": nm.evidence,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank M8/M9 near-miss windows from tower logs.")
    ap.add_argument("--tower-log", required=True, help="Tower NDJSON log to score.")
    ap.add_argument("--out", default="", help="Output NDJSON path (auto-timestamped if empty).")
    ap.add_argument("--window-s", type=float, default=300.0, help="Window size in seconds.")
    ap.add_argument("--stride-s", type=float, default=60.0, help="Stride in seconds.")
    ap.add_argument("--min-points", type=int, default=50, help="Minimum rows per window.")
    ap.add_argument("--symbol", default="", help="Optional symbol filter.")
    ap.add_argument("--top", type=int, default=50, help="Max rows to write.")
    args = ap.parse_args()

    tower_path = Path(args.tower_log)
    rows = load_ndjson(tower_path)
    if not rows:
        raise SystemExit(f"No rows found in {tower_path}")

    symbol = args.symbol.strip() or None
    ranked = rank_near_misses(
        rows,
        window_s=float(args.window_s),
        stride_s=float(args.stride_s),
        min_points=int(args.min_points),
        symbol_filter=symbol,
    )

    if args.top > 0:
        ranked = ranked[: args.top]

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = Path("logs/near_miss") / f"near_miss_rank_{ts}.ndjson"

    dump_ndjson(out_path, [_to_row(nm) for nm in ranked])
    print(f"[near-miss] wrote {len(ranked)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

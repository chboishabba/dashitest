from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


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


def dump_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, allow_nan=False))
            fh.write("\n")


def safe_get(d: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _pick_first(record: dict[str, Any], paths: tuple[str, ...]) -> Any:
    for path in paths:
        val = safe_get(record, path, None)
        if val is not None:
            return val
    return None


def _action_active(record: dict[str, Any]) -> Optional[bool]:
    m_val = _pick_first(record, ("M9.action.m", "P9.action.m", "action.m", "action_m"))
    if m_val is None:
        action_t = _pick_first(record, ("action_t", "P9.action_t", "M9.action_t"))
        if action_t is None:
            return None
        try:
            return bool(int(action_t) != 0)
        except (TypeError, ValueError):
            return None
    try:
        return bool(float(m_val) != 0.0)
    except (TypeError, ValueError):
        return None


def _phase6_closed(record: dict[str, Any]) -> Optional[bool]:
    phase6_open = _pick_first(record, ("phase6_open", "phase6_gate.open", "phase6_gate_open"))
    if phase6_open is None:
        return None
    return not bool(phase6_open)


@dataclass
class InvariantViolation:
    t: Any
    symbol: str
    rule: str
    detail: str


def check_record(record: dict[str, Any]) -> list[InvariantViolation]:
    violations: list[InvariantViolation] = []
    symbol = str(record.get("symbol") or "")
    t_val = record.get("ts") or record.get("t")

    m8_open = safe_get(record, "M8.open", None)
    m8_pre = safe_get(record, "M8.precheck", None)
    m8_horizon = safe_get(record, "M8.components.horizon_certified", None)
    refusal = safe_get(record, "P9.refusal", None)
    action_active = _action_active(record)
    phase6_closed = _phase6_closed(record)

    if m8_open is True and m8_pre is False:
        violations.append(
            InvariantViolation(t_val, symbol, "I1", "M8.open true but M8.precheck false")
        )
    if m8_open is True and m8_horizon is False:
        violations.append(
            InvariantViolation(t_val, symbol, "I2", "M8.open true but horizon_certified false")
        )
    if phase6_closed is True and m8_open is True:
        violations.append(
            InvariantViolation(t_val, symbol, "I3", "Phase-6 closed but M8.open true")
        )
    if phase6_closed is True and action_active is True:
        violations.append(
            InvariantViolation(t_val, symbol, "I3", "Phase-6 closed but action emitted")
        )
    if refusal in {"HOLD", "BAN"} and action_active is True:
        violations.append(
            InvariantViolation(t_val, symbol, "I6", "Witness refusal but action emitted")
        )
    if action_active is True:
        justification = safe_get(record, "M9.justification", None)
        if not isinstance(justification, dict):
            violations.append(
                InvariantViolation(t_val, symbol, "I5", "Missing M9.justification on action")
            )
        else:
            missing = [k for k in ("run_id", "M8", "P7", "witness") if k not in justification]
            if missing:
                violations.append(
                    InvariantViolation(
                        t_val,
                        symbol,
                        "I5",
                        f"Justification missing keys: {','.join(missing)}",
                    )
                )

    return violations


def check_rows(rows: list[dict[str, Any]]) -> list[InvariantViolation]:
    out: list[InvariantViolation] = []
    for row in rows:
        out.extend(check_record(row))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate M8 -> M9 handoff invariants on tower logs.")
    ap.add_argument("--tower-log", required=True, help="Tower NDJSON log to check.")
    ap.add_argument("--out", default="", help="Output NDJSON for violations (auto-timestamped if empty).")
    args = ap.parse_args()

    rows = load_ndjson(Path(args.tower_log))
    violations = check_rows(rows)

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = Path("logs/diagnostics") / f"m8_m9_invariants_{ts}.ndjson"

    dump_ndjson(
        out_path,
        [
            {"ts": v.t, "symbol": v.symbol, "rule": v.rule, "detail": v.detail}
            for v in violations
        ],
    )
    print(f"[m8->m9] wrote {len(violations)} violations -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

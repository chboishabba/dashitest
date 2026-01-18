from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(val):
        return val
    return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes", "y"}:
            return True
        if lowered in {"false", "f", "0", "no", "n"}:
            return False
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _pick_first(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data.get(key)
    return None


def _clean_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except Exception:
            pass
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
    try:
        item = value.item()
    except Exception:
        return value
    return _clean_json(item)


def build_tower_projection(
    step_row: dict[str, Any],
    run_id: str | None = None,
    posture_stable_min: int = 3,
) -> dict[str, Any]:
    posture = _safe_int(step_row.get("direction"))
    hold = _safe_int(step_row.get("hold"))
    state_age = _safe_int(step_row.get("state_age"))
    align_age = _safe_int(step_row.get("align_age"))
    stable = None
    if posture is not None and state_age is not None:
        stable = posture != 0 and state_age >= posture_stable_min
    if posture is None:
        a5 = None
    elif posture == 0:
        a5 = 0.5
    else:
        a5 = 1.0 if stable else 0.0 if stable is False else None

    p_bad = _safe_float(step_row.get("p_bad"))
    legitimacy_proxy = None if p_bad is None else 1.0 - p_bad
    exploitability_proxy = _safe_float(step_row.get("pred_edge"))

    permission = _safe_int(step_row.get("permission"))
    action_t = _safe_int(step_row.get("action_t"))
    boundary_abstain = _safe_int(step_row.get("boundary_abstain"))
    boundary_stable = None if boundary_abstain is None else boundary_abstain == 0
    phase8_open = _safe_bool(_pick_first(step_row, ("phase8_open", "phase8_ready", "phase8_gate_open")))
    actuator_mode_fixed = _safe_bool(step_row.get("actuator_mode_fixed"))
    ledger_ready = _safe_bool(step_row.get("ledger_ready"))
    horizon_certified = _safe_bool(step_row.get("horizon_certified"))
    m8_precheck = (
        boundary_stable is True
        and phase8_open is True
        and actuator_mode_fixed is True
        and ledger_ready is True
    )
    m8_reason = None
    if boundary_stable is False:
        m8_reason = "boundary_unstable"
    elif phase8_open is False:
        m8_reason = "phase8_closed"
    elif actuator_mode_fixed is False:
        m8_reason = "actuator_mode_unfixed"
    elif ledger_ready is False:
        m8_reason = "ledger_missing"
    elif None in {boundary_stable, phase8_open, actuator_mode_fixed, ledger_ready}:
        m8_reason = "missing_prereqs"
    elif horizon_certified in {None, False}:
        m8_reason = "missing_horizon_cert"
    if permission == -1:
        refusal = "BAN"
    elif permission == 0 or boundary_abstain == 1 or action_t == 0:
        refusal = "HOLD"
    else:
        refusal = "NONE"
    a9 = 0.0 if refusal == "BAN" else 0.5 if refusal == "HOLD" else 1.0

    record = {
        "t": _safe_int(step_row.get("t")),
        "ts": step_row.get("ts"),
        "symbol": step_row.get("symbol") or step_row.get("source") or "",
        "run_id": run_id or step_row.get("tape_id") or step_row.get("source") or "",
        "P1": {
            "available": False,
            "q": {
                "e64": _safe_float(step_row.get("q_e64")),
                "c64": _safe_float(step_row.get("q_c64")),
                "s64": _safe_float(step_row.get("q_s64")),
                "delta_e": _safe_float(step_row.get("q_de")),
                "delta_c": _safe_float(step_row.get("q_dc")),
                "delta_s": _safe_float(step_row.get("q_ds")),
            },
            "kappa": None,
            "eps": None,
            "closed_scales": None,
            "total_scales": None,
            "A1": None,
        },
        "P2": {"available": False, "window": None, "closed_fraction": None, "persistent": None, "A2": None},
        "P3": {"available": False, "renorm_residual": None, "threshold": None, "stable": None, "A3": None},
        "P4": {
            "available": False,
            "variance_across_scales": None,
            "max_allowed": None,
            "coherent": None,
            "A4": None,
        },
        "P5": {
            "available": True,
            "posture": posture,
            "posture_source": "direction",
            "hold": hold,
            "state_age": state_age,
            "align_age": align_age,
            "stable": stable,
            "A5": a5,
        },
        "P6": {
            "available": False,
            "legitimacy_proxy": legitimacy_proxy,
            "exploitability_proxy": exploitability_proxy,
            "agreement": None,
            "A6": None,
        },
        "P7": {
            "available": False,
            "tau_s": None,
            "rho_A": None,
            "rho_A_null": None,
            "delta_rho_A": None,
            "robust": None,
            "A7": None,
            "boundary_gate": {
                "enabled": _safe_int(step_row.get("boundary_gate_enabled")),
                "abstain": boundary_abstain,
                "edge_confidence": _safe_float(step_row.get("boundary_edge_confidence")),
                "cost_threshold": _safe_float(step_row.get("boundary_cost_threshold")),
                "reason": step_row.get("boundary_gate_reason") or "",
            },
        },
        "P8": {
            "available": False,
            "ready_count": None,
            "required": None,
            "window": None,
            "open": None,
            "reason": None,
            "A8": None,
        },
        "M8": {
            "available": True,
            "precheck": m8_precheck,
            "open": False,
            "A8": None,
            "components": {
                "boundary_stable": boundary_stable,
                "phase8_ready": phase8_open,
                "actuator_mode_fixed": actuator_mode_fixed,
                "ledger_ready": ledger_ready,
                "horizon_certified": horizon_certified if horizon_certified is not None else False,
            },
            "horizon": {
                "tau_s": None,
                "rho_A": None,
                "rho_null": None,
                "net_positive": None,
                "robust_eps": None,
            },
            "reason": m8_reason,
            "run_id": run_id or step_row.get("tape_id") or step_row.get("source") or "",
        },
        "P9": {
            "available": True,
            "permission": permission,
            "refusal": refusal,
            "A9": a9,
            "capital_pressure": _safe_int(step_row.get("capital_pressure")),
            "risk_budget": _safe_float(step_row.get("risk_budget")),
            "cap": _safe_float(step_row.get("cap")),
            "equity": _safe_float(step_row.get("equity")),
            "cash": _safe_float(step_row.get("cash")),
        },
    }
    return _clean_json(record)

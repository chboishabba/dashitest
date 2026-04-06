from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .family_memory import FamilyMemory, FamilyStepRecord, RuntimeFamilyState, classify_runtime_family_state
from .schema import FamilyClass, family_class_constructor_name


@dataclass(frozen=True)
class FamilyCertification:
    family_state: RuntimeFamilyState
    family_class: FamilyClass
    family_constructor: str
    certified: bool
    trade_certified: bool
    preserve_certified: bool
    cone_ok: bool
    fejer_ok: bool
    closest_ok: bool
    mdl_exact_ok: bool
    tail_localized: bool
    spread_dominated: bool
    hostile_regime: bool
    window_size: int
    arrow_boundary_share: float
    microstructure_kill_share: float
    reasons: tuple[str, ...]


def _share(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _family_class(records: list[FamilyStepRecord], *, tail_localized: bool) -> FamilyClass:
    status_counts = Counter(record.step_status for record in records)
    arrow_count = status_counts.get("arrow_boundary", 0)
    if tail_localized:
        return FamilyClass.MDL_TAIL_BOUNDARY
    if arrow_count >= 2:
        return FamilyClass.ARROW_LADDER
    if arrow_count == 1:
        return FamilyClass.SINGLE_ARROW_BREAK
    return FamilyClass.INTERIOR_FAMILY


def certify_family(memory: FamilyMemory) -> FamilyCertification:
    records = list(memory.recent_records)
    if not records:
        family_class = FamilyClass.INTERIOR_FAMILY
        return FamilyCertification(
            family_state=RuntimeFamilyState.INTERIOR_PERSISTENT,
            family_class=family_class,
            family_constructor=family_class_constructor_name(family_class),
            certified=True,
            trade_certified=True,
            preserve_certified=False,
            cone_ok=True,
            fejer_ok=True,
            closest_ok=True,
            mdl_exact_ok=True,
            tail_localized=False,
            spread_dominated=False,
            hostile_regime=False,
            window_size=0,
            arrow_boundary_share=0.0,
            microstructure_kill_share=0.0,
            reasons=tuple(),
        )

    window_size = len(records)
    status_counts = Counter(record.step_status for record in records)
    structural_violations = status_counts.get("structural_boundary", 0) + status_counts.get("outside", 0)
    arrow_count = status_counts.get("arrow_boundary", 0)
    mdl_breaks = sum(1 for record in records if record.mdl_delta > 1e-9 or "mdl" in record.reasons)
    micro_kills = sum(1 for record in records if record.microstructure_kills_edge)
    hazard_hostile = sum(1 for record in records if record.hazard_tightened)
    spread_stressed = sum(1 for record in records if record.spread_regime != "spread_clear")
    spread_dominated = micro_kills > 0
    terminal_hostile = any(record.hazard_tightened for record in records[-2:])
    hostile_rate = _share(hazard_hostile, window_size)
    spread_stressed_rate = _share(spread_stressed, window_size)
    reentry_ready = memory.hazard_reentry_ready and not terminal_hostile
    hostile_regime = (
        not reentry_ready
        and (
            (terminal_hostile and hostile_rate >= 0.35)
            or (terminal_hostile and spread_stressed_rate >= 0.60)
            or memory.hazard_hostile_streak >= 3
        )
    )
    churn_guard_active = memory.post_unwind_cooldown > 0

    cone_ok = structural_violations == 0
    fejer_ok = cone_ok and all(record.q_delta <= 1e-9 for record in records)
    closest_ok = cone_ok
    mdl_exact_ok = mdl_breaks == 0
    tail_localized = (
        (0 < mdl_breaks < window_size)
        or (0 < micro_kills < window_size)
        or (0 < hazard_hostile < window_size)
    )

    family_class = _family_class(records, tail_localized=tail_localized)
    family_state = classify_runtime_family_state(memory)
    trade_certified = (
        cone_ok
        and closest_ok
        and mdl_exact_ok
        and not spread_dominated
        and not hostile_regime
        and not churn_guard_active
    )
    preserve_certified = cone_ok and closest_ok and (spread_dominated or hostile_regime or churn_guard_active)
    certified = trade_certified or preserve_certified

    reasons: list[str] = []
    if not cone_ok:
        reasons.append("structural_escape")
    if not fejer_ok:
        reasons.append("cone_drift")
    if not mdl_exact_ok:
        reasons.append("mdl_tail_boundary")
    if spread_dominated:
        reasons.append("microstructure_kills_edge")
    if hazard_hostile > 0:
        reasons.append("hazard_tightening")
    if hostile_regime:
        reasons.append("hostile_regime")
    if churn_guard_active:
        reasons.append("hot_handed_churn")
    if reentry_ready:
        reasons.append("hazard_reentry_ready")

    return FamilyCertification(
        family_state=family_state,
        family_class=family_class,
        family_constructor=family_class_constructor_name(family_class),
        certified=certified,
        trade_certified=trade_certified,
        preserve_certified=preserve_certified,
        cone_ok=cone_ok,
        fejer_ok=fejer_ok,
        closest_ok=closest_ok,
        mdl_exact_ok=mdl_exact_ok,
        tail_localized=tail_localized,
        spread_dominated=spread_dominated,
        hostile_regime=hostile_regime,
        window_size=window_size,
        arrow_boundary_share=_share(arrow_count, window_size),
        microstructure_kill_share=_share(micro_kills, window_size),
        reasons=tuple(reasons),
    )

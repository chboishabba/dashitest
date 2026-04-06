from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .contracts import ExecutionDecision, StepStatus


class RuntimeFamilyState(str, Enum):
    INTERIOR_PERSISTENT = "interior_persistent"
    ARROW_BOUNDARY_RECOVERABLE = "arrow_boundary_recoverable"
    SINGLE_BREAK_CHURN = "single_break_churn"
    TAIL_BOUNDARY_STIFF = "tail_boundary_stiff"


@dataclass(frozen=True)
class FamilyStepRecord:
    step_status: str
    family_hint: str
    spread_regime: str
    microstructure_kills_edge: bool
    hazard_level: float
    hazard_tightened: bool
    q_delta: float
    mdl_delta: float
    eigen_overlap: float
    cost_survival_ratio: float
    reasons: tuple[str, ...]


@dataclass
class FamilyMemory:
    last_step_status: StepStatus = StepStatus.INTERIOR
    last_family_hint: str = "interior"
    family_state: RuntimeFamilyState = RuntimeFamilyState.INTERIOR_PERSISTENT
    arrow_boundary_streak: int = 0
    mdl_boundary_count: int = 0
    hazard_hostile_streak: int = 0
    hazard_calm_streak: int = 0
    last_hazard_level: float = 0.0
    open_position_age: int = 0
    post_unwind_cooldown: int = 0
    recent_statuses: list[str] = field(default_factory=list)
    recent_records: list[FamilyStepRecord] = field(default_factory=list)

    def observe(
        self,
        decision: ExecutionDecision,
        family_hint: str,
        *,
        spread_regime: str = "spread_clear",
        microstructure_kills_edge: bool = False,
        cost_survival_ratio: float = 0.0,
        hazard_level: float = 0.0,
        hazard_tightened: bool = False,
        prior_exposure: float = 0.0,
        new_exposure: float = 0.0,
        fill: float = 0.0,
    ) -> None:
        self.last_step_status = decision.status
        self.last_family_hint = family_hint
        base_hazard_level = max(0.0, min(1.0, float(hazard_level)))
        spread_uplift = 0.10 if spread_regime == "spread_stressed" else 0.0
        microstructure_uplift = 0.15 if microstructure_kills_edge else 0.0
        resolved_hazard_level = max(
            0.0,
            min(1.0, base_hazard_level + spread_uplift + microstructure_uplift),
        )
        resolved_hazard_tightened = bool(
            hazard_tightened
            or base_hazard_level >= 0.40
            or (microstructure_kills_edge and base_hazard_level >= 0.25)
            or (spread_regime == "spread_stressed" and base_hazard_level >= 0.30)
        )
        calm_reentry = bool(
            base_hazard_level < 0.25
            and spread_regime == "spread_clear"
            and not microstructure_kills_edge
        )
        self.last_hazard_level = resolved_hazard_level
        if decision.status == StepStatus.ARROW_BOUNDARY:
            self.arrow_boundary_streak += 1
        else:
            self.arrow_boundary_streak = 0
        if "mdl" in decision.reasons:
            self.mdl_boundary_count += 1
        if resolved_hazard_tightened:
            self.hazard_hostile_streak += 1
            self.hazard_calm_streak = 0
        else:
            self.hazard_calm_streak += 1 if calm_reentry else 0
            if calm_reentry and self.hazard_calm_streak >= 2:
                self.hazard_hostile_streak = 0
            else:
                self.hazard_hostile_streak = max(0, self.hazard_hostile_streak - 1)

        flat_prev = abs(float(prior_exposure)) <= 1e-9
        flat_next = abs(float(new_exposure)) <= 1e-9
        traded = abs(float(fill)) > 1e-9
        if flat_prev and not flat_next and traded:
            self.open_position_age = 0
        elif not flat_prev and not flat_next:
            self.open_position_age += 1
        elif not flat_prev and flat_next and traded:
            if self.open_position_age <= 0:
                self.post_unwind_cooldown = max(self.post_unwind_cooldown, 3)
            else:
                self.post_unwind_cooldown = max(0, self.post_unwind_cooldown - 1)
            self.open_position_age = 0
        elif flat_next and self.post_unwind_cooldown > 0:
            self.post_unwind_cooldown -= 1
        self.recent_statuses.append(decision.status.value)
        if len(self.recent_statuses) > 8:
            self.recent_statuses = self.recent_statuses[-8:]
        self.recent_records.append(
            FamilyStepRecord(
                step_status=decision.status.value,
                family_hint=family_hint,
                spread_regime=spread_regime,
                microstructure_kills_edge=bool(microstructure_kills_edge),
                hazard_level=resolved_hazard_level,
                hazard_tightened=resolved_hazard_tightened,
                q_delta=float(decision.metrics.q_delta),
                mdl_delta=float(decision.metrics.mdl_next - decision.metrics.mdl_prev),
                eigen_overlap=float(decision.metrics.eigen_overlap),
                cost_survival_ratio=float(cost_survival_ratio),
                reasons=tuple(decision.reasons),
            )
        )
        if len(self.recent_records) > 16:
            self.recent_records = self.recent_records[-16:]
        self.family_state = classify_runtime_family_state(self)

    @property
    def hazard_reentry_ready(self) -> bool:
        return self.hazard_calm_streak >= 2 and self.hazard_hostile_streak == 0


def classify_runtime_family_state(memory: FamilyMemory) -> RuntimeFamilyState:
    records = list(memory.recent_records[-8:])
    if not records:
        return RuntimeFamilyState.INTERIOR_PERSISTENT

    arrow_recent = sum(1 for record in records if record.step_status == StepStatus.ARROW_BOUNDARY.value)
    mdl_recent = any(record.mdl_delta > 1e-9 or "mdl" in record.reasons for record in records)
    micro_kill_recent = any(record.microstructure_kills_edge for record in records)
    hostile_recent = sum(1 for record in records if record.hazard_tightened)
    tail_localized = (
        mdl_recent
        or memory.last_family_hint == "mdl_tail_boundary"
        or (
            arrow_recent > 0
            and hostile_recent > 0
            and hostile_recent < len(records)
            and not memory.hazard_reentry_ready
        )
    )
    if tail_localized:
        return RuntimeFamilyState.TAIL_BOUNDARY_STIFF
    if memory.post_unwind_cooldown > 0 or (arrow_recent == 1 and micro_kill_recent):
        return RuntimeFamilyState.SINGLE_BREAK_CHURN
    if (
        arrow_recent > 0
        or memory.arrow_boundary_streak > 0
        or memory.last_step_status == StepStatus.ARROW_BOUNDARY
        or memory.hazard_reentry_ready
        or memory.last_family_hint == "arrow_ladder"
    ):
        return RuntimeFamilyState.ARROW_BOUNDARY_RECOVERABLE
    return RuntimeFamilyState.INTERIOR_PERSISTENT

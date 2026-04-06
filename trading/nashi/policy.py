from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ExecutionDecision, NashiContract, StepStatus
from .proposals import CandidateTransition
from .severity import SeverityCode, SeverityLevel, combine_codes
from .state import NashiState, NashiStepContext


@dataclass(frozen=True)
class NashiIntent:
    ts: int
    direction: int
    target_exposure: float
    hold: bool
    reason: str


@dataclass(frozen=True)
class NashiPolicyInput:
    context: NashiStepContext
    current_embedding: np.ndarray
    candidates: tuple[CandidateTransition, ...]


@dataclass(frozen=True)
class NashiPolicyOutput:
    intent: NashiIntent
    contract_decision: ExecutionDecision
    refusal: SeverityCode
    selected_candidate_id: str
    selected_candidate_reason: str
    selected_candidate_executable_viable: bool
    selected_candidate_governance_viable: bool
    selected_candidate_cost_viable: bool
    selected_candidate_survivability_viable: bool
    selected_candidate_survivability_score: float
    selected_candidate_viability_reason: str
    selected_family_hint: str
    selected_spread_regime: str
    rejected_candidates: tuple[str, ...]


class NashiPolicyRuntime:
    """
    Minimal runtime wrapper around the formal admissibility contract.

    The current repo can feed this from existing controllers while the new
    policy grows into a first-class trader.
    """

    def __init__(self, contract: NashiContract) -> None:
        self.contract = contract

    def step(self, state: NashiState, policy_input: NashiPolicyInput) -> NashiPolicyOutput:
        selected_candidate = None
        selected_decision = None
        selected_refusal = None
        rejected: list[str] = []
        hold_candidate = None

        for candidate in policy_input.candidates:
            if candidate.candidate_id == "hold":
                hold_candidate = candidate
                continue
            elif not self._candidate_density_allowed(candidate, policy_input.context):
                rejected.append(
                    f"{candidate.candidate_id}:density_blocked:{self._density_reason(candidate, policy_input.context)}"
                )
                continue
            elif not candidate.proposal_economics.executable_viable:
                rejected.append(
                    f"{candidate.candidate_id}:proposal_blocked:{candidate.proposal_economics.viability_reason}"
                )
                continue
            decision = self.contract.evaluate_step(
                policy_input.current_embedding,
                candidate.proposed_embedding.augmented_vector(),
            )
            refusal = self._refusal_from_decision(decision, policy_input.context)
            if not refusal.is_blocking and decision.accepted:
                selected_candidate = candidate
                selected_decision = decision
                selected_refusal = refusal
                break
            rejected.append(
                f"{candidate.candidate_id}:{decision.status.value}:{'/'.join(decision.reasons) or refusal.label}"
            )

        if selected_candidate is None and hold_candidate is not None:
            selected_candidate = hold_candidate
            selected_decision = self.contract.evaluate_step(
                policy_input.current_embedding,
                hold_candidate.proposed_embedding.augmented_vector(),
            )
            selected_refusal = self._refusal_from_decision(selected_decision, policy_input.context)
        if selected_candidate is None or selected_decision is None or selected_refusal is None:
            raise ValueError("policy_input must include at least one hold candidate")

        hold = selected_refusal.is_blocking or selected_candidate.candidate_id == "hold" or selected_decision.status == StepStatus.ARROW_BOUNDARY
        if hold:
            intent = NashiIntent(
                ts=policy_input.context.ts,
                direction=0,
                target_exposure=state.exposure,
                hold=True,
                reason=selected_refusal.label,
            )
        else:
            intent = NashiIntent(
                ts=policy_input.context.ts,
                direction=int(np.sign(selected_candidate.signed_target_exposure)),
                target_exposure=float(max(-1.0, min(1.0, selected_candidate.target_exposure))),
                hold=False,
                reason=selected_candidate.rationale,
            )
        return NashiPolicyOutput(
            intent=intent,
            contract_decision=selected_decision,
            refusal=selected_refusal,
            selected_candidate_id=selected_candidate.candidate_id,
            selected_candidate_reason=selected_candidate.rationale,
            selected_candidate_executable_viable=selected_candidate.proposal_economics.executable_viable,
            selected_candidate_governance_viable=selected_candidate.proposal_economics.governance_viable,
            selected_candidate_cost_viable=selected_candidate.proposal_economics.cost_viable,
            selected_candidate_survivability_viable=selected_candidate.proposal_economics.survivability_viable,
            selected_candidate_survivability_score=float(selected_candidate.proposal_economics.survivability_score),
            selected_candidate_viability_reason=selected_candidate.proposal_economics.viability_reason,
            selected_family_hint=selected_candidate.family_hint,
            selected_spread_regime=selected_candidate.spread_regime,
            rejected_candidates=tuple(rejected),
        )

    @staticmethod
    def _candidate_density_allowed(
        candidate: CandidateTransition,
        context: NashiStepContext,
    ) -> bool:
        if candidate.candidate_id == "hold":
            return True
        actionability = max(0.0, float(context.actionability))
        signal_strength = actionability * abs(float(context.edge))
        if candidate.priority >= 0.999:
            return actionability >= 0.75 and signal_strength >= 0.60
        if candidate.priority >= 0.5:
            return actionability >= 0.50 and signal_strength >= 0.35
        return actionability >= 0.30 and signal_strength >= 0.20

    @staticmethod
    def _density_reason(
        candidate: CandidateTransition,
        context: NashiStepContext,
    ) -> str:
        actionability = max(0.0, float(context.actionability))
        signal_strength = actionability * abs(float(context.edge))
        if candidate.priority >= 0.999:
            if actionability < 0.75:
                return "full_size_requires_actionability_0.75"
            return "full_size_requires_signal_0.60"
        if candidate.priority >= 0.5:
            if actionability < 0.50:
                return "half_size_requires_actionability_0.50"
            return "half_size_requires_signal_0.35"
        if actionability < 0.30:
            return "probe_requires_actionability_0.30"
        if signal_strength < 0.20:
            return "probe_requires_signal_0.20"
        return "density_blocked"

    def _refusal_from_decision(
        self,
        decision: ExecutionDecision,
        context: NashiStepContext,
    ) -> SeverityCode:
        codes: list[SeverityCode] = []
        if "cone" in decision.reasons or "basin" in decision.reasons:
            codes.append(SeverityCode(SeverityLevel.BAN, "structural_escape"))
        if "mdl" in decision.reasons:
            codes.append(SeverityCode(SeverityLevel.HOLD, "mdl_increase"))
        if "eigen" in decision.reasons:
            codes.append(SeverityCode(SeverityLevel.HOLD, "eigen_drift"))
        if "arrow" in decision.reasons:
            codes.append(SeverityCode(SeverityLevel.CAUTION, "arrow_boundary"))
        if context.drawdown > 0.20:
            codes.append(SeverityCode(SeverityLevel.BAN, "capital_drawdown"))
        if context.actionability < 0.25:
            codes.append(SeverityCode(SeverityLevel.HOLD, "low_actionability"))
        return combine_codes(*codes)

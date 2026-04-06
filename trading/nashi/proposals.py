from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .hazard import HazardObservation, proposal_hazard_density
from .phase9 import CapitalParams, ProposalEconomics, assess_proposal_economics
from .schema import ClosureEmbedding
from .state import NashiState, NashiStepContext


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _hazard_density(observation: "NashiObservation") -> float:
    context = observation.context
    return proposal_hazard_density(
        HazardObservation(
            hazard_score=float(context.hazard),
            hazard_regime=str(context.hazard_regime),
            hazard_p_bad=float(context.hazard_p_bad),
            hazard_bad_flag=bool(context.hazard_bad_flag),
            hazard_synthetic_bad=False,
            hazard_drawdown_pressure=0.0,
            hazard_vol_pressure=0.0,
            hazard_spread_pressure=0.0,
            hazard_contextual_pressure=float(context.hazard_contextual_pressure),
            hazard_contextual_active=bool(context.hazard_contextual_active),
            hazard_contextual_label=str(context.hazard_contextual_label),
            hazard_ema=float(context.hazard_ema),
            hazard_persistence=float(context.hazard_persistence),
            hazard_trend=float(context.hazard_trend),
            hazard_cooldown=float(context.hazard_cooldown),
        )
    )


@dataclass(frozen=True)
class NashiObservation:
    context: NashiStepContext
    current_embedding: ClosureEmbedding
    state_signal: int


@dataclass(frozen=True)
class CandidateTransition:
    candidate_id: str
    signed_target_exposure: float
    proposed_embedding: ClosureEmbedding
    rationale: str
    family_hint: str
    spread_regime: str
    priority: float
    proposal_economics: ProposalEconomics

    @property
    def direction(self) -> int:
        return _sign(self.signed_target_exposure)

    @property
    def target_exposure(self) -> float:
        return abs(float(self.signed_target_exposure))


class ProposalGenerator:
    """
    Bounded candidate generator over the canonical embedding.

    This replaces the previous single heuristic proposal with a small action
    lattice that the admissibility layer can audit candidate-by-candidate.
    """

    def __init__(
        self,
        *,
        base_size: float = 1.0,
        edge_threshold: float = 0.05,
        levels: Iterable[float] = (1.0, 0.5, 0.25),
        capital_params: CapitalParams | None = None,
    ) -> None:
        self.base_size = float(base_size)
        self.edge_threshold = float(edge_threshold)
        self.levels = tuple(float(level) for level in levels if float(level) > 0.0)
        self.capital_params = capital_params or CapitalParams()

    def generate(self, observation: NashiObservation, state: NashiState) -> list[CandidateTransition]:
        candidates: list[CandidateTransition] = []

        signed_direction = self._direction(observation)
        actionability = _clamp(observation.context.actionability, 0.0, 1.0)
        hazard_density = _hazard_density(observation)
        hazard_drag = _clamp(
            0.70 * hazard_density + 0.20 * float(observation.context.hazard_trend) + 0.10 * float(observation.context.hazard_cooldown),
            0.0,
            1.0,
        )
        spread_regime = self._spread_regime(observation)
        desired_arrow = _clamp(
            max(state.last_arrow, actionability),
            0.0,
            1.0,
        )
        if signed_direction != 0:
            exposure_scale = _clamp(1.0 - 0.60 * hazard_drag, 0.25, 1.0)
            max_exposure = self.base_size * actionability * exposure_scale
            for level in self._ordered_levels(hazard_density, float(observation.context.hazard_trend)):
                signed_target = signed_direction * max_exposure * level
                proposal_economics = assess_proposal_economics(
                    capital_prev=state.capital,
                    exposure_prev=state.exposure,
                    proposed_exposure=float(signed_target),
                    edge=observation.context.edge,
                    actionability=actionability,
                    hazard=hazard_density,
                    spread_bps=observation.context.spread_bps,
                    drawdown=observation.context.drawdown,
                    edge_persistence=observation.context.edge_persistence,
                    edge_shock=observation.context.edge_shock,
                    microstructure_pressure=observation.context.microstructure_pressure,
                    family_cooldown=observation.context.family_cooldown,
                    params=self.capital_params,
                )
                candidates.append(
                    CandidateTransition(
                        candidate_id=f"{'long' if signed_direction > 0 else 'short'}_{level:.2f}",
                        signed_target_exposure=float(signed_target),
                        proposed_embedding=self._proposed_embedding(
                            observation.current_embedding,
                            signed_target,
                            desired_arrow=desired_arrow,
                        ),
                        rationale=f"{spread_regime}_edge_aligned_{level:.2f}",
                        family_hint=self._family_hint(state, actionability),
                        spread_regime=spread_regime,
                        priority=float(level),
                        proposal_economics=proposal_economics,
                    )
                )
        hold_economics = assess_proposal_economics(
            capital_prev=state.capital,
            exposure_prev=state.exposure,
            proposed_exposure=float(state.exposure),
            edge=observation.context.edge,
            actionability=actionability,
            hazard=hazard_density,
            spread_bps=observation.context.spread_bps,
            drawdown=observation.context.drawdown,
            edge_persistence=observation.context.edge_persistence,
            edge_shock=observation.context.edge_shock,
            microstructure_pressure=observation.context.microstructure_pressure,
            family_cooldown=observation.context.family_cooldown,
            params=self.capital_params,
        )
        candidates.append(
            CandidateTransition(
                candidate_id="hold",
                signed_target_exposure=float(state.exposure),
                proposed_embedding=observation.current_embedding,
                rationale="hold_current",
                family_hint=state.family_memory.last_family_hint,
                spread_regime=spread_regime,
                priority=0.0,
                proposal_economics=hold_economics,
            )
        )
        return candidates

    @staticmethod
    def _family_hint(state: NashiState, actionability: float) -> str:
        if state.family_memory.post_unwind_cooldown > 0:
            return "churn_guard"
        if state.family_memory.arrow_boundary_streak > 0:
            return "arrow_ladder"
        if state.family_memory.mdl_boundary_count > 0 and actionability < 0.5:
            return "mdl_tail_boundary"
        return "interior"

    def _ordered_levels(self, hazard_density: float, hazard_trend: float) -> tuple[float, ...]:
        ascending = tuple(sorted(self.levels))
        if hazard_density >= 0.65 or hazard_trend >= 0.35:
            return ascending
        if hazard_density >= 0.40 or hazard_trend >= 0.20:
            medium = [level for level in ascending if 0.25 < level < 1.0]
            probes = [level for level in ascending if level <= 0.25]
            full = [level for level in ascending if level >= 1.0]
            return tuple(medium + probes + full)
        return self.levels

    def _direction(self, observation: NashiObservation) -> int:
        if abs(observation.context.edge) >= self.edge_threshold:
            return _sign(observation.context.edge)
        return int(observation.state_signal)

    @staticmethod
    def _proposed_embedding(
        current: ClosureEmbedding,
        signed_target_exposure: float,
        *,
        desired_arrow: float,
    ) -> ClosureEmbedding:
        next_dnorm = float(np.tanh(signed_target_exposure))
        delta_dnorm = abs(next_dnorm - current.v_dnorm)
        arrow_spend = max(0.0, desired_arrow - current.v_arrow)
        next_depth = _clamp(current.v_depth - delta_dnorm - arrow_spend, -1.0, 1.0)
        return ClosureEmbedding(
            v_pnorm=current.v_pnorm,
            v_dnorm=next_dnorm,
            v_depth=next_depth,
            v_arrow=desired_arrow,
        )

    @staticmethod
    def _spread_regime(observation: NashiObservation) -> str:
        if observation.context.cost_survival_ratio <= 0.0:
            return "microstructure_kills_edge"
        if observation.context.microstructure_pressure >= 0.75:
            return "spread_stressed"
        return "spread_clear"

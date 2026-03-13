from __future__ import annotations

from dataclasses import dataclass
import math

from .state import CoarseState


@dataclass(frozen=True)
class ActionWeights:
    w_return: float = 1.35
    w_alignment: float = 0.6
    w_action: float = 0.45
    w_contract: float = 0.5
    w_branch: float = 0.22
    w_diffusion: float = 0.22
    w_stress: float = 0.42
    w_drawdown: float = 0.48
    w_churn: float = 0.22
    w_inventory: float = 0.12


@dataclass(frozen=True)
class ActionScoreBreakdown:
    signed_return: float
    alignment: float
    actionability: float
    contraction_gain: float
    branch_cost: float
    diffusion_cost: float
    stress_cost: float
    drawdown_cost: float
    churn_cost: float
    inventory_cost: float
    reward_block: float
    penalty_block: float
    score_mode: str
    total: float


class ActionFunctional:
    """
    Score one transition in the coarse-state beam tree.
    """

    def __init__(
        self,
        weights: ActionWeights | None = None,
        score_mode: str = "ratio",
        score_scale: float = 1.0,
    ):
        self.weights = weights or ActionWeights()
        self.score_mode = score_mode
        self.score_scale = float(score_scale)

    def score_transition(
        self,
        current: CoarseState,
        nxt: CoarseState,
        next_exposure: float,
        step_return: float,
        step_branch_risk: float,
        step_diffusion_risk: float,
    ) -> ActionScoreBreakdown:
        weights = self.weights
        signed_return = step_return * next_exposure
        alignment = max(0.0, nxt.triadic_bias * next_exposure)
        actionability = nxt.actionability * abs(next_exposure)
        contraction_gain = max(0.0, nxt.contraction - current.contraction)
        branch_cost = max(0.0, step_branch_risk)
        diffusion_cost = max(nxt.diffusion, step_diffusion_risk)
        stress_cost = nxt.stress * abs(next_exposure)
        drawdown_cost = nxt.drawdown * abs(next_exposure)
        churn_cost = abs(next_exposure - current.current_exposure)
        inventory_cost = abs(next_exposure) * max(0.0, nxt.diffusion - nxt.contraction)
        reward_block = (
            weights.w_return * signed_return
            + weights.w_alignment * alignment
            + weights.w_action * actionability
            + weights.w_contract * contraction_gain
        )
        penalty_block = (
            weights.w_branch * branch_cost
            + weights.w_diffusion * diffusion_cost
            + weights.w_stress * stress_cost
            + weights.w_drawdown * drawdown_cost
            + weights.w_churn * churn_cost
            + weights.w_inventory * inventory_cost
        )
        if self.score_mode == "ratio":
            total = reward_block / (1.0 + penalty_block)
        elif self.score_mode == "scaled_diff":
            total = self.score_scale * (reward_block - penalty_block)
        elif self.score_mode == "logistic":
            total = math.tanh(self.score_scale * (reward_block - penalty_block))
        else:
            raise ValueError(f"unknown score_mode: {self.score_mode}")
        return ActionScoreBreakdown(
            signed_return=signed_return,
            alignment=alignment,
            actionability=actionability,
            contraction_gain=contraction_gain,
            branch_cost=branch_cost,
            diffusion_cost=diffusion_cost,
            stress_cost=stress_cost,
            drawdown_cost=drawdown_cost,
            churn_cost=churn_cost,
            inventory_cost=inventory_cost,
            reward_block=reward_block,
            penalty_block=penalty_block,
            score_mode=self.score_mode,
            total=total,
        )

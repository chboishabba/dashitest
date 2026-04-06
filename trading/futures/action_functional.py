from __future__ import annotations

from dataclasses import dataclass
import math

from .state import CoarseState

import numpy as np


@dataclass(frozen=True)
class ActionWeights:
    w_return: float = 1.35
    w_alignment: float = 0.6
    w_action: float = 0.45
    w_contract: float = 0.5
    w_branch: float = 0.22
    w_diffusion: float = 0.22
    w_stress: float = 0.42
    # Ablation: combine correlated uncertainty terms (branch/diffusion/stress) into one block.
    w_uncertainty: float = (0.22 + 0.22 + 0.42) / 3.0
    w_drawdown: float = 0.48
    w_churn: float = 0.22
    w_inventory: float = 0.12


@dataclass(frozen=True)
class ActionScoreBreakdown:
    signed_return: float
    return_mode: str
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
    penalty_mode: str
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
        penalty_mode: str = "explicit",
        return_mode: str = "directional",
    ):
        self.weights = weights or ActionWeights()
        self.score_mode = score_mode
        self.score_scale = float(score_scale)
        self.penalty_mode = str(penalty_mode)
        self.return_mode = str(return_mode)

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
        if self.return_mode == "directional":
            signed_return = step_return * next_exposure
        elif self.return_mode == "abs":
            signed_return = abs(step_return) * abs(next_exposure)
        else:
            raise ValueError(f"unknown return_mode: {self.return_mode}")
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
        if self.penalty_mode == "explicit":
            penalty_block = (
                weights.w_branch * branch_cost
                + weights.w_diffusion * diffusion_cost
                + weights.w_stress * stress_cost
                + weights.w_drawdown * drawdown_cost
                + weights.w_churn * churn_cost
                + weights.w_inventory * inventory_cost
            )
        elif self.penalty_mode == "merged_uncertainty":
            uncertainty_cost = branch_cost + diffusion_cost + stress_cost
            penalty_block = (
                weights.w_uncertainty * uncertainty_cost
                + weights.w_drawdown * drawdown_cost
                + weights.w_churn * churn_cost
                + weights.w_inventory * inventory_cost
            )
        else:
            raise ValueError(f"unknown penalty_mode: {self.penalty_mode}")
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
            return_mode=self.return_mode,
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
            penalty_mode=self.penalty_mode,
            total=total,
        )

    def score_transition_batch(
        self,
        *,
        current_contraction: np.ndarray,
        current_exposure: np.ndarray,
        nxt_triadic_bias: np.ndarray,
        nxt_actionability: np.ndarray,
        nxt_contraction: np.ndarray,
        nxt_diffusion: np.ndarray,
        nxt_stress: np.ndarray,
        nxt_drawdown: np.ndarray,
        next_exposure: np.ndarray,
        step_return: np.ndarray,
        step_branch_risk: np.ndarray,
        step_diffusion_risk: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Vectorized transition scoring.

        Returns a dict of component arrays so callers can either:
        - consume only `total` for pruning, or
        - materialize an `ActionScoreBreakdown` for the chosen nodes.
        """
        weights = self.weights

        current_contraction = np.asarray(current_contraction, dtype=np.float32)
        current_exposure = np.asarray(current_exposure, dtype=np.float32)
        nxt_triadic_bias = np.asarray(nxt_triadic_bias, dtype=np.float32)
        nxt_actionability = np.asarray(nxt_actionability, dtype=np.float32)
        nxt_contraction = np.asarray(nxt_contraction, dtype=np.float32)
        nxt_diffusion = np.asarray(nxt_diffusion, dtype=np.float32)
        nxt_stress = np.asarray(nxt_stress, dtype=np.float32)
        nxt_drawdown = np.asarray(nxt_drawdown, dtype=np.float32)
        next_exposure = np.asarray(next_exposure, dtype=np.float32)
        step_return = np.asarray(step_return, dtype=np.float32)
        step_branch_risk = np.asarray(step_branch_risk, dtype=np.float32)
        step_diffusion_risk = np.asarray(step_diffusion_risk, dtype=np.float32)

        abs_next_exposure = np.abs(next_exposure)
        if self.return_mode == "directional":
            signed_return = step_return * next_exposure
        elif self.return_mode == "abs":
            signed_return = np.abs(step_return) * abs_next_exposure
        else:
            raise ValueError(f"unknown return_mode: {self.return_mode}")

        alignment = np.maximum(0.0, nxt_triadic_bias * next_exposure)
        actionability = nxt_actionability * abs_next_exposure
        contraction_gain = np.maximum(0.0, nxt_contraction - current_contraction)
        branch_cost = np.maximum(0.0, step_branch_risk)
        diffusion_cost = np.maximum(nxt_diffusion, step_diffusion_risk)
        stress_cost = nxt_stress * abs_next_exposure
        drawdown_cost = nxt_drawdown * abs_next_exposure
        churn_cost = np.abs(next_exposure - current_exposure)
        inventory_cost = abs_next_exposure * np.maximum(0.0, nxt_diffusion - nxt_contraction)

        reward_block = (
            weights.w_return * signed_return
            + weights.w_alignment * alignment
            + weights.w_action * actionability
            + weights.w_contract * contraction_gain
        )
        if self.penalty_mode == "explicit":
            penalty_block = (
                weights.w_branch * branch_cost
                + weights.w_diffusion * diffusion_cost
                + weights.w_stress * stress_cost
                + weights.w_drawdown * drawdown_cost
                + weights.w_churn * churn_cost
                + weights.w_inventory * inventory_cost
            )
        elif self.penalty_mode == "merged_uncertainty":
            uncertainty_cost = branch_cost + diffusion_cost + stress_cost
            penalty_block = (
                weights.w_uncertainty * uncertainty_cost
                + weights.w_drawdown * drawdown_cost
                + weights.w_churn * churn_cost
                + weights.w_inventory * inventory_cost
            )
        else:
            raise ValueError(f"unknown penalty_mode: {self.penalty_mode}")

        if self.score_mode == "ratio":
            total = reward_block / (1.0 + penalty_block)
        elif self.score_mode == "scaled_diff":
            total = self.score_scale * (reward_block - penalty_block)
        elif self.score_mode == "logistic":
            total = np.tanh(self.score_scale * (reward_block - penalty_block))
        else:
            raise ValueError(f"unknown score_mode: {self.score_mode}")

        return {
            "signed_return": signed_return,
            "alignment": alignment,
            "actionability": actionability,
            "contraction_gain": contraction_gain,
            "branch_cost": branch_cost,
            "diffusion_cost": diffusion_cost,
            "stress_cost": stress_cost,
            "drawdown_cost": drawdown_cost,
            "churn_cost": churn_cost,
            "inventory_cost": inventory_cost,
            "reward_block": reward_block,
            "penalty_block": penalty_block,
            "total": total,
        }

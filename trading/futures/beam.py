from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

try:
    from trading.intent import Intent
except ModuleNotFoundError:
    from intent import Intent

from .action_functional import ActionFunctional, ActionScoreBreakdown
from .basin import BasinClassifier, BasinMass, normalized_entropy
from .state import CoarseState


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


@dataclass(frozen=True)
class BeamConfig:
    horizon: int = 4
    beam_width: int = 12
    label_min_quota: int = 1
    label_quota_frac: float = 0.25
    flat_return_band: float = 0.02
    flat_cost_floor: float = 0.0
    exposure_step: float = 0.25
    probability_floor: float = 1e-3
    action_grid: tuple[int, ...] = (-1, 0, 1)


@dataclass(frozen=True)
class TransitionCandidate:
    label: str
    probability: float
    next_state: CoarseState
    step_return: float
    step_branch_risk: float
    step_diffusion_risk: float


@dataclass(frozen=True)
class BeamSummary:
    entropy: float
    best_score: float
    long_mass: float
    short_mass: float
    flat_mass: float
    curvature: float
    flat_distance: float
    best_terminal_return: float
    best_terminal_risk: float
    contraction: float
    diffusion: float

    def to_log_fields(self) -> dict[str, float]:
        return {
            "beam_entropy": self.entropy,
            "beam_best_score": self.best_score,
            "beam_long_mass": self.long_mass,
            "beam_short_mass": self.short_mass,
            "beam_flat_mass": self.flat_mass,
            "beam_curvature": self.curvature,
            "beam_flat_distance": self.flat_distance,
            "beam_best_return": self.best_terminal_return,
            "beam_best_risk": self.best_terminal_risk,
            "beam_contraction": self.contraction,
            "beam_diffusion": self.diffusion,
        }


@dataclass(frozen=True)
class BeamDecisionDiagnostics:
    hold_reason_primary: str
    hold_entropy: int
    hold_score: int
    hold_basin: int
    hold_flat_basin: int
    hold_curvature: int
    entropy_value: float
    entropy_threshold: float
    curvature_value: float
    curvature_threshold: float
    score_value: float
    score_threshold: float
    score_reward: float
    score_penalty: float
    score_adjusted: float
    score_mode: str
    directional_mass: float
    directional_margin: float
    flat_mass: float
    selected_direction: int
    gating_mode: str
    kernel_mode: str
    kernel_fallback_used: int
    kernel_source_count: int
    kernel_bucket_count: int
    kernel_label_count_long: int
    kernel_label_count_short: int
    kernel_label_count_flat: int
    kernel_label_count_stress: int
    kernel_lambda: float
    kernel_asset_count: int
    kernel_global_count: int
    beam_step_counts: str

    def to_log_fields(self) -> dict[str, float | int | str]:
        return {
            "shadow_hold_reason_primary": self.hold_reason_primary,
            "shadow_hold_entropy": self.hold_entropy,
            "shadow_hold_score": self.hold_score,
            "shadow_hold_basin": self.hold_basin,
            "shadow_hold_flat_basin": self.hold_flat_basin,
            "shadow_hold_curvature": self.hold_curvature,
            "shadow_entropy_value": self.entropy_value,
            "shadow_entropy_threshold": self.entropy_threshold,
            "shadow_curvature_value": self.curvature_value,
            "shadow_curvature_threshold": self.curvature_threshold,
            "shadow_score_value": self.score_value,
            "shadow_score_threshold": self.score_threshold,
            "shadow_score_reward": self.score_reward,
            "shadow_score_penalty": self.score_penalty,
            "shadow_score_adjusted": self.score_adjusted,
            "shadow_score_mode": self.score_mode,
            "shadow_directional_mass": self.directional_mass,
            "shadow_directional_margin": self.directional_margin,
            "shadow_kernel_mode": self.kernel_mode,
            "shadow_kernel_fallback_used": self.kernel_fallback_used,
            "shadow_kernel_source_count": self.kernel_source_count,
            "shadow_kernel_bucket_count": self.kernel_bucket_count,
            "shadow_kernel_label_count_long": self.kernel_label_count_long,
            "shadow_kernel_label_count_short": self.kernel_label_count_short,
            "shadow_kernel_label_count_flat": self.kernel_label_count_flat,
            "shadow_kernel_label_count_stress": self.kernel_label_count_stress,
            "shadow_kernel_lambda": self.kernel_lambda,
            "shadow_kernel_asset_count": self.kernel_asset_count,
            "shadow_kernel_global_count": self.kernel_global_count,
            "shadow_gating_mode": self.gating_mode,
            "shadow_beam_step_counts": self.beam_step_counts,
        }


@dataclass(frozen=True)
class BeamPolicyStep:
    intent: Intent
    summary: BeamSummary
    diagnostics: BeamDecisionDiagnostics


@dataclass
class BeamNode:
    state: CoarseState
    depth: int
    cumulative_score: float
    cumulative_log_prob: float
    action_history: list[int] = field(default_factory=list)
    exposure_history: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    breakdowns: list[ActionScoreBreakdown] = field(default_factory=list)
    return_history: list[float] = field(default_factory=list)
    branch_risk_history: list[float] = field(default_factory=list)
    diffusion_risk_history: list[float] = field(default_factory=list)
    label_history: list[str] = field(default_factory=list)
    last_label: str = "flat"

    @property
    def terminal_exposure(self) -> float:
        if self.exposure_history:
            return self.exposure_history[-1]
        return self.state.current_exposure

    @property
    def weight(self) -> float:
        return math.exp(self.cumulative_log_prob)

    @property
    def predicted_return(self) -> float:
        return sum(self.return_history)

    @property
    def predicted_risk(self) -> float:
        return sum(self.branch_risk_history) + sum(self.diffusion_risk_history)


class HeuristicTransitionModel:
    """
    Small, inspectable transition generator for future-tree experiments.
    """

    def __init__(self, continuation_prob: float = 0.55, mean_revert_prob: float = 0.25, stress_prob: float = 0.20):
        probs = [continuation_prob, mean_revert_prob, stress_prob]
        total = sum(probs)
        self.continuation_prob = continuation_prob / total
        self.mean_revert_prob = mean_revert_prob / total
        self.stress_prob = stress_prob / total

    def expand(self, state: CoarseState, action: int, next_exposure: float) -> Iterable[TransitionCandidate]:
        directional_push = 0.6 * state.drift + 0.4 * state.edge
        alignment = state.triadic_bias * action
        continuation_return = directional_push + 0.3 * alignment
        revert_return = -0.5 * directional_push + 0.1 * alignment
        stress_return = continuation_return - 0.8 * state.stress - 0.4 * abs(next_exposure)

        base_contraction = state.contraction + 0.1 * max(alignment, 0.0) - 0.08 * abs(next_exposure - state.current_exposure)
        base_diffusion = state.diffusion + 0.1 * (1.0 - state.actionability)
        cont_state = CoarseState(
            drift=_clip(0.7 * state.drift + 0.3 * action, -1.0, 1.0),
            triadic_bias=_clip(0.8 * state.triadic_bias + 0.2 * action, -1.0, 1.0),
            actionability=_clip(state.actionability + 0.1 * max(alignment, 0.0) - 0.05 * state.stress, 0.0, 1.0),
            stress=_clip(0.9 * state.stress + 0.05 * abs(next_exposure - state.current_exposure), 0.0, 1.0),
            edge=_clip(0.8 * state.edge + 0.2 * action, -1.0, 1.0),
            contraction=_clip(base_contraction, 0.0, 1.0),
            diffusion=_clip(base_diffusion - 0.1 * max(alignment, 0.0), 0.0, 1.0),
            drawdown=_clip(max(0.0, state.drawdown - 0.05 * max(continuation_return, 0.0)), 0.0, 1.0),
            current_exposure=next_exposure,
        )
        revert_state = CoarseState(
            drift=_clip(0.6 * state.drift - 0.25 * action, -1.0, 1.0),
            triadic_bias=_clip(0.7 * state.triadic_bias - 0.15 * action, -1.0, 1.0),
            actionability=_clip(state.actionability - 0.1 * abs(action), 0.0, 1.0),
            stress=_clip(state.stress + 0.05 * abs(action), 0.0, 1.0),
            edge=_clip(0.5 * state.edge - 0.2 * action, -1.0, 1.0),
            contraction=_clip(base_contraction - 0.1, 0.0, 1.0),
            diffusion=_clip(base_diffusion + 0.1, 0.0, 1.0),
            drawdown=_clip(state.drawdown + 0.05 * abs(next_exposure), 0.0, 1.0),
            current_exposure=next_exposure,
        )
        stress_state = CoarseState(
            drift=_clip(0.3 * state.drift - 0.2 * action, -1.0, 1.0),
            triadic_bias=_clip(0.5 * state.triadic_bias, -1.0, 1.0),
            actionability=_clip(state.actionability - 0.2, 0.0, 1.0),
            stress=_clip(state.stress + 0.25 + 0.1 * abs(action), 0.0, 1.0),
            edge=_clip(state.edge - 0.4 * abs(action), -1.0, 1.0),
            contraction=_clip(base_contraction - 0.2, 0.0, 1.0),
            diffusion=_clip(base_diffusion + 0.25, 0.0, 1.0),
            drawdown=_clip(state.drawdown + 0.15 * abs(next_exposure), 0.0, 1.0),
            current_exposure=next_exposure,
        )
        return (
            TransitionCandidate(
                "continue",
                self.continuation_prob,
                cont_state,
                continuation_return,
                0.10 + 0.10 * abs(alignment),
                0.15 + 0.25 * cont_state.diffusion,
            ),
            TransitionCandidate(
                "revert",
                self.mean_revert_prob,
                revert_state,
                revert_return,
                0.15 + 0.10 * abs(action),
                0.25 + 0.25 * revert_state.diffusion,
            ),
            TransitionCandidate(
                "stress",
                self.stress_prob,
                stress_state,
                stress_return,
                0.35 + 0.25 * stress_state.stress,
                0.30 + 0.35 * stress_state.diffusion,
            ),
        )


class FutureBeamSearch:
    def __init__(
        self,
        action_functional: ActionFunctional,
        transition_model: HeuristicTransitionModel,
        beam_config: BeamConfig | None = None,
    ):
        self.action_functional = action_functional
        self.transition_model = transition_model
        self.beam_config = beam_config or BeamConfig()
        self.basin_classifier = BasinClassifier(
            flat_return_band=self.beam_config.flat_return_band,
            flat_cost_floor=self.beam_config.flat_cost_floor,
        )
        self.last_step_counts: list[dict[str, int]] = []

    def search(self, initial_state: CoarseState) -> list[BeamNode]:
        beam = [
            BeamNode(
                state=initial_state,
                depth=0,
                cumulative_score=0.0,
                cumulative_log_prob=0.0,
            )
        ]
        cfg = self.beam_config
        for depth in range(cfg.horizon):
            next_beam: list[BeamNode] = []
            for node in beam:
                for action in cfg.action_grid:
                    next_exposure = _clip(node.state.current_exposure + action * cfg.exposure_step, -1.0, 1.0)
                    for candidate in self.transition_model.expand(node.state, action, next_exposure):
                        if candidate.probability < cfg.probability_floor:
                            continue
                        breakdown = self.action_functional.score_transition(
                            current=node.state,
                            nxt=candidate.next_state,
                            next_exposure=next_exposure,
                            step_return=candidate.step_return,
                            step_branch_risk=candidate.step_branch_risk,
                            step_diffusion_risk=candidate.step_diffusion_risk,
                        )
                        next_beam.append(
                            BeamNode(
                                state=candidate.next_state,
                                depth=depth + 1,
                                cumulative_score=node.cumulative_score + breakdown.total,
                                cumulative_log_prob=node.cumulative_log_prob + math.log(candidate.probability),
                                action_history=node.action_history + [action],
                                exposure_history=node.exposure_history + [next_exposure],
                                labels=node.labels + [candidate.label],
                                breakdowns=node.breakdowns + [breakdown],
                                return_history=node.return_history + [candidate.step_return],
                                branch_risk_history=node.branch_risk_history + [candidate.step_branch_risk],
                                diffusion_risk_history=node.diffusion_risk_history + [candidate.step_diffusion_risk],
                                label_history=node.label_history + [candidate.label],
                                last_label=candidate.label,
                            )
                        )
            next_beam = self._select_diverse_beam(next_beam, cfg)
            beam = next_beam
            self.last_step_counts.append(self._count_labels(beam, depth=depth + 1))
            if not beam:
                break
        return beam

    @staticmethod
    def _label_bucket(label: str) -> str:
        lower = label.lower()
        if "flat" in lower:
            return "flat"
        if "down" in lower or "short" in lower:
            return "short"
        if "stress" in lower:
            return "flat"
        return "long"

    def _count_labels(self, beam: list[BeamNode], *, depth: int) -> dict[str, int]:
        counts = {"depth": depth, "long": 0, "short": 0, "flat": 0}
        for node in beam:
            bucket = self._label_bucket(node.last_label)
            counts[bucket] += 1
        return counts

    def _select_diverse_beam(self, nodes: list[BeamNode], cfg: BeamConfig) -> list[BeamNode]:
        if not nodes:
            return []
        nodes_sorted = sorted(nodes, key=lambda item: (item.cumulative_score, item.cumulative_log_prob), reverse=True)
        label_buckets: dict[str, list[BeamNode]] = {"long": [], "short": [], "flat": []}
        for node in nodes_sorted:
            label = self._label_bucket(node.last_label)
            label_buckets[label].append(node)
        beam_width = int(cfg.beam_width)
        quota = max(int(cfg.label_min_quota), int(math.floor(cfg.label_quota_frac * beam_width)))
        selected: list[BeamNode] = []
        for label in ("long", "short", "flat"):
            bucket = label_buckets[label]
            if bucket:
                selected.extend(bucket[: min(quota, len(bucket))])
        if len(selected) < beam_width:
            selected_set = {id(node) for node in selected}
            for node in nodes_sorted:
                if id(node) in selected_set:
                    continue
                selected.append(node)
                if len(selected) >= beam_width:
                    break
        return selected[:beam_width]

    def summarize(self, beam: list[BeamNode]) -> BeamSummary:
        if not beam:
            return BeamSummary(0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0)
        basin: BasinMass = self.basin_classifier.classify(beam)
        best = beam[0]
        total_weight = sum(node.weight for node in beam)
        if total_weight <= 0:
            contraction = 0.0
            diffusion = 1.0
        else:
            contraction = sum(node.weight * node.state.contraction for node in beam) / total_weight
            diffusion = sum(node.weight * node.state.diffusion for node in beam) / total_weight
        curvature = basin.long_mass + basin.short_mass - basin.flat_mass
        flat_distance = math.sqrt(
            basin.long_mass ** 2 + basin.short_mass ** 2 + (basin.flat_mass - 1.0) ** 2
        )
        return BeamSummary(
            entropy=normalized_entropy(node.weight for node in beam),
            best_score=best.cumulative_score,
            long_mass=basin.long_mass,
            short_mass=basin.short_mass,
            flat_mass=basin.flat_mass,
            curvature=curvature,
            flat_distance=flat_distance,
            best_terminal_return=best.predicted_return,
            best_terminal_risk=best.predicted_risk,
            contraction=contraction,
            diffusion=diffusion,
        )


class BeamIntentPolicy:
    """
    Convert beam-search terminal states back into the existing Intent contract.
    """

    def __init__(
        self,
        symbol: str,
        search: FutureBeamSearch,
        exposure_cap: float = 1.0,
        entropy_threshold: float = 0.96,
        entropy_gate_mode: str = "logistic",
        entropy_gate_center: float = 0.955,
        entropy_gate_tau: float = 0.01,
        entropy_score_floor: float = 0.05,
        entropy_margin_floor: float = 0.20,
        flat_mass_threshold: float = 0.50,
        flat_score_threshold: float = 0.10,
        score_threshold: float = -0.05,
        directional_mass_threshold: float = 0.55,
        directional_margin_threshold: float = 0.15,
        curvature_threshold: float = 0.0,
        score_curvature_weight: bool = False,
        gating_mode: str = "lex",
    ):
        self.symbol = symbol
        self.search = search
        self.exposure_cap = float(exposure_cap)
        self.entropy_threshold = float(entropy_threshold)
        self.entropy_gate_mode = str(entropy_gate_mode)
        self.entropy_gate_center = float(entropy_gate_center)
        self.entropy_gate_tau = max(abs(float(entropy_gate_tau)), 1e-6)
        self.entropy_score_floor = float(entropy_score_floor)
        self.entropy_margin_floor = float(entropy_margin_floor)
        self.flat_mass_threshold = float(flat_mass_threshold)
        self.flat_score_threshold = float(flat_score_threshold)
        self.score_threshold = float(score_threshold)
        self.directional_mass_threshold = float(directional_mass_threshold)
        self.directional_margin_threshold = float(directional_margin_threshold)
        self.curvature_threshold = float(curvature_threshold)
        self.score_curvature_weight = bool(score_curvature_weight)
        self.gating_mode = gating_mode

    def _entropy_weight(self, entropy: float) -> float:
        if self.entropy_gate_mode == "hard":
            return 0.0 if entropy > self.entropy_threshold else 1.0
        center = self.entropy_gate_center
        tau = self.entropy_gate_tau
        z = (float(entropy) - center) / tau
        if z >= 50.0:
            return 0.0
        if z <= -50.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(z))

    def _kernel_metadata(self) -> dict[str, int | float | str]:
        metadata = getattr(self.search.transition_model, "metadata", {}) or {}
        return {
            "kernel_mode": str(metadata.get("kernel_mode", "heuristic")),
            "kernel_fallback_used": int(bool(metadata.get("fallback_used", False))),
            "kernel_source_count": int(metadata.get("source_count", 0)),
            "kernel_bucket_count": int(metadata.get("bucket_count", 0)),
            "kernel_label_count_long": int(metadata.get("label_count_long", 0)),
            "kernel_label_count_short": int(metadata.get("label_count_short", 0)),
            "kernel_label_count_flat": int(metadata.get("label_count_flat", 0)),
            "kernel_label_count_stress": int(metadata.get("label_count_stress", 0)),
            "kernel_lambda": float(metadata.get("kernel_lambda", 0.0)),
            "kernel_asset_count": int(metadata.get("kernel_asset_count", 0)),
            "kernel_global_count": int(metadata.get("kernel_global_count", 0)),
        }

    def _diagnostics(
        self,
        *,
        summary: BeamSummary,
        hold_reason_primary: str,
        selected_direction: int,
        score_reward: float,
        score_penalty: float,
        score_adjusted: float,
    ) -> BeamDecisionDiagnostics:
        kernel = self._kernel_metadata()
        directional_mass = max(summary.long_mass, summary.short_mass)
        directional_margin = abs(summary.long_mass - summary.short_mass)
        return BeamDecisionDiagnostics(
            hold_reason_primary=hold_reason_primary,
            hold_entropy=int(hold_reason_primary == "entropy"),
            hold_score=int(hold_reason_primary == "score"),
            hold_basin=int(hold_reason_primary == "basin"),
            hold_flat_basin=int(hold_reason_primary == "flat_basin"),
            hold_curvature=int(hold_reason_primary == "curvature"),
            entropy_value=summary.entropy,
            entropy_threshold=self.entropy_threshold,
            curvature_value=summary.curvature,
            curvature_threshold=self.curvature_threshold,
            score_value=summary.best_score,
            score_threshold=self.score_threshold,
            score_reward=score_reward,
            score_penalty=score_penalty,
            score_adjusted=score_adjusted,
            score_mode=str(getattr(self.search.action_functional, "score_mode", "unknown")),
            directional_mass=directional_mass,
            directional_margin=directional_margin,
            flat_mass=summary.flat_mass,
            selected_direction=selected_direction,
            gating_mode=self.gating_mode,
            kernel_mode=str(kernel["kernel_mode"]),
            kernel_fallback_used=int(kernel["kernel_fallback_used"]),
            kernel_source_count=int(kernel["kernel_source_count"]),
            kernel_bucket_count=int(kernel["kernel_bucket_count"]),
            kernel_label_count_long=int(kernel["kernel_label_count_long"]),
            kernel_label_count_short=int(kernel["kernel_label_count_short"]),
            kernel_label_count_flat=int(kernel["kernel_label_count_flat"]),
            kernel_label_count_stress=int(kernel["kernel_label_count_stress"]),
            kernel_lambda=float(kernel.get("kernel_lambda", 0.0)),
            kernel_asset_count=int(kernel.get("kernel_asset_count", 0)),
            kernel_global_count=int(kernel.get("kernel_global_count", 0)),
            beam_step_counts=self._format_step_counts(),
        )

    def _format_step_counts(self) -> str:
        steps = getattr(self.search, "last_step_counts", []) or []
        if not steps:
            return ""
        parts = []
        for step in steps:
            parts.append(
                f"d{step.get('depth', 0)}:{step.get('long', 0)},{step.get('short', 0)},{step.get('flat', 0)}"
            )
        return "|".join(parts)

    def _hold_step(
        self,
        *,
        ts: int,
        initial_state: CoarseState,
        summary: BeamSummary,
        hold_reason_primary: str,
        reason: str,
        selected_direction: int = 0,
        actionability: float | None = None,
        score_reward: float = 0.0,
        score_penalty: float = 0.0,
        score_adjusted: float = 0.0,
    ) -> BeamPolicyStep:
        intent = Intent(
            ts=int(ts),
            symbol=self.symbol,
            direction=0,
            target_exposure=0.0,
            urgency=0.0,
            ttl_ms=500,
            hold=True,
            actionability=initial_state.actionability if actionability is None else float(actionability),
            reason=reason,
        )
        return BeamPolicyStep(
            intent=intent,
            summary=summary,
            diagnostics=self._diagnostics(
                summary=summary,
                hold_reason_primary=hold_reason_primary,
                selected_direction=selected_direction,
                score_reward=score_reward,
                score_penalty=score_penalty,
                score_adjusted=score_adjusted,
            ),
        )

    def step(self, ts: int, initial_state: CoarseState) -> BeamPolicyStep:
        beam = self.search.search(initial_state)
        summary = self.search.summarize(beam)
        if not beam:
            return self._hold_step(
                ts=ts,
                initial_state=initial_state,
                summary=summary,
                hold_reason_primary="empty",
                reason="beam_empty",
                actionability=0.0,
            )
        best = beam[0]
        score_reward = 0.0
        score_penalty = 0.0
        if best.breakdowns:
            last_breakdown = best.breakdowns[-1]
            score_reward = float(getattr(last_breakdown, "reward_block", 0.0))
            score_penalty = float(getattr(last_breakdown, "penalty_block", 0.0))
        directional_mass = max(summary.long_mass, summary.short_mass)
        directional_margin = abs(summary.long_mass - summary.short_mass)
        score_adjusted = best.cumulative_score * (1.0 - summary.flat_mass)
        if self.score_curvature_weight:
            score_adjusted *= summary.curvature
        score_adjusted *= self._entropy_weight(summary.entropy)
        if self.gating_mode == "score_only":
            if score_adjusted <= self.score_threshold:
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="score",
                    reason=f"beam_score_hold score={best.cumulative_score:.3f}",
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
        elif self.gating_mode == "lex":
            if self.curvature_threshold > 0.0 and summary.curvature < self.curvature_threshold:
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="curvature",
                    reason=f"beam_curvature_hold curvature={summary.curvature:.3f}",
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
            if self.entropy_gate_mode == "hard" and summary.entropy > self.entropy_threshold:
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="entropy",
                    reason=f"beam_entropy_hold entropy={summary.entropy:.3f} margin={directional_margin:.3f}",
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
            if directional_mass < self.directional_mass_threshold or directional_margin < self.directional_margin_threshold:
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="basin",
                    reason=(
                        f"beam_basin_hold long={summary.long_mass:.3f} "
                        f"short={summary.short_mass:.3f} flat={summary.flat_mass:.3f}"
                    ),
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
            if score_adjusted <= self.score_threshold:
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="score",
                    reason=f"beam_score_hold score={best.cumulative_score:.3f}",
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
        else:
            if (
                self.entropy_gate_mode == "hard"
                and
                summary.entropy > self.entropy_threshold
                and directional_margin < self.entropy_margin_floor
                and score_adjusted < self.entropy_score_floor
            ):
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="entropy",
                    reason=(
                        f"beam_entropy_hold entropy={summary.entropy:.3f} "
                        f"margin={directional_margin:.3f} score={score_adjusted:.3f}"
                    ),
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
            if summary.flat_mass >= self.flat_mass_threshold and score_adjusted < self.flat_score_threshold:
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="flat_basin",
                    reason=f"beam_flat_hold score={best.cumulative_score:.3f} flat_mass={summary.flat_mass:.3f}",
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
            if score_adjusted <= self.score_threshold:
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="score",
                    reason=f"beam_score_hold score={best.cumulative_score:.3f}",
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
            if directional_mass < self.directional_mass_threshold or directional_margin < self.directional_margin_threshold:
                return self._hold_step(
                    ts=ts,
                    initial_state=initial_state,
                    summary=summary,
                    hold_reason_primary="basin",
                    reason=(
                        f"beam_basin_hold long={summary.long_mass:.3f} "
                        f"short={summary.short_mass:.3f} flat={summary.flat_mass:.3f}"
                    ),
                    score_reward=score_reward,
                    score_penalty=score_penalty,
                    score_adjusted=score_adjusted,
                )
        if summary.long_mass > summary.short_mass:
            direction = 1
        else:
            direction = -1
        signed_best = next((node for node in beam if (node.predicted_return > 0) == (direction > 0)), best)
        target_exposure = min(abs(signed_best.terminal_exposure), self.exposure_cap)
        urgency = _clip(0.35 + 0.45 * initial_state.actionability + 0.2 * max(signed_best.cumulative_score, 0.0), 0.0, 1.0)
        return BeamPolicyStep(
            intent=Intent(
                ts=int(ts),
                symbol=self.symbol,
                direction=direction,
                target_exposure=target_exposure,
                urgency=urgency,
                ttl_ms=500,
                hold=False,
                actionability=initial_state.actionability,
                reason=(
                    f"beam_policy score={signed_best.cumulative_score:.3f} "
                    f"long={summary.long_mass:.3f} short={summary.short_mass:.3f} flat={summary.flat_mass:.3f}"
                ),
            ),
            summary=summary,
            diagnostics=self._diagnostics(
                summary=summary,
                hold_reason_primary="act",
                selected_direction=direction,
                score_reward=score_reward,
                score_penalty=score_penalty,
                score_adjusted=score_adjusted,
            ),
        )

    def select_intent(self, ts: int, initial_state: CoarseState) -> Intent:
        return self.step(ts=ts, initial_state=initial_state).intent

from __future__ import annotations

from dataclasses import dataclass


def clip_unit(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def sign_unit(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass(frozen=True)
class ObservationSnapshot:
    price_return: float
    trend_return: float
    realized_vol: float
    triadic_state: int
    actionability: float
    stress: float
    edge: float
    drawdown: float
    current_exposure: float


@dataclass(frozen=True)
class CoarseState:
    drift: float
    triadic_bias: float
    actionability: float
    stress: float
    edge: float
    contraction: float
    diffusion: float
    drawdown: float
    current_exposure: float

    def action_direction(self) -> int:
        return sign_unit(self.triadic_bias if self.actionability >= 0.5 else self.edge)


class CoarseStateEstimator:
    """
    Shared coarse-state estimator for baseline and futures-side policies.
    """

    def __init__(self, vol_scale: float = 0.02, drawdown_scale: float = 0.2, edge_scale: float = 0.01):
        self.vol_scale = max(float(vol_scale), 1e-9)
        self.drawdown_scale = max(float(drawdown_scale), 1e-9)
        self.edge_scale = max(float(edge_scale), 1e-9)

    def estimate(self, snapshot: ObservationSnapshot) -> CoarseState:
        _ = snapshot.realized_vol
        drift = clip_unit((snapshot.price_return + 0.5 * snapshot.trend_return) / self.vol_scale, -1.0, 1.0)
        triadic_bias = float(clip_unit(snapshot.triadic_state, -1, 1))
        actionability = clip_unit(snapshot.actionability, 0.0, 1.0)
        stress = clip_unit(snapshot.stress, 0.0, 1.0)
        edge = clip_unit(snapshot.edge / self.edge_scale, -1.0, 1.0)
        drawdown = clip_unit(snapshot.drawdown / self.drawdown_scale, 0.0, 1.0)
        contraction = clip_unit(0.5 * actionability + 0.3 * max(edge, 0.0) + 0.2 * (1.0 - stress), 0.0, 1.0)
        diffusion = clip_unit(1.0 - contraction + 0.5 * abs(drift - triadic_bias), 0.0, 1.0)
        current_exposure = clip_unit(snapshot.current_exposure, -1.0, 1.0)
        return CoarseState(
            drift=drift,
            triadic_bias=triadic_bias,
            actionability=actionability,
            stress=stress,
            edge=edge,
            contraction=contraction,
            diffusion=diffusion,
            drawdown=drawdown,
            current_exposure=current_exposure,
        )

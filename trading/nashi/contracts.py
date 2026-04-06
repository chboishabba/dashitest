from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

import numpy as np


Vector = np.ndarray
EigenMass = Mapping[str, float]


class StepStatus(str, Enum):
    INTERIOR = "interior"
    ARROW_BOUNDARY = "arrow_boundary"
    STRUCTURAL_BOUNDARY = "structural_boundary"
    OUTSIDE = "outside"


@dataclass(frozen=True)
class StepMetrics:
    arrow_ok: bool
    cone_ok: bool
    basin_ok: bool
    mdl_ok: bool
    eigen_overlap: float
    q_delta: float
    mdl_prev: float
    mdl_next: float


@dataclass(frozen=True)
class AcceptanceThresholds:
    eps_arrow: float = 1e-9
    eps_cone: float = 1e-9
    min_eigen_overlap: float = 0.80


@dataclass(frozen=True)
class ExecutionDecision:
    accepted: bool
    status: StepStatus
    metrics: StepMetrics
    reasons: tuple[str, ...]


def q_masked(x: Vector, mask: Vector) -> float:
    return float(np.sum(mask * np.square(x)))


def normalized_overlap(lhs: EigenMass, rhs: EigenMass) -> float:
    keys = sorted(set(lhs) | set(rhs))
    mins = 0.0
    maxs = 0.0
    for key in keys:
        a = float(lhs.get(key, 0.0))
        b = float(rhs.get(key, 0.0))
        mins += min(a, b)
        maxs += max(a, b)
    if maxs <= 0.0:
        return 1.0
    return mins / maxs


class NashiContract:
    """
    Runtime mirror of the Agda-side execution admissibility contract.

    This is intentionally small and explicit so it can sit in front of a
    learned or heuristic policy without inheriting its failure modes.
    """

    def __init__(
        self,
        *,
        mask: Sequence[float],
        projection: Callable[[Vector], Vector],
        mdl_fn: Callable[[Vector], float],
        arrow_fn: Callable[[Vector], float],
        basin_pred: Callable[[Vector], bool],
        eigen_fn: Callable[[Vector], EigenMass],
        thresholds: AcceptanceThresholds | None = None,
    ) -> None:
        self.mask = np.asarray(mask, dtype=float)
        self.projection = projection
        self.mdl_fn = mdl_fn
        self.arrow_fn = arrow_fn
        self.basin_pred = basin_pred
        self.eigen_fn = eigen_fn
        self.thresholds = thresholds or AcceptanceThresholds()

    def evaluate_step(self, x_prev: Vector, x_next: Vector) -> ExecutionDecision:
        x_prev = np.asarray(x_prev, dtype=float)
        x_next = np.asarray(x_next, dtype=float)

        dx = x_next - x_prev
        p_prev = self.projection(x_prev)
        p_next = self.projection(x_next)
        p_dx = self.projection(dx)

        arrow_prev = float(self.arrow_fn(x_prev))
        arrow_next = float(self.arrow_fn(x_next))
        arrow_ok = arrow_next >= arrow_prev - self.thresholds.eps_arrow

        q_delta = q_masked(p_dx, self.mask)
        cone_ok = q_delta <= self.thresholds.eps_cone

        basin_ok = self.basin_pred(p_prev) and self.basin_pred(p_next)

        mdl_prev = float(self.mdl_fn(x_prev))
        mdl_next = float(self.mdl_fn(x_next))
        mdl_ok = mdl_next <= mdl_prev + self.thresholds.eps_arrow

        eig_prev = self.eigen_fn(p_prev)
        eig_next = self.eigen_fn(p_next)
        eigen_overlap = normalized_overlap(eig_prev, eig_next)
        eigen_ok = eigen_overlap >= self.thresholds.min_eigen_overlap

        reasons: list[str] = []
        if not arrow_ok:
            reasons.append("arrow")
        if not cone_ok:
            reasons.append("cone")
        if not basin_ok:
            reasons.append("basin")
        if not mdl_ok:
            reasons.append("mdl")
        if not eigen_ok:
            reasons.append("eigen")

        status = StepStatus.INTERIOR
        if cone_ok and not arrow_ok:
            status = StepStatus.ARROW_BOUNDARY
        elif not cone_ok and arrow_ok:
            status = StepStatus.STRUCTURAL_BOUNDARY
        elif not cone_ok and not arrow_ok:
            status = StepStatus.OUTSIDE

        return ExecutionDecision(
            accepted=len(reasons) == 0,
            status=status,
            metrics=StepMetrics(
                arrow_ok=arrow_ok,
                cone_ok=cone_ok,
                basin_ok=basin_ok,
                mdl_ok=mdl_ok,
                eigen_overlap=eigen_overlap,
                q_delta=q_delta,
                mdl_prev=mdl_prev,
                mdl_next=mdl_next,
            ),
            reasons=tuple(reasons),
        )

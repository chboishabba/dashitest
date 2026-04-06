from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from .contracts import StepStatus


CANONICAL_FEATURE_COLUMNS: tuple[str, ...] = ("v_pnorm", "v_dnorm", "v_depth")
CANONICAL_ARROW_COLUMN = "v_arrow"
CANONICAL_CONE_MASK = np.asarray([1.0, 1.0, -1.0], dtype=float)

ARROW_PROFILES: dict[str, float] = {
    "strict": 0.0,
    "boundary": 1e-2,
    "lenient": 1e-1,
}


class FamilyClass(str, Enum):
    INTERIOR_FAMILY = "interior_family"
    ARROW_LADDER = "arrow_ladder"
    SINGLE_ARROW_BREAK = "single_arrow_break"
    MDL_TAIL_BOUNDARY = "mdl_tail_boundary"


@dataclass(frozen=True)
class ClosureEmbedding:
    v_pnorm: float
    v_dnorm: float
    v_depth: float
    v_arrow: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ClosureEmbedding":
        values: dict[str, float] = {}
        for key in CANONICAL_FEATURE_COLUMNS + (CANONICAL_ARROW_COLUMN,):
            raw = row.get(key)
            if raw is None:
                raise KeyError(f"missing embedding field: {key}")
            values[key] = float(raw)
        return cls(**values)

    def feature_vector(self) -> np.ndarray:
        return np.asarray(
            [self.v_pnorm, self.v_dnorm, self.v_depth],
            dtype=float,
        )

    def augmented_vector(self) -> np.ndarray:
        return np.asarray(
            [self.v_pnorm, self.v_dnorm, self.v_depth, self.v_arrow],
            dtype=float,
        )


def step_status_constructor_name(status: StepStatus | str) -> str:
    mapping = {
        StepStatus.INTERIOR.value: "Interior",
        StepStatus.ARROW_BOUNDARY.value: "ArrowBoundary",
        StepStatus.STRUCTURAL_BOUNDARY.value: "StructuralBoundary",
        StepStatus.OUTSIDE.value: "Outside",
    }
    key = status.value if isinstance(status, StepStatus) else str(status)
    return mapping[key]


def family_class_constructor_name(family_class: FamilyClass | str) -> str:
    mapping = {
        FamilyClass.INTERIOR_FAMILY.value: "InteriorFamily",
        FamilyClass.ARROW_LADDER.value: "ArrowLadderFamily",
        FamilyClass.SINGLE_ARROW_BREAK.value: "SingleArrowBreakFamily",
        FamilyClass.MDL_TAIL_BOUNDARY.value: "MDLTailBoundaryFamily",
        "mixed_hard_axis_outlier": "MDLTailBoundaryFamily",
    }
    key = family_class.value if isinstance(family_class, FamilyClass) else str(family_class)
    return mapping[key]

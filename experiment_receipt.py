"""Family-indexed evidence receipts for the heterogeneous dashitest harness.

Plots, CSVs and JSON logs are useful artifacts, but no single `passed` bit can
mean the same thing across trading, compression, CA, quotient-learning,
backend-parity and arithmetic experiments.  This adapter makes experiment
family, provenance, criterion timing and artifact scope explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentFamily(str, Enum):
    TRADING = "trading"
    EPISTEMIC_CA = "epistemic_ca"
    COMPRESSION = "compression"
    TREE_DIFFUSION = "tree_diffusion"
    QUOTIENT_LEARNING = "quotient_learning"
    REACTION_DIFFUSION = "reaction_diffusion"
    VALUATION_PRIMES = "valuation_primes"
    BACKEND_PARITY = "backend_parity"
    SPARSE_LEARNING = "sparse_learning"
    TERNARY_ARITHMETIC = "ternary_arithmetic"
    OTHER_DECLARED = "other_declared"


@dataclass(frozen=True)
class ExperimentCriterion:
    name: str
    declared_before_evaluation: bool
    justification: str


@dataclass(frozen=True)
class ExperimentArtifactReceipt:
    family: ExperimentFamily
    artifact_path: str
    artifact_kind: str
    producer: str
    criterion: ExperimentCriterion | None
    provenance: dict[str, Any]
    diagnostic_only: bool = False
    finite_run_only: bool = True
    artifact_is_proof: bool = False
    posthoc_criterion_is_independent_validation: bool = False

    @property
    def artifact_exists(self) -> bool:
        return Path(self.artifact_path).exists()

    @property
    def criterion_predeclared(self) -> bool:
        return bool(
            self.criterion is not None
            and self.criterion.declared_before_evaluation
        )

    @property
    def eligible_for_formal_intake(self) -> bool:
        return bool(
            self.artifact_exists
            and self.criterion_predeclared
            and not self.diagnostic_only
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_exists"] = self.artifact_exists
        data["criterion_predeclared"] = self.criterion_predeclared
        data["eligible_for_formal_intake"] = self.eligible_for_formal_intake
        return data


def make_receipt(
    *,
    family: ExperimentFamily,
    artifact_path: str,
    artifact_kind: str,
    producer: str,
    provenance: dict[str, Any],
    criterion: ExperimentCriterion | None = None,
    diagnostic_only: bool = False,
) -> ExperimentArtifactReceipt:
    """Create a fail-closed artifact receipt.

    A missing or post-hoc criterion leaves the artifact available for research
    inspection but ineligible for formal intake.  Diagnostic-only tools such as
    the epistemic CA remain diagnostic even when they emit reproducible files.
    """

    return ExperimentArtifactReceipt(
        family=family,
        artifact_path=artifact_path,
        artifact_kind=artifact_kind,
        producer=producer,
        criterion=criterion,
        provenance=dict(provenance),
        diagnostic_only=diagnostic_only,
    )

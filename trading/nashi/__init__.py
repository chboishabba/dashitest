"""
Nashi: Agda-formalism-based trading scaffold.

This package is a clean room for a stricter trader that treats execution
admissibility and refusal as first-class runtime contracts.
"""

from .contracts import (
    AcceptanceThresholds,
    ExecutionDecision,
    NashiContract,
    StepMetrics,
    StepStatus,
)
from .bridge_agda import (
    ARROW_PROFILES,
    build_closure_embedding,
    canonical_arrow,
    canonical_mask,
    canonical_projection,
    classify_family,
    classify_step,
    default_agda_repo_root,
    default_source_repo_root,
    load_dasl_source_model,
    project_row_to_augmented_vector,
    source_eigen_fn,
    source_support_basin_pred,
)
from .policy import NashiIntent, NashiPolicyInput, NashiPolicyOutput, NashiPolicyRuntime
from .phase9 import (
    CapitalLedgerRow,
    CapitalParams,
    MetaWitnessDirectives,
    MetaWitnessState,
    Phase9Decision,
    clamp_exposure,
    estimate_expected_surplus,
    evaluate_meta_witness,
    make_phase9_decision,
    update_capital,
)
from .runtime import NashiArtifacts, NashiMarketAdapter, default_bars, run_nashi_bars
from .proposals import CandidateTransition, NashiObservation, ProposalGenerator
from .schema import (
    CANONICAL_ARROW_COLUMN,
    CANONICAL_CONE_MASK,
    CANONICAL_FEATURE_COLUMNS,
    ClosureEmbedding,
    FamilyClass,
    family_class_constructor_name,
    step_status_constructor_name,
)
from .severity import SeverityCode, SeverityLevel, combine_codes
from .state import NashiState, NashiStepContext
from .telemetry import NashiTelemetry
from .family_memory import FamilyMemory, FamilyStepRecord
from .family_certification import FamilyCertification, certify_family

__all__ = [
    "ARROW_PROFILES",
    "AcceptanceThresholds",
    "CANONICAL_ARROW_COLUMN",
    "CANONICAL_CONE_MASK",
    "CANONICAL_FEATURE_COLUMNS",
    "CapitalLedgerRow",
    "CapitalParams",
    "CandidateTransition",
    "ClosureEmbedding",
    "FamilyMemory",
    "FamilyCertification",
    "FamilyStepRecord",
    "ExecutionDecision",
    "FamilyClass",
    "NashiContract",
    "NashiArtifacts",
    "NashiIntent",
    "NashiObservation",
    "NashiMarketAdapter",
    "NashiPolicyInput",
    "NashiPolicyOutput",
    "NashiPolicyRuntime",
    "NashiState",
    "NashiStepContext",
    "NashiTelemetry",
    "ProposalGenerator",
    "MetaWitnessDirectives",
    "MetaWitnessState",
    "Phase9Decision",
    "SeverityCode",
    "SeverityLevel",
    "StepMetrics",
    "StepStatus",
    "build_closure_embedding",
    "canonical_arrow",
    "canonical_mask",
    "canonical_projection",
    "classify_family",
    "classify_step",
    "clamp_exposure",
    "combine_codes",
    "default_agda_repo_root",
    "default_bars",
    "default_source_repo_root",
    "family_class_constructor_name",
    "estimate_expected_surplus",
    "evaluate_meta_witness",
    "load_dasl_source_model",
    "make_phase9_decision",
    "project_row_to_augmented_vector",
    "certify_family",
    "source_eigen_fn",
    "source_support_basin_pred",
    "step_status_constructor_name",
    "update_capital",
    "run_nashi_bars",
]

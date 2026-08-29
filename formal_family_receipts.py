"""Family-specific BIDI receipts for selected dashitest experiment lanes.

The global `experiment_receipt.py` is only a routing envelope.  These schemas
encode the actual consumer semantics for three mature experiment families:

- tree diffusion / bridge acceptance;
- Phase-3 quotient learning diagnostics;
- reference/executor backend parity.

They intentionally do not share one universal `accept` meaning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class TreeDiffusionCriterion:
    max_mean_cf: float
    max_mean_fc: float
    max_asym_gap: float
    min_abs_corr_leak: float
    max_nonlocal_leak: float | None
    declared_before_evaluation: bool
    baseline_description: str


@dataclass(frozen=True)
class TreeDiffusionReceipt:
    mean_cf: float
    mean_fc: float
    asym_gap: float
    corr_cf_leak: float
    corr_fc_leak: float
    mean_nonlocal_leak: float
    criterion: TreeDiffusionCriterion
    criterion_satisfied: bool
    finite_benchmark_only: bool = True
    proves_universal_tree_transport: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_tree_diffusion_receipt(
    stats: dict[str, Any],
    criterion: TreeDiffusionCriterion,
) -> TreeDiffusionReceipt:
    mean_cf = float(stats["mean_cf"])
    mean_fc = float(stats["mean_fc"])
    asym_gap = float(stats["asym_gap"])
    corr_cf = float(stats["corr_cf_leak"])
    corr_fc = float(stats["corr_fc_leak"])
    nonlocal_leak = float(stats.get("mean_nonlocal_leak", 0.0))

    satisfied = bool(
        criterion.declared_before_evaluation
        and mean_cf <= criterion.max_mean_cf
        and mean_fc <= criterion.max_mean_fc
        and asym_gap <= criterion.max_asym_gap
        and max(abs(corr_cf), abs(corr_fc)) >= criterion.min_abs_corr_leak
        and (
            criterion.max_nonlocal_leak is None
            or nonlocal_leak <= criterion.max_nonlocal_leak
        )
    )
    return TreeDiffusionReceipt(
        mean_cf=mean_cf,
        mean_fc=mean_fc,
        asym_gap=asym_gap,
        corr_cf_leak=corr_cf,
        corr_fc_leak=corr_fc,
        mean_nonlocal_leak=nonlocal_leak,
        criterion=criterion,
        criterion_satisfied=satisfied,
    )


@dataclass(frozen=True)
class Phase3QuotientReceipt:
    source: str
    epochs: int
    has_task_loss: bool
    has_quotient_loss: bool
    has_mdl_cost: bool
    has_alpha: bool
    has_plan_hit: bool
    diagnostic_schema_complete: bool
    universal_quotient_learning_claimed: bool = False
    posthoc_final_loss_threshold_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_phase3_quotient_receipt(path: str | Path) -> Phase3QuotientReceipt:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: Sequence[dict[str, Any]]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("epochs"), list):
        rows = payload["epochs"]
    elif isinstance(payload, dict) and isinstance(payload.get("history"), list):
        rows = payload["history"]
    else:
        rows = []

    def every_has(key: str) -> bool:
        return bool(rows) and all(key in row for row in rows)

    fields = {
        "task_loss": every_has("task_loss"),
        "quotient_loss": every_has("quotient_loss"),
        "mdl_cost": every_has("mdl_cost"),
        "alpha": every_has("alpha"),
        "plan_hit": every_has("plan_hit"),
    }
    return Phase3QuotientReceipt(
        source=str(source),
        epochs=len(rows),
        has_task_loss=fields["task_loss"],
        has_quotient_loss=fields["quotient_loss"],
        has_mdl_cost=fields["mdl_cost"],
        has_alpha=fields["alpha"],
        has_plan_hit=fields["plan_hit"],
        diagnostic_schema_complete=all(fields.values()),
    )


@dataclass(frozen=True)
class BackendParityCriterion:
    atol: float
    rtol: float
    declared_before_evaluation: bool
    reference_backend: str
    executor_backend: str


@dataclass(frozen=True)
class BackendParityFamilyReceipt:
    sample_count: int
    max_abs_error: float
    max_relative_error: float
    criterion: BackendParityCriterion
    parity_supported: bool
    performance_equivalence_claimed: bool = False
    semantic_equivalence_beyond_declared_output_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_backend_parity_receipt(
    reference: Sequence[float],
    executor: Sequence[float],
    criterion: BackendParityCriterion,
) -> BackendParityFamilyReceipt:
    if len(reference) != len(executor):
        raise ValueError("reference/executor outputs must have equal length")
    if not reference:
        raise ValueError("parity receipt requires at least one sample")

    abs_errors = [abs(float(a) - float(b)) for a, b in zip(reference, executor)]
    rel_errors = [
        err / max(abs(float(a)), abs(float(b)), 1e-30)
        for a, b, err in zip(reference, executor, abs_errors)
    ]
    max_abs = max(abs_errors)
    max_rel = max(rel_errors)
    supported = bool(
        criterion.declared_before_evaluation
        and all(
            err <= criterion.atol + criterion.rtol * abs(float(a))
            for a, err in zip(reference, abs_errors)
        )
    )
    return BackendParityFamilyReceipt(
        sample_count=len(reference),
        max_abs_error=max_abs,
        max_relative_error=max_rel,
        criterion=criterion,
        parity_supported=supported,
    )

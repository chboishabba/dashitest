from __future__ import annotations

import json

from formal_family_receipts import (
    BackendParityCriterion,
    TreeDiffusionCriterion,
    build_backend_parity_receipt,
    build_tree_diffusion_receipt,
    load_phase3_quotient_receipt,
)


def test_tree_diffusion_uses_its_declared_multi_metric_acceptance() -> None:
    criterion = TreeDiffusionCriterion(
        max_mean_cf=0.8,
        max_mean_fc=0.9,
        max_asym_gap=0.2,
        min_abs_corr_leak=0.4,
        max_nonlocal_leak=0.1,
        declared_before_evaluation=True,
        baseline_description="frozen baseline",
    )
    stats = {
        "mean_cf": 0.7,
        "mean_fc": 0.8,
        "asym_gap": 0.1,
        "corr_cf_leak": 0.5,
        "corr_fc_leak": 0.2,
        "mean_nonlocal_leak": 0.05,
    }
    receipt = build_tree_diffusion_receipt(stats, criterion)
    assert receipt.criterion_satisfied is True
    assert receipt.proves_universal_tree_transport is False


def test_phase3_log_completeness_is_diagnostic_not_universal_claim(tmp_path) -> None:
    path = tmp_path / "phase3.json"
    path.write_text(
        json.dumps(
            [
                {
                    "epoch": 0,
                    "task_loss": 1.0,
                    "quotient_loss": 0.5,
                    "mdl_cost": 0.2,
                    "alpha": 0.1,
                    "plan_hit": 0.8,
                }
            ]
        ),
        encoding="utf-8",
    )
    receipt = load_phase3_quotient_receipt(path)
    assert receipt.diagnostic_schema_complete is True
    assert receipt.universal_quotient_learning_claimed is False
    assert receipt.posthoc_final_loss_threshold_claimed is False


def test_backend_parity_is_output_relation_not_performance_or_full_semantics() -> None:
    criterion = BackendParityCriterion(
        atol=1e-6,
        rtol=1e-6,
        declared_before_evaluation=True,
        reference_backend="jax_reference",
        executor_backend="vulkan",
    )
    receipt = build_backend_parity_receipt([1.0, 2.0], [1.0, 2.0 + 1e-7], criterion)
    assert receipt.parity_supported is True
    assert receipt.performance_equivalence_claimed is False
    assert receipt.semantic_equivalence_beyond_declared_output_claimed is False

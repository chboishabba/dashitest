import csv
from futures import (
    ActionFunctional,
    BeamConfig,
    BeamDecisionDiagnostics,
    BeamIntentPolicy,
    BeamPolicyStep,
    BeamSummary,
    CoarseState,
    CoarseStateEstimator,
    FutureBeamSearch,
    HeuristicTransitionModel,
    LearnedTransitionModel,
    ResidualTransitionModel,
    ShrinkageTransitionModel,
    ObservationSnapshot,
    build_transition_model_from_logs,
)
from futures.learned import TransitionObservation
import numpy as np
import engine.loop as loopmod
from engine.loop import run_trading_loop
from policy.shadow_runner import ShadowPolicyRunner
from trading_io.logs import beam_summary_to_log_fields
import pathlib
import tempfile


def test_coarse_state_estimator_bounds():
    estimator = CoarseStateEstimator()
    state = estimator.estimate(
        ObservationSnapshot(
            price_return=0.004,
            trend_return=0.006,
            realized_vol=0.01,
            triadic_state=1,
            actionability=0.8,
            stress=0.2,
            edge=0.003,
            drawdown=0.04,
            current_exposure=0.25,
        )
    )
    assert -1.0 <= state.drift <= 1.0
    assert -1.0 <= state.edge <= 1.0
    assert 0.0 <= state.contraction <= 1.0
    assert 0.0 <= state.diffusion <= 1.0
    assert state.action_direction() == 1


def test_action_functional_score_modes_compute_expected_values():
    current = CoarseState(
        drift=0.2,
        triadic_bias=0.1,
        actionability=0.8,
        stress=0.2,
        edge=0.05,
        contraction=0.4,
        diffusion=0.6,
        drawdown=0.1,
        current_exposure=0.0,
    )
    nxt = CoarseState(
        drift=0.3,
        triadic_bias=0.2,
        actionability=0.7,
        stress=0.25,
        edge=0.1,
        contraction=0.55,
        diffusion=0.5,
        drawdown=0.15,
        current_exposure=0.25,
    )
    step_return = 0.04
    step_branch_risk = 0.2
    step_diffusion_risk = 0.35
    next_exposure = 0.5
    weights = ActionFunctional().weights
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
    ratio_score = reward_block / (1.0 + penalty_block)
    scaled_score = 2.0 * (reward_block - penalty_block)
    logistic_score = np.tanh(2.0 * (reward_block - penalty_block))

    ratio = ActionFunctional(score_mode="ratio")
    scaled = ActionFunctional(score_mode="scaled_diff", score_scale=2.0)
    logistic = ActionFunctional(score_mode="logistic", score_scale=2.0)

    ratio_breakdown = ratio.score_transition(current, nxt, next_exposure, step_return, step_branch_risk, step_diffusion_risk)
    scaled_breakdown = scaled.score_transition(current, nxt, next_exposure, step_return, step_branch_risk, step_diffusion_risk)
    logistic_breakdown = logistic.score_transition(current, nxt, next_exposure, step_return, step_branch_risk, step_diffusion_risk)

    assert abs(ratio_breakdown.reward_block - reward_block) < 1e-9
    assert abs(ratio_breakdown.penalty_block - penalty_block) < 1e-9
    assert abs(ratio_breakdown.total - ratio_score) < 1e-9
    assert abs(scaled_breakdown.total - scaled_score) < 1e-9
    assert abs(logistic_breakdown.total - logistic_score) < 1e-9


def test_beam_policy_emits_directional_intent_for_constructive_state():
    estimator = CoarseStateEstimator()
    initial = estimator.estimate(
        ObservationSnapshot(
            price_return=0.006,
            trend_return=0.008,
            realized_vol=0.01,
            triadic_state=1,
            actionability=0.9,
            stress=0.1,
            edge=0.006,
            drawdown=0.01,
            current_exposure=0.0,
        )
    )
    search = FutureBeamSearch(
        action_functional=ActionFunctional(),
        transition_model=HeuristicTransitionModel(),
        beam_config=BeamConfig(horizon=3, beam_width=8, exposure_step=0.25),
    )
    policy = BeamIntentPolicy(symbol="BTCUSDT", search=search)
    result = policy.step(ts=123, initial_state=initial)
    intent = result.intent
    summary = result.summary
    assert summary.long_mass >= summary.short_mass
    assert summary.best_score > 0.0
    assert "beam_" in intent.reason
    assert isinstance(summary, BeamSummary)
    assert isinstance(result.diagnostics, BeamDecisionDiagnostics)
    if intent.direction == 1:
        assert intent.target_exposure > 0.0
    else:
        assert intent.hold


def test_shadow_runner_and_log_fields_surface_beam_summary():
    estimator = CoarseStateEstimator()
    initial = estimator.estimate(
        ObservationSnapshot(
            price_return=0.006,
            trend_return=0.008,
            realized_vol=0.01,
            triadic_state=1,
            actionability=0.9,
            stress=0.1,
            edge=0.006,
            drawdown=0.01,
            current_exposure=0.0,
        )
    )

    class BasePolicy:
        def step(self, ts, initial_state):
            from intent import Intent

            return Intent(
                ts=int(ts),
                symbol="BTCUSDT",
                direction=0,
                target_exposure=0.0,
                urgency=0.0,
                ttl_ms=500,
                hold=True,
                actionability=initial_state.actionability,
                reason="base_hold",
            )

    futures_policy = BeamIntentPolicy(
        symbol="BTCUSDT",
        search=FutureBeamSearch(
            action_functional=ActionFunctional(),
            transition_model=HeuristicTransitionModel(),
            beam_config=BeamConfig(horizon=3, beam_width=8, exposure_step=0.25),
        ),
    )
    shadow = ShadowPolicyRunner(BasePolicy(), futures_policy)
    result = shadow.step(ts=123, initial_state=initial)
    fields = result.to_log_fields()
    beam_fields = beam_summary_to_log_fields(result.beam_summary)
    assert fields["live_direction"] == 0
    assert "shadow_direction" in fields
    assert "shadow_hold_reason_primary" in fields
    assert "shadow_kernel_mode" in fields
    assert "shadow_score_mode" in fields
    assert "shadow_gating_mode" in fields
    assert "beam_entropy" in beam_fields
    assert "beam_curvature" in beam_fields
    assert "beam_flat_distance" in beam_fields
    assert beam_fields["beam_long_mass"] is not None


def test_learned_transition_model_emits_flat_candidate_from_observations():
    estimator = CoarseStateEstimator()
    current = estimator.estimate(
        ObservationSnapshot(
            price_return=0.001,
            trend_return=0.001,
            realized_vol=0.01,
            triadic_state=0,
            actionability=0.5,
            stress=0.2,
            edge=0.0,
            drawdown=0.0,
            current_exposure=0.0,
        )
    )
    flat_next = estimator.estimate(
        ObservationSnapshot(
            price_return=0.0,
            trend_return=0.0,
            realized_vol=0.01,
            triadic_state=0,
            actionability=0.5,
            stress=0.25,
            edge=0.0,
            drawdown=0.0,
            current_exposure=0.0,
        )
    )
    model = LearnedTransitionModel.from_observations(
        [
            TransitionObservation(
                current_state=current,
                action=0,
                next_exposure=0.0,
                next_state=flat_next,
                step_return=0.0,
                branch_risk=0.2,
                diffusion_risk=0.2,
                realized_vol=0.01,
            )
            for _ in range(12)
        ],
        min_bucket_samples=4,
    )
    candidates = list(model.expand(current, action=0, next_exposure=0.0))
    assert candidates
    assert any(candidate.label == "learned_flat" for candidate in candidates)
    assert any(abs(candidate.step_return) <= 1e-9 for candidate in candidates)


def test_build_transition_model_from_logs_falls_back_when_no_data():
    with tempfile.TemporaryDirectory() as tmp:
        model = build_transition_model_from_logs(
            symbol_name="missing_symbol",
            log_dir=pathlib.Path(tmp),
        )
    assert isinstance(model, HeuristicTransitionModel)
    assert model.metadata["kernel_mode"] == "per_asset"


def test_build_transition_model_from_logs_supports_global_per_asset_residual_and_shrinkage_modes():
    headers = ["price_ret", "sigma_slow", "direction", "permission", "stress", "edge_ema", "equity", "pos"]
    rows = []
    for idx in range(20):
        rows.append(
            {
                "price_ret": "0.001" if idx % 2 == 0 else "-0.001",
                "sigma_slow": "0.01",
                "direction": "1" if idx % 3 == 0 else "0",
                "permission": "1",
                "stress": "0.2",
                "edge_ema": "0.001",
                "equity": "100000",
                "pos": "0.25" if idx % 4 else "0.0",
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for name in ("trading_log_btc.csv", "trading_log_spy.csv"):
            path = root / name
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
        global_model = build_transition_model_from_logs(symbol_name="btc", log_dir=root, mode="global")
        asset_model = build_transition_model_from_logs(symbol_name="btc", log_dir=root, mode="per_asset")
        residual_model = build_transition_model_from_logs(symbol_name="btc", log_dir=root, mode="residual")
        shrinkage_model = build_transition_model_from_logs(symbol_name="btc", log_dir=root, mode="shrinkage")
    assert getattr(global_model, "metadata", {})["kernel_mode"] == "global"
    assert getattr(asset_model, "metadata", {})["kernel_mode"] == "per_asset"
    assert isinstance(residual_model, ResidualTransitionModel)
    assert residual_model.metadata["kernel_mode"] == "residual"
    assert isinstance(shrinkage_model, ShrinkageTransitionModel)
    assert shrinkage_model.metadata["kernel_mode"] == "shrinkage"


def test_beam_policy_reports_entropy_and_flat_basin_separately():
    summary = BeamSummary(
        entropy=0.95,
        best_score=0.0,
        long_mass=0.52,
        short_mass=0.48,
        flat_mass=0.0,
        curvature=1.0,
        flat_distance=0.0,
        best_terminal_return=0.0,
        best_terminal_risk=0.0,
        contraction=0.5,
        diffusion=0.5,
    )
    policy = BeamIntentPolicy(
        symbol="BTCUSDT",
        search=FutureBeamSearch(ActionFunctional(), HeuristicTransitionModel()),
    )
    diagnostics = policy._diagnostics(
        summary=summary,
        hold_reason_primary="entropy",
        selected_direction=0,
        score_reward=0.0,
        score_penalty=0.0,
        score_adjusted=0.0,
    )
    assert diagnostics.hold_reason_primary == "entropy"
    assert diagnostics.hold_entropy == 1
    assert diagnostics.hold_flat_basin == 0


def test_run_trading_loop_emits_shadow_fields_when_enabled():
    price = np.array([100.0, 101.0, 102.0, 101.5, 102.5], dtype=float)
    volume = np.ones_like(price) * 1000.0
    summary, rows = run_trading_loop(
        price=price,
        volume=volume,
        source="test_shadow",
        time_index=None,
        max_steps=4,
        sleep_s=0.0,
        log_path=None,
        trade_log_path=None,
        tower_log_path=None,
        log_level="quiet",
        shadow_futures=True,
    )
    assert summary["steps"] > 0
    assert rows
    last = rows[-1]
    assert "shadow_direction" in last
    assert "beam_entropy" in last
    assert "beam_long_mass" in last
    assert "live_direction" in last
    assert "live_fill_intended" in last
    assert "live_next_pos_intended" in last
    assert "shadow_hold_reason_primary" in last
    assert "shadow_kernel_mode" in last
    assert "shadow_hold_curvature" in last


def test_shadow_snapshot_is_built_before_apply_execution():
    price = np.array([100.0, 101.0], dtype=float)
    volume = np.ones_like(price) * 1000.0
    captured = {"snapshot": None, "shadow_called": False}
    original_estimator = loopmod.CoarseStateEstimator
    original_shadow_runner = loopmod.ShadowPolicyRunner
    original_apply_execution = loopmod.apply_execution

    class RecordingEstimator:
        def estimate(self, snapshot):
            captured["snapshot"] = snapshot
            return original_estimator().estimate(snapshot)

    class RecordingStep:
        def to_log_fields(self):
            return {"beam_entropy": 0.0, "shadow_direction": 0}

    class RecordingShadowRunner:
        def __init__(self, base_policy, futures_policy):
            self.base_policy = base_policy
            self.futures_policy = futures_policy

        def step(self, *args, **kwargs):
            captured["shadow_called"] = True
            return RecordingStep()

    def raising_apply_execution(*args, **kwargs):
        raise RuntimeError("sentinel_apply_execution")

    loopmod.CoarseStateEstimator = RecordingEstimator
    loopmod.ShadowPolicyRunner = RecordingShadowRunner
    loopmod.apply_execution = raising_apply_execution
    try:
        try:
            run_trading_loop(
                price=price,
                volume=volume,
                source="test_shadow_order",
                time_index=None,
                max_steps=1,
                sleep_s=0.0,
                log_path=None,
                trade_log_path=None,
                tower_log_path=None,
                log_level="quiet",
                shadow_futures=True,
            )
        except RuntimeError as exc:
            assert str(exc) == "sentinel_apply_execution"
        else:
            raise AssertionError("expected sentinel apply_execution failure")
    finally:
        loopmod.CoarseStateEstimator = original_estimator
        loopmod.ShadowPolicyRunner = original_shadow_runner
        loopmod.apply_execution = original_apply_execution

    assert captured["shadow_called"] is True
    assert captured["snapshot"] is not None
    assert captured["snapshot"].current_exposure == 0.0


if __name__ == "__main__":
    test_coarse_state_estimator_bounds()
    test_beam_policy_emits_directional_intent_for_constructive_state()
    test_shadow_runner_and_log_fields_surface_beam_summary()
    test_learned_transition_model_emits_flat_candidate_from_observations()
    test_build_transition_model_from_logs_falls_back_when_no_data()
    test_build_transition_model_from_logs_supports_global_per_asset_residual_and_shrinkage_modes()
    test_beam_policy_reports_entropy_and_flat_basin_separately()
    test_run_trading_loop_emits_shadow_fields_when_enabled()
    test_shadow_snapshot_is_built_before_apply_execution()
    print("ok")

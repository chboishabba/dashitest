from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import time

import numpy as np
import pandas as pd

try:
    from bar_exec import BarExecution
except ModuleNotFoundError:  # pragma: no cover
    from trading.bar_exec import BarExecution

try:
    from intent import Intent
except ModuleNotFoundError:  # pragma: no cover
    from trading.intent import Intent

try:
    from signals.triadic import compute_triadic_state
except ModuleNotFoundError:  # pragma: no cover
    from trading.signals.triadic import compute_triadic_state

try:
    from trading_io.prices import find_btc_csv, find_stooq_csv, load_price_frame
except ModuleNotFoundError:  # pragma: no cover
    from trading_io.prices import find_btc_csv, find_stooq_csv, load_price_frame

from .bridges import make_canonical_contract
from .family_certification import FamilyCertification, certify_family
from .hazard import attach_hazard_observables, governance_hazard_level, row_hazard_observation
from .phase9 import CapitalParams, Phase9Decision, make_phase9_decision
from .policy import NashiPolicyInput, NashiPolicyRuntime
from .proposals import NashiObservation, ProposalGenerator
from .schema import ClosureEmbedding
from .state import NashiState, NashiStepContext
from .telemetry import NashiTelemetry


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _timestamp_to_ms(raw: object, fallback_index: int) -> int:
    if raw is None:
        return fallback_index * 1000
    if isinstance(raw, (int, float, np.integer, np.floating)):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            abs_value = abs(value)
            if abs_value >= 10**18:
                return value // 1_000_000
            if abs_value >= 10**15:
                return value // 1_000
            if abs_value >= 10**11:
                return value
            if abs_value >= 10**9:
                return value * 1000
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        try:
            return int(raw)  # already milliseconds or index-like
        except (TypeError, ValueError):
            return fallback_index * 1000
    return int(ts.value // 1_000_000)


def _hazard_source_fields(context: NashiStepContext, *, hazard_active: bool) -> tuple[str, str]:
    contextual_active = bool(context.hazard_contextual_active or context.hazard_contextual_pressure > 1e-9)
    if not hazard_active and not contextual_active:
        return "none", "none"
    if contextual_active:
        return "contextual", "contextual" if hazard_active else "none"
    if hazard_active:
        return "synthetic_only", "synthetic_only"
    return "none", "none"


def _ensure_bars(
    bars: pd.DataFrame,
    *,
    ts_values: np.ndarray | None = None,
    default_spread_bps: float = 2.0,
) -> pd.DataFrame:
    frame = bars.copy()
    if "close" not in frame.columns:
        raise ValueError("bars must include a 'close' column")
    if "open" not in frame.columns:
        frame["open"] = frame["close"]
    if "high" not in frame.columns:
        frame["high"] = frame["close"]
    if "low" not in frame.columns:
        frame["low"] = frame["close"]
    if "volume" not in frame.columns:
        frame["volume"] = 1.0
    if "state" not in frame.columns:
        frame["state"] = compute_triadic_state(frame["close"].to_numpy(dtype=float))
    if ts_values is not None:
        frame["raw_ts"] = ts_values
    elif "ts" in frame.columns:
        frame["raw_ts"] = frame["ts"]
    elif "timestamp" in frame.columns:
        frame["raw_ts"] = frame["timestamp"]
    else:
        frame["raw_ts"] = np.arange(len(frame))
    frame = frame.reset_index(drop=True)
    frame["t"] = np.arange(len(frame))
    frame["ts"] = [
        _timestamp_to_ms(raw, idx)
        for idx, raw in enumerate(frame["raw_ts"].tolist())
    ]
    if "spread" not in frame.columns:
        if "bid" in frame.columns and "ask" in frame.columns:
            frame["spread"] = pd.to_numeric(frame["ask"], errors="coerce") - pd.to_numeric(frame["bid"], errors="coerce")
        else:
            frame["spread"] = np.nan
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame["high"], errors="coerce").fillna(frame["close"]).to_numpy(dtype=float)
    low = pd.to_numeric(frame["low"], errors="coerce").fillna(frame["close"]).to_numpy(dtype=float)
    raw_spread = pd.to_numeric(frame["spread"], errors="coerce").to_numpy(dtype=float)
    range_proxy = np.maximum(0.0, high - low)
    bps_floor = np.maximum(close, 1e-9) * (default_spread_bps * 1e-4)
    synthetic_spread = np.maximum(bps_floor, 0.1 * range_proxy)
    spread = np.where(np.isfinite(raw_spread) & (raw_spread >= 0.0), raw_spread, synthetic_spread)
    frame["spread"] = spread
    bid = pd.to_numeric(frame["bid"], errors="coerce") if "bid" in frame.columns else pd.Series(np.nan, index=frame.index)
    ask = pd.to_numeric(frame["ask"], errors="coerce") if "ask" in frame.columns else pd.Series(np.nan, index=frame.index)
    frame["bid"] = np.where(np.isfinite(bid.to_numpy(dtype=float)), bid.to_numpy(dtype=float), close - 0.5 * spread)
    frame["ask"] = np.where(np.isfinite(ask.to_numpy(dtype=float)), ask.to_numpy(dtype=float), close + 0.5 * spread)
    frame["quote_mode"] = np.where(
        ("bid" in bars.columns) and ("ask" in bars.columns),
        "quoted",
        "synthetic",
    )
    return frame


@dataclass(frozen=True)
class NashiArtifacts:
    step_log_path: Path
    decision_ndjson_path: Path
    ohlc_ndjson_path: Path
    duckdb_path: Path
    family_csv_path: Path | None = None
    family_ndjson_path: Path | None = None


@dataclass(frozen=True)
class RealizedAlignment:
    expected_surplus: float = 0.0
    expected_gross_surplus: float = 0.0
    expected_cost: float = 0.0
    active: bool = False


class NashiMarketAdapter:
    """
    Bootstrap adapter from repo bar rows into canonical Nashi embeddings.

    This is intentionally small and observable. It is a compatibility bridge,
    not the final Agda-faithful proposal engine.
    """

    def __init__(
        self,
        *,
        base_size: float = 1.0,
        vol_window: int = 32,
        ttl_ms: int = 60_000,
    ) -> None:
        self.base_size = float(base_size)
        self.vol_window = max(4, int(vol_window))
        self.ttl_ms = int(ttl_ms)
        self.proposal_generator = ProposalGenerator(base_size=base_size)

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.prepare_with_context(bars, contextual_windows=None)

    def prepare_with_context(
        self,
        bars: pd.DataFrame,
        *,
        contextual_windows: pd.DataFrame | Path | str | None = None,
    ) -> pd.DataFrame:
        frame = bars.copy()
        close = frame["close"].to_numpy(dtype=float)
        state = frame["state"].to_numpy(dtype=float)
        returns = np.zeros(len(frame), dtype=float)
        if len(close) > 1:
            returns[1:] = np.diff(close) / np.maximum(close[:-1], 1e-9)
        vol = pd.Series(returns).rolling(self.vol_window, min_periods=2).std().fillna(0.0).to_numpy()
        vol_ref = float(np.nanmedian(vol[vol > 0])) if np.any(vol > 0) else 1e-4
        vol_ref = max(vol_ref, 1e-6)
        z_ret = returns / np.maximum(vol, vol_ref)
        edge = np.tanh(z_ret)
        stress = np.clip(vol / vol_ref, 0.0, 2.0)
        state_boost = np.clip(np.abs(state), 0.0, 1.0)
        actionability = np.clip(np.abs(edge) * (1.0 - 0.35 * np.clip(stress, 0.0, 1.0)) + 0.15 * state_boost, 0.0, 1.0)
        urgency = np.clip(0.2 + 0.8 * actionability, 0.0, 1.0)
        edge_series = pd.Series(edge, dtype=float)
        prior_edge_mean = edge_series.rolling(4, min_periods=1).mean().shift(1).fillna(0.0)
        prior_same_sign = (
            np.sign(edge_series.to_numpy(dtype=float)) == np.sign(prior_edge_mean.to_numpy(dtype=float))
        ) & (np.sign(edge_series.to_numpy(dtype=float)) != 0)
        edge_persistence = np.where(
            prior_same_sign,
            np.minimum(
                1.0,
                np.abs(prior_edge_mean.to_numpy(dtype=float)) / np.maximum(np.abs(edge), 1e-9),
            ),
            0.0,
        )
        edge_shock = np.clip(np.abs(edge - prior_edge_mean.to_numpy(dtype=float)), 0.0, 1.0)
        frame["price_return"] = returns
        frame["realized_vol"] = vol
        frame["edge"] = edge
        frame["edge_persistence"] = edge_persistence
        frame["edge_shock"] = edge_shock
        frame["stress"] = stress
        frame["actionability"] = actionability
        frame["urgency"] = urgency
        return attach_hazard_observables(
            frame,
            stress_window=max(self.vol_window * 3, 24),
            sigma_window=max(self.vol_window, 16),
            contextual_windows=contextual_windows,
        )

    def observe(
        self,
        row: Mapping[str, object],
        state: NashiState,
        drawdown: float,
    ) -> NashiObservation:
        price = float(row["close"])
        spread = float(row["spread"])
        spread_bps = 1e4 * spread / max(price, 1e-9)
        actionability = float(row["actionability"])
        edge = float(row["edge"])
        hazard = row_hazard_observation(dict(row))
        microstructure_pressure = _clamp(spread / max(abs(edge) * price, 1e-9), 0.0, 4.0)
        cost_survival_ratio = (abs(edge) * max(actionability, 1e-9)) / max(spread_bps * 1e-4, 1e-9)
        context = NashiStepContext(
            ts=int(row["ts"]),
            price=price,
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            spread=spread,
            spread_bps=spread_bps,
            price_return=float(row["price_return"]),
            realized_vol=float(row["realized_vol"]),
            actionability=actionability,
            edge=edge,
            edge_persistence=float(row.get("edge_persistence", 0.0)),
            edge_shock=float(row.get("edge_shock", 0.0)),
            stress=float(row["stress"]),
            microstructure_pressure=microstructure_pressure,
            cost_survival_ratio=cost_survival_ratio,
            drawdown=float(drawdown),
            current_exposure=float(state.exposure),
            family_cooldown=min(1.0, float(state.family_memory.post_unwind_cooldown) / 3.0),
            hazard=hazard.hazard_score,
            hazard_regime=hazard.hazard_regime,
            hazard_p_bad=hazard.hazard_p_bad,
            hazard_bad_flag=hazard.hazard_bad_flag,
            hazard_contextual_pressure=hazard.hazard_contextual_pressure,
            hazard_contextual_active=bool(getattr(hazard, "hazard_contextual_active", False)),
            hazard_contextual_label=str(getattr(hazard, "hazard_contextual_label", "")),
            hazard_ema=hazard.hazard_ema,
            hazard_persistence=hazard.hazard_persistence,
            hazard_trend=hazard.hazard_trend,
            hazard_cooldown=hazard.hazard_cooldown,
        )
        current_embedding = self._current_embedding(row, state)
        state_signal = int(np.sign(float(row["state"])))
        return NashiObservation(
            context=context,
            current_embedding=current_embedding,
            state_signal=state_signal,
        )

    def build_policy_input(
        self,
        observation: NashiObservation,
        state: NashiState,
    ) -> NashiPolicyInput:
        candidates = self.proposal_generator.generate(observation, state)
        return NashiPolicyInput(
            context=observation.context,
            current_embedding=observation.current_embedding.augmented_vector(),
            candidates=tuple(candidates),
        )

    def to_repo_intent(self, policy_input: NashiPolicyInput, symbol: str, output) -> Intent:
        signed_target = float(output.intent.direction) * float(output.intent.target_exposure)
        current_signed = float(policy_input.context.current_exposure)
        if output.intent.hold:
            signed_target = current_signed
        direction = _sign(signed_target)
        target_exposure = abs(signed_target)
        return Intent(
            ts=output.intent.ts,
            symbol=symbol,
            direction=direction,
            target_exposure=target_exposure,
            urgency=_clamp(float(policy_input.context.actionability), 0.0, 1.0),
            ttl_ms=self.ttl_ms,
            hold=bool(output.intent.hold),
            actionability=_clamp(float(policy_input.context.actionability), 0.0, 1.0),
            reason=output.intent.reason,
        )

    @staticmethod
    def apply_phase9(
        repo_intent: Intent,
        *,
        state: NashiState,
        phase9: Phase9Decision,
    ) -> Intent:
        signed_target = repo_intent.direction * repo_intent.target_exposure
        if phase9.directives.force_ban:
            signed_target = 0.0
        elif phase9.directives.force_hold:
            signed_target = state.exposure
        else:
            signed_target = max(
                -phase9.directives.max_abs_exposure,
                min(phase9.directives.max_abs_exposure, signed_target),
            )
        direction = _sign(signed_target)
        target_exposure = abs(signed_target)
        hold = bool(
            (
                phase9.directives.force_hold
                and not phase9.directives.force_ban
            )
            or abs(signed_target - state.exposure) <= 1e-9
            or (direction == 0 and abs(state.exposure) < 1e-9)
        )
        return Intent(
            ts=repo_intent.ts,
            symbol=repo_intent.symbol,
            direction=direction,
            target_exposure=target_exposure,
            urgency=repo_intent.urgency,
            ttl_ms=repo_intent.ttl_ms,
            hold=hold,
            actionability=repo_intent.actionability,
            reason=phase9.directives.reason if phase9.directives.reason != "normal" else repo_intent.reason,
        )

    def make_step_row(
        self,
        *,
        row: Mapping[str, object],
        symbol: str,
        repo_intent: Intent,
        result: Mapping[str, object],
        policy_input: NashiPolicyInput,
        output,
        phase9: Phase9Decision,
        phase9_params: CapitalParams,
        family: FamilyCertification,
        pnl: float,
        alignment: RealizedAlignment,
    ) -> dict[str, object]:
        metrics = output.contract_decision.metrics
        executed_expected_surplus = float(phase9.executed_expected_surplus)
        realized_surplus = float(phase9.ledger.realized_surplus)
        aligned_expected_surplus = float(alignment.expected_surplus)
        aligned_expected_gross_surplus = float(alignment.expected_gross_surplus)
        aligned_expected_cost = float(alignment.expected_cost)
        aligned_realized_surplus = float(phase9.ledger.pnl_mtm)
        aligned_realized_efficiency = 0.0
        if abs(aligned_expected_surplus) > 1e-9:
            aligned_realized_efficiency = aligned_realized_surplus / aligned_expected_surplus
        execution_fee_cost = float(result.get("fee", 0.0))
        execution_slippage_cost = float(result.get("slippage_cost", 0.0))
        execution_cost_realized = execution_fee_cost + execution_slippage_cost
        execution_cost_gap = execution_cost_realized - float(phase9.executed_expected_cost)
        requested_turnover = abs(float(result.get("requested_delta", 0.0)))
        filled_turnover = abs(float(result.get("filled", 0.0)))
        execution_fill_ratio = 0.0
        if requested_turnover > 1e-9:
            execution_fill_ratio = filled_turnover / requested_turnover
        hazard_directive = "hazard_" in str(phase9.directives.reason)
        hazard_name = str(policy_input.context.hazard_contextual_label or policy_input.context.hazard_regime)
        executable_opportunity = bool(
            executed_expected_surplus > 1e-9
            and not phase9.directives.force_hold
            and not phase9.directives.force_ban
        )
        realized_efficiency = 0.0
        if abs(executed_expected_surplus) > 1e-9:
            realized_efficiency = realized_surplus / executed_expected_surplus
        hazard_source, hazard_tightened_source = _hazard_source_fields(
            policy_input.context,
            hazard_active=bool(phase9.hazard_tightened),
        )
        return {
            "t": int(row["t"]),
            "ts": int(row["ts"]),
            "symbol": symbol,
            "price": float(row["close"]),
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
            "spread": float(row["spread"]),
            "spread_bps": float(policy_input.context.spread_bps),
            "volume": float(row["volume"]),
            "state": int(row["state"]),
            "acceptable": bool(output.contract_decision.accepted),
            "intent_direction": int(repo_intent.direction),
            "intent_target": float(repo_intent.target_exposure),
            "urgency": float(repo_intent.urgency),
            "actionability": float(policy_input.context.actionability),
            "ell": float(policy_input.context.edge),
            "fill": float(result["filled"]),
            "fill_price": float(result["fill_price"]),
            "fee": float(result["fee"]),
            "pnl": float(pnl),
            "exposure": float(result["exposure"]),
            "slippage": float(result["slippage"]),
            "reason": str(repo_intent.reason),
            "action": int(repo_intent.direction),
            "hold": int(bool(repo_intent.hold)),
            "z_vel": float(policy_input.context.price_return),
            "edge": float(policy_input.context.edge),
            "edge_persistence": float(policy_input.context.edge_persistence),
            "edge_shock": float(policy_input.context.edge_shock),
            "stress": float(policy_input.context.stress),
            "microstructure_pressure": float(policy_input.context.microstructure_pressure),
            "cost_survival_ratio": float(policy_input.context.cost_survival_ratio),
            "family_cooldown": float(policy_input.context.family_cooldown),
            "drawdown": float(policy_input.context.drawdown),
            "hazard": float(policy_input.context.hazard),
            "hazard_regime": str(policy_input.context.hazard_regime),
            "hazard_p_bad": float(policy_input.context.hazard_p_bad),
            "hazard_bad_flag": int(bool(policy_input.context.hazard_bad_flag)),
            "hazard_contextual_pressure": float(policy_input.context.hazard_contextual_pressure),
            "hazard_contextual_active": int(bool(policy_input.context.hazard_contextual_active)),
            "hazard_contextual_label": str(policy_input.context.hazard_contextual_label),
            "hazard_ema": float(policy_input.context.hazard_ema),
            "hazard_persistence": float(policy_input.context.hazard_persistence),
            "hazard_trend": float(policy_input.context.hazard_trend),
            "hazard_cooldown": float(policy_input.context.hazard_cooldown),
            "nashi_status": output.contract_decision.status.value,
            "nashi_refusal": output.refusal.label,
            "nashi_reasons": "|".join(output.contract_decision.reasons),
            "nashi_candidate_id": output.selected_candidate_id,
            "nashi_candidate_reason": output.selected_candidate_reason,
            "nashi_spread_regime": output.selected_spread_regime,
            "nashi_rejected_candidates": "|".join(output.rejected_candidates),
            "nashi_family_class": family.family_class.value,
            "nashi_family_constructor": family.family_constructor,
            "nashi_family_certified": int(bool(family.certified)),
            "nashi_family_trade_certified": int(bool(family.trade_certified)),
            "nashi_family_preserve_certified": int(bool(family.preserve_certified)),
            "nashi_family_tail_localized": int(bool(family.tail_localized)),
            "nashi_family_spread_dominated": int(bool(family.spread_dominated)),
            "nashi_family_hostile_regime": int(bool(family.hostile_regime)),
            "nashi_family_arrow_boundary_share": float(family.arrow_boundary_share),
            "nashi_family_microstructure_kill_share": float(family.microstructure_kill_share),
            "nashi_family_reasons": "|".join(family.reasons),
            "nashi_q_delta": float(metrics.q_delta),
            "nashi_eigen_overlap": float(metrics.eigen_overlap),
            "nashi_mdl_prev": float(metrics.mdl_prev),
            "nashi_mdl_next": float(metrics.mdl_next),
            "capital_C": float(phase9.ledger.capital_next),
            "capital_dd": float(phase9.ledger.capital_dd),
            "kappa_t": float(phase9.ledger.kappa_t),
            "mw_reason": phase9.directives.reason,
            "mw_refusal_level": phase9.directives.refusal.level.name.lower(),
            "mw_forced_hold": int(bool(phase9.directives.force_hold)),
            "mw_forced_ban": int(bool(phase9.directives.force_ban)),
            "mw_max_exposure": float(phase9.directives.max_abs_exposure),
            "proposed_executable_opportunity": int(bool(output.selected_candidate_executable_viable)),
            "proposed_governance_viable": int(bool(output.selected_candidate_governance_viable)),
            "proposed_cost_viable": int(bool(output.selected_candidate_cost_viable)),
            "proposed_survivability_viable": int(bool(output.selected_candidate_survivability_viable)),
            "proposed_survivability_score": float(output.selected_candidate_survivability_score),
            "proposed_viability_reason": output.selected_candidate_viability_reason,
            "proposed_expected_surplus": float(phase9.proposed_expected_surplus),
            "proposed_expected_gross_surplus": float(phase9.proposed_expected_gross_surplus),
            "proposed_expected_cost": float(phase9.proposed_expected_cost),
            "proposed_cost_survival_ratio": float(phase9.proposed_cost_survival_ratio),
            "expected_surplus": float(phase9.expected_surplus),
            "expected_gross_surplus": float(phase9.expected_gross_surplus),
            "expected_cost": float(phase9.expected_cost),
            "phase9_cost_survival_ratio": float(phase9.cost_survival_ratio),
            "executed_expected_surplus": float(phase9.executed_expected_surplus),
            "executed_expected_gross_surplus": float(phase9.executed_expected_gross_surplus),
            "executed_expected_cost": float(phase9.executed_expected_cost),
            "executed_cost_survival_ratio": float(phase9.executed_cost_survival_ratio),
            "aligned_expected_surplus": aligned_expected_surplus,
            "aligned_expected_gross_surplus": aligned_expected_gross_surplus,
            "aligned_expected_cost": aligned_expected_cost,
            "aligned_realized_surplus": aligned_realized_surplus,
            "aligned_realized_efficiency": float(aligned_realized_efficiency),
            "execution_fee_cost": execution_fee_cost,
            "execution_slippage_cost": execution_slippage_cost,
            "execution_cost_realized": execution_cost_realized,
            "execution_cost_gap": execution_cost_gap,
            "execution_requested_turnover": requested_turnover,
            "execution_filled_turnover": filled_turnover,
            "execution_fill_ratio": float(execution_fill_ratio),
            "executable_opportunity": int(executable_opportunity),
            "realized_efficiency": float(realized_efficiency),
            "phase9_microstructure_kills_edge": int(bool(phase9.microstructure_kills_edge)),
            "hazard_active": int(bool(phase9.hazard_tightened)),
            "hazard_source": hazard_source,
            "hazard_tightened_source": hazard_tightened_source,
            "hazard_name": hazard_name,
            "hazard_reason": phase9.regime_label if phase9.hazard_tightened else "",
            "hazard_forced_hold": int(bool(hazard_directive and phase9.directives.force_hold)),
            "hazard_forced_ban": int(bool(hazard_directive and phase9.directives.force_ban)),
            "phase9_cfg_hazard_clamp_threshold": float(phase9_params.hazard_clamp_threshold),
            "phase9_cfg_hazard_hold_threshold": float(phase9_params.hazard_hold_threshold),
            "phase9_cfg_hazard_ban_threshold": float(phase9_params.hazard_ban_threshold),
            "phase9_cfg_hazard_survival_floor_add": float(phase9_params.hazard_survival_floor_add),
            "phase9_cfg_hazard_exposure_tightening": float(phase9_params.hazard_exposure_tightening),
            "phase9_cfg_hazard_min_exposure_scale": float(phase9_params.hazard_min_exposure_scale),
            "phase9_cfg_min_expected_surplus": float(phase9_params.min_expected_surplus),
            "phase9_cfg_min_actionability": float(phase9_params.min_actionability),
            "phase9_cfg_min_edge": float(phase9_params.min_edge),
            "phase9_cfg_microstructure_floor": float(phase9_params.microstructure_survival_floor),
            "phase9_cfg_microstructure_floor_min": float(phase9_params.microstructure_survival_floor_min),
            "phase9_cfg_microstructure_relief": float(phase9_params.microstructure_relief_strength),
            "phase9_cfg_microstructure_min_turnover": float(phase9_params.microstructure_min_turnover),
            "phase9_cfg_microstructure_min_gross": float(phase9_params.microstructure_min_expected_gross),
            "realized_surplus": realized_surplus,
            "justification_chain": phase9.justification_chain,
            "just_regime": phase9.regime_label,
            "just_posture": phase9.posture_label,
            "just_actuator": phase9.actuator_label,
            "just_cost_model": phase9.cost_model_label,
        }

    @staticmethod
    def make_decision_row(
        *,
        row: Mapping[str, object],
        symbol: str,
        result_exposure: float,
        repo_intent: Intent,
        phase9: Phase9Decision,
    ) -> dict[str, object]:
        hazard_active = bool(phase9.hazard_tightened)
        return {
            "timestamp": int(row["ts"]),
            "symbol": symbol,
            "state": int(row["state"]),
            "direction": _sign(result_exposure),
            "target_exposure": abs(float(result_exposure)),
            "urgency": float(repo_intent.urgency),
            "hold": bool(repo_intent.hold),
            "actionability": float(repo_intent.actionability),
            "reason": str(repo_intent.reason),
            "hazard_active": hazard_active,
            "hazard_reason": phase9.regime_label if hazard_active else "",
            "hazard_name": str(row.get("hazard_contextual_label") or row.get("hazard_regime", "")),
        }

    @staticmethod
    def make_ohlc_row(row: Mapping[str, object], symbol: str) -> dict[str, object]:
        close = float(row["close"])
        return {
            "timestamp": int(row["ts"]),
            "symbol": symbol,
            "open": float(row.get("open", close)),
            "high": float(row.get("high", close)),
            "low": float(row.get("low", close)),
            "close": close,
            "volume": float(row["volume"]),
            "trades": 1,
        }

    @staticmethod
    def make_family_row(
        row: Mapping[str, object],
        *,
        symbol: str,
        family: FamilyCertification,
    ) -> dict[str, object]:
        return {
            "t": int(row["t"]),
            "timestamp": int(row["ts"]),
            "symbol": symbol,
            "family_class": family.family_class.value,
            "family_constructor": family.family_constructor,
            "certified": bool(family.certified),
            "trade_certified": bool(family.trade_certified),
            "preserve_certified": bool(family.preserve_certified),
            "tail_localized": bool(family.tail_localized),
            "spread_dominated": bool(family.spread_dominated),
            "hostile_regime": bool(family.hostile_regime),
            "arrow_boundary_share": float(family.arrow_boundary_share),
            "microstructure_kill_share": float(family.microstructure_kill_share),
            "window_size": int(family.window_size),
            "reasons": "|".join(family.reasons),
        }

    @staticmethod
    def _current_embedding(row: Mapping[str, object], state: NashiState) -> ClosureEmbedding:
        dnorm = np.tanh(float(state.exposure))
        depth = _clamp(float(row["actionability"]) + 0.5 * (1.0 - abs(dnorm)), 0.0, 1.0)
        return ClosureEmbedding(
            v_pnorm=float(np.tanh(float(row["edge"]))),
            v_dnorm=float(dnorm),
            v_depth=float(depth),
            v_arrow=_clamp(float(state.last_arrow), 0.0, 1.0),
        )

def default_bars(csv_path: Path | None = None, *, default_spread_bps: float = 2.0) -> tuple[pd.DataFrame, str]:
    source_path = csv_path
    if source_path is None:
        source_path = find_btc_csv()
    if source_path is None:
        source_path = find_stooq_csv()
    if source_path is None:
        raise FileNotFoundError("No cached CSV available for Nashi run.")
    frame = load_price_frame(source_path)
    bars = _ensure_bars(frame, default_spread_bps=default_spread_bps)
    return bars, str(source_path)


def run_nashi_bars(
    bars: pd.DataFrame,
    *,
    symbol: str,
    artifacts: NashiArtifacts,
    source_label: str,
    repo_root_for_contract: Path | None = None,
    base_size: float = 1.0,
    initial_capital: float = 100_000.0,
    phase9_params: CapitalParams | None = None,
    default_spread_bps: float = 2.0,
    contextual_hazard_csv: Path | None = None,
    progress_fn: Callable[[int, int, float], None] | None = None,
) -> pd.DataFrame:
    bars = _ensure_bars(bars, default_spread_bps=default_spread_bps)
    adapter = NashiMarketAdapter(base_size=base_size)
    frame = adapter.prepare_with_context(bars, contextual_windows=contextual_hazard_csv)
    contract = make_canonical_contract(repo_root_for_contract)
    runtime = NashiPolicyRuntime(contract)
    executor = BarExecution()
    phase9_params = phase9_params or CapitalParams()
    telemetry = NashiTelemetry(
        step_log_path=artifacts.step_log_path,
        decision_ndjson_path=artifacts.decision_ndjson_path,
        ohlc_ndjson_path=artifacts.ohlc_ndjson_path,
        duckdb_path=artifacts.duckdb_path,
        family_csv_path=artifacts.family_csv_path,
        family_ndjson_path=artifacts.family_ndjson_path,
        source_label=source_label,
    )

    state = NashiState(capital=initial_capital, cash=initial_capital)
    logs: list[dict[str, object]] = []
    alignment = RealizedAlignment()
    equity = 1.0
    peak_equity = 1.0
    prev_price: float | None = None
    started_at = time.perf_counter()
    total_rows = len(frame)

    try:
        for idx, row in enumerate(frame.to_dict(orient="records"), start=1):
            price = float(row["close"])
            if prev_price is not None and prev_price > 0.0:
                ret = (price / prev_price) - 1.0
                equity *= 1.0 + float(state.exposure) * ret
            peak_equity = max(peak_equity, equity)
            drawdown = 0.0 if peak_equity <= 0.0 else max(0.0, 1.0 - (equity / peak_equity))

            observation = adapter.observe(row, state, drawdown)
            policy_input = adapter.build_policy_input(observation, state)
            output = runtime.step(state, policy_input)
            repo_intent = adapter.to_repo_intent(policy_input, symbol, output)
            proposed_signed = repo_intent.direction * repo_intent.target_exposure
            prev_exposure = float(state.exposure)
            phase9 = make_phase9_decision(
                capital_prev=state.capital,
                exposure_prev=prev_exposure,
                proposed_exposure=proposed_signed,
                price_return=float(policy_input.context.price_return),
                edge=float(policy_input.context.edge),
                actionability=float(policy_input.context.actionability),
                drawdown=float(drawdown),
                sigma=float(policy_input.context.realized_vol),
                hazard=governance_hazard_level(
                    row_hazard_observation(
                        {
                            "hazard_score": policy_input.context.hazard,
                            "hazard_regime": policy_input.context.hazard_regime,
                            "hazard_p_bad": policy_input.context.hazard_p_bad,
                            "hazard_bad_flag": policy_input.context.hazard_bad_flag,
                            "hazard_contextual_pressure": policy_input.context.hazard_contextual_pressure,
                            "hazard_contextual_active": policy_input.context.hazard_contextual_active,
                            "hazard_contextual_label": policy_input.context.hazard_contextual_label,
                            "hazard_ema": policy_input.context.hazard_ema,
                            "hazard_persistence": policy_input.context.hazard_persistence,
                            "hazard_trend": policy_input.context.hazard_trend,
                            "hazard_cooldown": policy_input.context.hazard_cooldown,
                        }
                    )
                ),
                spread_bps=float(policy_input.context.spread_bps),
                phase6_open=True,
                phase7_ready=float(policy_input.context.edge) > 0.0,
                # A prior-row blocking refusal is not, by itself, a current-row
                # contradiction. Let the current Phase-9 rules decide whether the
                # proposed move is still sparse, banned, or cost-dominated.
                prior_refusal_active=False,
                params=phase9_params,
            )
            repo_intent = adapter.apply_phase9(repo_intent, state=state, phase9=phase9)
            result = executor.execute(
                repo_intent,
                price,
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                capital=float(state.capital),
            )

            equity += float(result["pnl"])
            peak_equity = max(peak_equity, equity)
            pnl = (equity - 1.0) * initial_capital

            state.exposure = float(result["exposure"])
            state.last_price = price
            if output.contract_decision.accepted:
                selected_candidate = next(
                    (
                        candidate
                        for candidate in policy_input.candidates
                        if candidate.candidate_id == output.selected_candidate_id
                    ),
                    None,
                )
                if selected_candidate is not None:
                    state.last_arrow = max(state.last_arrow, float(selected_candidate.proposed_embedding.v_arrow))
            state.last_mdl = float(output.contract_decision.metrics.mdl_next)
            state.refusal = phase9.directives.refusal if phase9.directives.refusal.level > output.refusal.level else output.refusal
            state.capital = float(phase9.ledger.capital_next)
            state.cash = state.capital - state.exposure * price
            state.capital_drawdown = float(phase9.ledger.capital_dd)
            state.family_memory.observe(
                output.contract_decision,
                output.selected_family_hint,
                spread_regime=output.selected_spread_regime,
                microstructure_kills_edge=phase9.microstructure_kills_edge,
                cost_survival_ratio=phase9.cost_survival_ratio,
                hazard_level=phase9.hazard_level,
                hazard_tightened=phase9.hazard_tightened,
                prior_exposure=prev_exposure,
                new_exposure=float(result["exposure"]),
                fill=float(result.get("filled", 0.0)),
            )
            family = certify_family(state.family_memory)

            step_row = adapter.make_step_row(
                row=row,
                symbol=symbol,
                repo_intent=repo_intent,
                result=result,
                policy_input=policy_input,
                output=output,
                phase9=phase9,
                phase9_params=phase9_params,
                family=family,
                pnl=pnl,
                alignment=alignment,
            )
            decision_row = adapter.make_decision_row(
                row=row,
                symbol=symbol,
                result_exposure=float(result["exposure"]),
                repo_intent=repo_intent,
                phase9=phase9,
            )
            ohlc_row = adapter.make_ohlc_row(row, symbol)
            family_row = adapter.make_family_row(row, symbol=symbol, family=family)

            telemetry.emit(step_row=step_row, decision_row=decision_row, ohlc_row=ohlc_row, family_row=family_row)
            logs.append(step_row)
            alignment = RealizedAlignment(
                expected_surplus=float(phase9.executed_expected_surplus),
                expected_gross_surplus=float(phase9.executed_expected_gross_surplus),
                expected_cost=float(phase9.executed_expected_cost),
                active=bool(abs(float(phase9.executed_expected_surplus)) > 1e-9),
            )
            prev_price = price
            if progress_fn is not None:
                progress_fn(idx, total_rows, time.perf_counter() - started_at)
    finally:
        telemetry.close()

    return pd.DataFrame(logs)

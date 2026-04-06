from __future__ import annotations

from dataclasses import dataclass

from .severity import SeverityCode, SeverityLevel, combine_codes


@dataclass(frozen=True)
class CapitalParams:
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0003
    margin_rate: float = 0.0002
    lambda_dd: float = 0.75
    lambda_sigma: float = 0.10
    lambda_posture: float = 0.02
    beta: float = 0.50
    capital_floor: float = 1000.0
    drawdown_limit: float = 0.25
    min_expected_surplus: float = 0.0
    min_actionability: float = 0.20
    min_edge: float = 0.01
    microstructure_survival_floor: float = 1.05
    microstructure_survival_floor_min: float = 0.60
    microstructure_relief_strength: float = 0.40
    microstructure_min_turnover: float = 1e-3
    microstructure_min_expected_gross: float = 1e-6
    hazard_reentry_threshold: float = 0.22
    hazard_reentry_relief: float = 0.35
    hazard_clamp_threshold: float = 0.48
    hazard_hold_threshold: float = 0.78
    hazard_ban_threshold: float = 0.96
    hazard_survival_floor_add: float = 0.40
    hazard_exposure_tightening: float = 0.60
    hazard_min_exposure_scale: float = 0.15
    survivability_min: float = 0.28
    survivability_shock_weight: float = 0.75
    survivability_hazard_weight: float = 0.20
    survivability_cooldown_weight: float = 0.20
    continuation_min: float = 0.18
    continuation_grace_bars: int = 2
    continuation_opposition_weight: float = 0.55
    continuation_shock_weight: float = 0.20
    continuation_hazard_weight: float = 0.10
    continuation_age_weight: float = 0.15


@dataclass(frozen=True)
class CapitalLedgerRow:
    capital_prev: float
    capital_next: float
    capital_dd: float
    kappa_t: float
    pnl_mtm: float
    friction_cost: float
    risk_tax: float
    realized_surplus: float
    dx: float


@dataclass(frozen=True)
class MetaWitnessState:
    phase6_open: bool
    phase7_ready: bool
    actionability: float
    expected_surplus: float
    drawdown: float
    turnover: float
    stress: float
    spread_bps: float
    cost_survival_ratio: float
    microstructure_kills_edge: bool
    current_exposure: float
    proposed_exposure: float
    hazard: float = 0.0
    prior_refusal_active: bool = False
    continuation_active: bool = False
    continuation_supported: bool = False
    continuation_expected_surplus: float = 0.0


@dataclass(frozen=True)
class ProposalEconomics:
    proposed_exposure: float
    turnover: float
    expected_surplus: float
    expected_gross_surplus: float
    expected_cost: float
    cost_survival_ratio: float
    governance_viable: bool
    cost_viable: bool
    executable_viable: bool
    survivability_viable: bool
    survivability_score: float
    viability_reason: str


@dataclass(frozen=True)
class MetaWitnessDirectives:
    refusal: SeverityCode
    force_hold: bool
    force_ban: bool
    freeze_learning: bool
    max_abs_exposure: float
    reason: str


@dataclass(frozen=True)
class Phase9Decision:
    directives: MetaWitnessDirectives
    ledger: CapitalLedgerRow
    expected_surplus: float
    expected_gross_surplus: float
    expected_cost: float
    cost_survival_ratio: float
    proposed_expected_surplus: float
    proposed_expected_gross_surplus: float
    proposed_expected_cost: float
    proposed_cost_survival_ratio: float
    executed_expected_surplus: float
    executed_expected_gross_surplus: float
    executed_expected_cost: float
    executed_cost_survival_ratio: float
    microstructure_kills_edge: bool
    continuation_active: bool
    continuation_supported: bool
    continuation_expected_surplus: float
    hazard_tightened: bool
    hazard_level: float
    justification_chain: str
    regime_label: str
    posture_label: str
    actuator_label: str
    cost_model_label: str


def estimate_expected_surplus(
    *,
    capital: float,
    current_exposure: float,
    proposed_exposure: float,
    edge: float,
    actionability: float,
    kappa_t: float,
    spread_bps: float = 0.0,
) -> tuple[float, float, float, float]:
    delta = abs(proposed_exposure - current_exposure)
    expected_gross = max(0.0, capital) * delta * max(0.0, abs(edge)) * max(0.0, actionability)
    spread_cost = max(0.0, capital) * delta * max(0.0, spread_bps) * 1e-4
    expected_cost = max(0.0, capital) * delta * max(0.0, kappa_t) + spread_cost
    expected_surplus = expected_gross - expected_cost
    if expected_cost > 0.0:
        cost_survival_ratio = expected_gross / expected_cost
    elif expected_gross > 0.0:
        cost_survival_ratio = float("inf")
    else:
        cost_survival_ratio = 0.0
    return expected_surplus, expected_gross, expected_cost, cost_survival_ratio


def estimate_continuation_surplus(
    *,
    capital: float,
    current_exposure: float,
    edge: float,
    actionability: float,
) -> tuple[float, float, float, float]:
    holding = abs(float(current_exposure))
    expected_gross = max(0.0, capital) * holding * max(0.0, abs(edge)) * max(0.0, actionability)
    expected_cost = 0.0
    expected_surplus = expected_gross
    cost_survival_ratio = float("inf") if expected_gross > 0.0 else 0.0
    return expected_surplus, expected_gross, expected_cost, cost_survival_ratio


def microstructure_survival_floor(
    *,
    edge: float,
    actionability: float,
    hazard_level: float = 0.0,
    params: CapitalParams,
) -> float:
    signal_strength = max(0.0, abs(edge)) * max(0.0, actionability)
    relief = params.microstructure_relief_strength * signal_strength
    tightened_floor = params.microstructure_survival_floor - relief + params.hazard_survival_floor_add * max(0.0, hazard_level)
    return max(params.microstructure_survival_floor_min, tightened_floor)


def estimate_survivability_score(
    *,
    actionability: float,
    edge_persistence: float,
    edge_shock: float,
    microstructure_pressure: float,
    hazard_level: float,
    family_cooldown: float,
    params: CapitalParams,
) -> float:
    support = (
        0.45 * max(0.0, min(1.0, actionability))
        + 0.40 * max(0.0, min(1.0, edge_persistence))
        + 0.15 * max(0.0, 1.0 - min(1.0, microstructure_pressure))
    )
    penalty = (
        params.survivability_shock_weight * max(0.0, min(1.0, edge_shock))
        + params.survivability_hazard_weight * max(0.0, min(1.0, hazard_level))
        + params.survivability_cooldown_weight * max(0.0, min(1.0, family_cooldown))
    )
    return max(0.0, min(1.0, support - penalty))


def normalize_hazard_level(
    *,
    hazard: float = 0.0,
) -> float:
    if hazard != hazard:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, float(hazard)))


def effective_hazard_level(
    *,
    hazard: float = 0.0,
    params: CapitalParams,
) -> float:
    raw = normalize_hazard_level(hazard=hazard)
    if raw <= params.hazard_reentry_threshold:
        return 0.0
    if raw >= params.hazard_clamp_threshold:
        return raw
    span = max(params.hazard_clamp_threshold - params.hazard_reentry_threshold, 1e-9)
    progress = (raw - params.hazard_reentry_threshold) / span
    relief = params.hazard_reentry_relief + (1.0 - params.hazard_reentry_relief) * progress
    return max(0.0, min(1.0, raw * relief))


def update_capital(
    *,
    capital_prev: float,
    exposure_prev: float,
    price_return: float,
    dx: float,
    kappa_t: float,
    posture_active: bool,
    drawdown: float,
    sigma: float,
    params: CapitalParams,
) -> CapitalLedgerRow:
    pnl_mtm = capital_prev * exposure_prev * price_return
    friction_cost = capital_prev * kappa_t * abs(dx)
    risk_tax = capital_prev * (
        params.lambda_dd * max(0.0, drawdown)
        + params.lambda_sigma * max(0.0, sigma)
        + (0.0 if posture_active else params.lambda_posture)
    )
    realized_surplus = pnl_mtm - friction_cost
    capital_next = max(params.capital_floor, capital_prev + realized_surplus - risk_tax)
    capital_dd = 0.0
    if capital_prev > 0.0:
        capital_dd = max(0.0, 1.0 - (capital_next / capital_prev))
    return CapitalLedgerRow(
        capital_prev=capital_prev,
        capital_next=capital_next,
        capital_dd=capital_dd,
        kappa_t=kappa_t,
        pnl_mtm=pnl_mtm,
        friction_cost=friction_cost,
        risk_tax=risk_tax,
        realized_surplus=realized_surplus,
        dx=dx,
    )


def clamp_exposure(
    *,
    proposed_exposure: float,
    capital: float,
    kappa_t: float,
    drawdown: float,
    hazard_level: float = 0.0,
    params: CapitalParams,
) -> float:
    effective_capital = max(capital, params.capital_floor)
    denom = max(1e-9, effective_capital * (kappa_t + 1e-9))
    budget = params.beta * effective_capital
    cap_limit = min(1.0, budget / denom)
    dd_scale = max(0.0, 1.0 - max(0.0, drawdown) / max(params.drawdown_limit, 1e-9))
    hazard_scale = max(
        params.hazard_min_exposure_scale,
        1.0 - params.hazard_exposure_tightening * max(0.0, hazard_level),
    )
    cap_limit *= hazard_scale
    return max(-cap_limit * dd_scale, min(cap_limit * dd_scale, proposed_exposure))


def evaluate_meta_witness(
    *,
    state: MetaWitnessState,
    params: CapitalParams,
) -> MetaWitnessDirectives:
    codes: list[SeverityCode] = []
    reasons: list[str] = []

    if not state.phase6_open:
        codes.append(SeverityCode(SeverityLevel.BAN, "no_phase6_authority"))
        reasons.append("R0")
    if not state.phase7_ready:
        level = SeverityLevel.BAN if state.turnover > 0.05 else SeverityLevel.HOLD
        codes.append(SeverityCode(level, "net_asymmetry_nonpositive"))
        reasons.append("R1")
    sparse_support = (
        state.actionability < params.min_actionability
        or state.expected_surplus <= params.min_expected_surplus
    )
    if sparse_support and not (state.continuation_active and state.continuation_supported):
        codes.append(SeverityCode(SeverityLevel.HOLD, "sparse_support"))
        reasons.append("R2")
    if state.drawdown > params.drawdown_limit:
        codes.append(SeverityCode(SeverityLevel.BAN, "capital_drawdown_breach"))
        reasons.append("R3")
    if state.turnover > 0.25 and state.expected_surplus <= 0.0:
        codes.append(SeverityCode(SeverityLevel.BAN, "cost_dominated_churn"))
        reasons.append("R4")
    if state.continuation_active and not state.continuation_supported:
        codes.append(SeverityCode(SeverityLevel.BAN, "continuation_unviable"))
        reasons.append("R8")
    if state.hazard >= params.hazard_ban_threshold and state.turnover > params.microstructure_min_turnover:
        codes.append(SeverityCode(SeverityLevel.BAN, "hazard_hostile"))
        reasons.append("R7")
    elif state.hazard >= params.hazard_hold_threshold and state.turnover > params.microstructure_min_turnover:
        codes.append(SeverityCode(SeverityLevel.HOLD, "hazard_observe"))
        reasons.append("R7")
    elif state.hazard >= params.hazard_clamp_threshold:
        codes.append(SeverityCode(SeverityLevel.CAUTION, "hazard_tightening"))
        reasons.append("R7")
    if state.microstructure_kills_edge:
        level = SeverityLevel.BAN if state.turnover > 0.05 else SeverityLevel.HOLD
        codes.append(SeverityCode(level, "microstructure_kills_edge"))
        reasons.append("R6")
    if state.prior_refusal_active and abs(state.proposed_exposure - state.current_exposure) > 1e-9:
        codes.append(SeverityCode(SeverityLevel.PARADOX, "policy_inconsistency"))
        reasons.append("R5")

    refusal = combine_codes(*codes)
    if refusal.level >= SeverityLevel.BAN:
        max_abs_exposure = 0.0
    elif refusal.level >= SeverityLevel.HOLD:
        max_abs_exposure = abs(state.current_exposure)
    else:
        max_abs_exposure = max(
            params.hazard_min_exposure_scale,
            1.0 - params.hazard_exposure_tightening * max(0.0, state.hazard),
        )
    return MetaWitnessDirectives(
        refusal=refusal,
        force_hold=refusal.level >= SeverityLevel.HOLD,
        force_ban=refusal.level >= SeverityLevel.BAN,
        freeze_learning=refusal.level >= SeverityLevel.BAN,
        max_abs_exposure=max_abs_exposure,
        reason=refusal.label if not reasons else f"{refusal.label}:{'+'.join(reasons)}",
    )


def make_phase9_decision(
    *,
    capital_prev: float,
    exposure_prev: float,
    proposed_exposure: float,
    price_return: float,
    edge: float,
    actionability: float,
    drawdown: float,
    sigma: float,
    hazard: float = 0.0,
    spread_bps: float = 0.0,
    phase6_open: bool = True,
    phase7_ready: bool = True,
    prior_refusal_active: bool = False,
    params: CapitalParams | None = None,
) -> Phase9Decision:
    params = params or CapitalParams()
    kappa_t = params.fee_rate + params.slippage_rate + params.margin_rate
    hazard_level = effective_hazard_level(hazard=hazard, params=params)
    turnover = abs(proposed_exposure - exposure_prev)
    continuation_active = abs(exposure_prev) > 1e-9 and turnover <= 1e-9
    proposed_expected_surplus, proposed_expected_gross_surplus, proposed_expected_cost, proposed_cost_survival_ratio = estimate_expected_surplus(
        capital=capital_prev,
        current_exposure=exposure_prev,
        proposed_exposure=proposed_exposure,
        edge=edge,
        actionability=actionability,
        kappa_t=kappa_t,
        spread_bps=spread_bps,
    )
    continuation_expected_surplus, continuation_expected_gross_surplus, continuation_expected_cost, continuation_cost_survival_ratio = estimate_continuation_surplus(
        capital=capital_prev,
        current_exposure=exposure_prev,
        edge=edge,
        actionability=actionability,
    )
    continuation_actionability_floor = min(params.min_actionability, 0.10)
    continuation_supported = (
        continuation_active
        and abs(edge) >= params.min_edge
        and actionability >= continuation_actionability_floor
        and continuation_expected_surplus > params.min_expected_surplus
    )
    effective_survival_floor = microstructure_survival_floor(
        edge=edge,
        actionability=actionability,
        hazard_level=hazard_level,
        params=params,
    )
    trade_attempt_active = (
        turnover > params.microstructure_min_turnover
        and proposed_expected_gross_surplus > params.microstructure_min_expected_gross
    )
    microstructure_kills_edge = trade_attempt_active and proposed_cost_survival_ratio < effective_survival_floor
    clamped_exposure = clamp_exposure(
        proposed_exposure=proposed_exposure,
        capital=capital_prev,
        kappa_t=kappa_t,
        drawdown=drawdown,
        hazard_level=hazard_level,
        params=params,
    )
    directives = evaluate_meta_witness(
        state=MetaWitnessState(
            phase6_open=phase6_open,
            phase7_ready=(
                phase7_ready and continuation_supported
                if continuation_active
                else phase7_ready and proposed_expected_surplus > params.min_expected_surplus
            ),
            actionability=actionability,
            expected_surplus=continuation_expected_surplus if continuation_active else proposed_expected_surplus,
            drawdown=drawdown,
            turnover=abs(clamped_exposure - exposure_prev),
            stress=sigma,
            spread_bps=spread_bps,
            cost_survival_ratio=proposed_cost_survival_ratio,
            microstructure_kills_edge=microstructure_kills_edge,
            current_exposure=exposure_prev,
            proposed_exposure=clamped_exposure,
            hazard=hazard_level,
            prior_refusal_active=prior_refusal_active,
            continuation_active=continuation_active,
            continuation_supported=continuation_supported,
            continuation_expected_surplus=continuation_expected_surplus,
        ),
        params=params,
    )
    if directives.force_ban:
        final_exposure = 0.0
    elif directives.force_hold:
        final_exposure = exposure_prev
    else:
        final_exposure = max(-directives.max_abs_exposure, min(directives.max_abs_exposure, clamped_exposure))

    executed_expected_surplus, executed_expected_gross_surplus, executed_expected_cost, executed_cost_survival_ratio = estimate_expected_surplus(
        capital=capital_prev,
        current_exposure=exposure_prev,
        proposed_exposure=final_exposure,
        edge=edge,
        actionability=actionability,
        kappa_t=kappa_t,
        spread_bps=spread_bps,
    )

    ledger = update_capital(
        capital_prev=capital_prev,
        exposure_prev=exposure_prev,
        price_return=price_return,
        dx=final_exposure - exposure_prev,
        kappa_t=kappa_t,
        posture_active=not directives.force_hold,
        drawdown=drawdown,
        sigma=sigma,
        params=params,
    )
    hazard_tightened = hazard_level >= params.hazard_clamp_threshold
    if hazard_level >= params.hazard_ban_threshold:
        regime_label = "hazard_hostile"
    elif hazard_tightened:
        regime_label = "hazard_tightened"
    elif continuation_active and not continuation_supported:
        regime_label = "continuation_unviable"
    elif continuation_active and continuation_supported:
        regime_label = "continuation_supported"
    elif microstructure_kills_edge:
        regime_label = "microstructure_kills_edge"
    elif spread_bps > 5.0:
        regime_label = "spread_stressed_edge"
    else:
        regime_label = "positive_edge" if abs(edge) >= params.min_edge else "flat_edge"
    posture_label = "observe" if directives.force_hold else "trade_normal"
    actuator_label = "bar_exec"
    cost_model_label = "phase9_capital_kernel_v2_spread"
    justification_chain = " -> ".join(
        [
            regime_label,
            posture_label,
            actuator_label,
            cost_model_label,
            f"proposed_expected_gross={proposed_expected_gross_surplus:.6f}",
            f"proposed_expected_cost={proposed_expected_cost:.6f}",
            f"proposed_cost_survival_ratio={proposed_cost_survival_ratio:.6f}",
            f"proposed_expected_surplus={proposed_expected_surplus:.6f}",
            f"continuation_active={int(continuation_active)}",
            f"continuation_expected_surplus={continuation_expected_surplus:.6f}",
            f"continuation_supported={int(continuation_supported)}",
            f"hazard_level={hazard_level:.6f}",
            f"executed_expected_gross={executed_expected_gross_surplus:.6f}",
            f"executed_expected_cost={executed_expected_cost:.6f}",
            f"executed_cost_survival_ratio={executed_cost_survival_ratio:.6f}",
            f"executed_expected_surplus={executed_expected_surplus:.6f}",
            f"realized_surplus={ledger.realized_surplus:.6f}",
        ]
    )
    return Phase9Decision(
        directives=directives,
        ledger=ledger,
        expected_surplus=executed_expected_surplus,
        expected_gross_surplus=executed_expected_gross_surplus,
        expected_cost=executed_expected_cost,
        cost_survival_ratio=executed_cost_survival_ratio,
        proposed_expected_surplus=proposed_expected_surplus,
        proposed_expected_gross_surplus=proposed_expected_gross_surplus,
        proposed_expected_cost=proposed_expected_cost,
        proposed_cost_survival_ratio=proposed_cost_survival_ratio,
        executed_expected_surplus=executed_expected_surplus,
        executed_expected_gross_surplus=executed_expected_gross_surplus,
        executed_expected_cost=executed_expected_cost,
        executed_cost_survival_ratio=executed_cost_survival_ratio,
        microstructure_kills_edge=microstructure_kills_edge,
        continuation_active=continuation_active,
        continuation_supported=continuation_supported,
        continuation_expected_surplus=continuation_expected_surplus,
        hazard_tightened=hazard_tightened,
        hazard_level=hazard_level,
        justification_chain=justification_chain,
        regime_label=regime_label,
        posture_label=posture_label,
        actuator_label=actuator_label,
        cost_model_label=cost_model_label,
    )


def assess_proposal_economics(
    *,
    capital_prev: float,
    exposure_prev: float,
    proposed_exposure: float,
    edge: float,
    actionability: float,
    spread_bps: float = 0.0,
    drawdown: float = 0.0,
    hazard: float = 0.0,
    edge_persistence: float = 0.0,
    edge_shock: float = 0.0,
    microstructure_pressure: float = 0.0,
    family_cooldown: float = 0.0,
    params: CapitalParams | None = None,
) -> ProposalEconomics:
    params = params or CapitalParams()
    kappa_t = params.fee_rate + params.slippage_rate + params.margin_rate
    hazard_level = effective_hazard_level(hazard=hazard, params=params)
    expected_surplus, expected_gross_surplus, expected_cost, cost_survival_ratio = estimate_expected_surplus(
        capital=capital_prev,
        current_exposure=exposure_prev,
        proposed_exposure=proposed_exposure,
        edge=edge,
        actionability=actionability,
        kappa_t=kappa_t,
        spread_bps=spread_bps,
    )
    turnover = abs(proposed_exposure - exposure_prev)
    effective_survival_floor = microstructure_survival_floor(
        edge=edge,
        actionability=actionability,
        hazard_level=hazard_level,
        params=params,
    )
    survivability_score = estimate_survivability_score(
        actionability=actionability,
        edge_persistence=edge_persistence,
        edge_shock=edge_shock,
        microstructure_pressure=microstructure_pressure,
        hazard_level=hazard_level,
        family_cooldown=family_cooldown,
        params=params,
    )
    survivability_viable = survivability_score >= params.survivability_min
    cost_viable = (
        turnover <= params.microstructure_min_turnover
        or expected_gross_surplus <= params.microstructure_min_expected_gross
        or cost_survival_ratio >= effective_survival_floor
    )
    governance_viable = (
        actionability >= params.min_actionability
        and expected_surplus > params.min_expected_surplus
        and drawdown <= params.drawdown_limit
        and hazard_level < params.hazard_hold_threshold
        and survivability_viable
    )
    executable_viable = governance_viable and cost_viable

    if turnover <= params.microstructure_min_turnover:
        viability_reason = "below_turnover_floor"
    elif expected_gross_surplus <= params.microstructure_min_expected_gross:
        viability_reason = "below_gross_floor"
    elif drawdown > params.drawdown_limit:
        viability_reason = "drawdown_breach"
    elif actionability < params.min_actionability:
        viability_reason = "low_actionability"
    elif expected_surplus <= params.min_expected_surplus:
        viability_reason = "nonpositive_expected_surplus"
    elif hazard_level >= params.hazard_ban_threshold:
        viability_reason = "hazard_hostile"
    elif hazard_level >= params.hazard_hold_threshold:
        viability_reason = "hazard_observe"
    elif family_cooldown > 1e-9:
        viability_reason = "post_unwind_cooldown"
    elif not survivability_viable:
        viability_reason = "survivability_boundary"
    elif cost_survival_ratio < effective_survival_floor:
        viability_reason = "microstructure_kills_edge"
    else:
        viability_reason = "executable"

    return ProposalEconomics(
        proposed_exposure=proposed_exposure,
        turnover=turnover,
        expected_surplus=expected_surplus,
        expected_gross_surplus=expected_gross_surplus,
        expected_cost=expected_cost,
        cost_survival_ratio=cost_survival_ratio,
        governance_viable=governance_viable,
        cost_viable=cost_viable,
        executable_viable=executable_viable,
        survivability_viable=survivability_viable,
        survivability_score=survivability_score,
        viability_reason=viability_reason,
    )

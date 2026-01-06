from __future__ import annotations

from dataclasses import dataclass

from ternary import clip_ternary_sum


@dataclass
class ThesisState:
    d: int
    s: int
    a: int
    c: int
    v: int


@dataclass
class ThesisInputs:
    plane_sign: int
    plane_sign_flips_w: int
    plane_would_veto: int
    stress: float
    p_bad: float
    shadow_would_promote: int


@dataclass
class ThesisParams:
    a_max: int
    cooldown: int
    p_bad_lo: float
    p_bad_hi: float
    stress_lo: float
    stress_hi: float


@dataclass
class ThesisDerived:
    alpha: int
    beta: int
    rho: int
    ds: int
    sum: int


@dataclass
class ThesisEvent:
    event: str
    reason: str
    exit_trigger: bool


def step_thesis_memory(state: ThesisState, inputs: ThesisInputs, params: ThesisParams):
    d_star = inputs.plane_sign if state.d == 0 else state.d
    if inputs.plane_sign == 0:
        alpha = 0
    elif inputs.plane_sign == d_star:
        alpha = 1
    else:
        alpha = -1
    if inputs.plane_would_veto == 1 or inputs.plane_sign_flips_w > 1:
        beta = -1
    elif inputs.plane_sign_flips_w <= 1:
        beta = 1
    else:
        beta = 0
    if inputs.p_bad >= params.p_bad_hi or inputs.stress >= params.stress_hi:
        rho = -1
    elif inputs.p_bad <= params.p_bad_lo and inputs.stress <= params.stress_lo:
        rho = 1
    else:
        rho = 0
    sum_trits = alpha + beta + rho
    ds = clip_ternary_sum(sum_trits)
    derived = ThesisDerived(alpha=alpha, beta=beta, rho=rho, ds=ds, sum=sum_trits)
    c_next = state.c - 1 if state.c > 0 else 0
    event = ThesisEvent(event="thesis_none", reason="", exit_trigger=False)
    if state.d == 0:
        if (
            c_next == 0
            and inputs.shadow_would_promote == 1
            and beta >= 0
            and rho >= 0
            and d_star != 0
        ):
            new_state = ThesisState(d=d_star, s=1, a=0, c=0, v=0)
            event = ThesisEvent(event="thesis_enter", reason="shadow_promote", exit_trigger=False)
        else:
            new_state = ThesisState(d=0, s=0, a=0, c=c_next, v=0)
    else:
        s_tmp = max(0, min(2, state.s + ds))
        a_tmp = state.a + 1
        bad_evidence = alpha == -1 or beta == -1 or rho == -1
        if bad_evidence:
            v_tmp = min(2, state.v + 1)
        else:
            v_tmp = max(0, state.v - 1)
        exit_reason = ""
        if v_tmp == 2:
            exit_reason = "thesis_invalidated"
        elif s_tmp == 0:
            exit_reason = "thesis_decay"
        elif a_tmp >= params.a_max:
            exit_reason = "thesis_timeout"
        if exit_reason:
            new_state = ThesisState(d=0, s=0, a=0, c=params.cooldown, v=0)
            event = ThesisEvent(event="thesis_exit", reason=exit_reason, exit_trigger=True)
        else:
            new_state = ThesisState(d=state.d, s=s_tmp, a=a_tmp, c=c_next, v=v_tmp)
            if ds > 0:
                reason = "reinforce"
            elif ds < 0:
                reason = "decay"
            else:
                reason = "hold"
            event = ThesisEvent(event="thesis_update", reason=reason, exit_trigger=False)
    return new_state, derived, event


def apply_thesis_constraints(
    thesis_d: int,
    derived: ThesisDerived,
    proposed_action: int,
    hard_veto: bool,
    exit_trigger: bool,
):
    action_t = proposed_action
    override = ""
    if thesis_d == 0:
        return action_t, override
    if proposed_action == -thesis_d:
        if exit_trigger:
            action_t = 0
        else:
            if derived.beta == -1 or derived.rho == -1 or hard_veto:
                action_t = 0
            else:
                action_t = thesis_d
        override = "thesis_no_flipflop"
    elif proposed_action == 0:
        if hard_veto:
            action_t = 0
        elif derived.beta >= 0 and derived.rho >= 0 and not exit_trigger:
            action_t = thesis_d
            override = "thesis_hold_bias"
        else:
            action_t = 0
            if derived.beta == -1:
                override = "thesis_unstable"
            elif derived.rho == -1:
                override = "thesis_risk"
    return action_t, override

from __future__ import annotations


def compute_cap(
    volume_t: float,
    ret: float,
    z_vel: float,
    sigma_target: float,
    sigma: float,
    veto_sigma: float,
    cash: float,
    pos: float,
    price_t: float,
    risk_frac: float | None,
    contract_mult: float,
    participation_cap: float,
    cap_hard_max: float,
    edge_gate: bool,
    edge_t: int,
    plane_index: int,
    edge_decay: float,
    risk_headroom_low: float,
    risk_headroom_high: float,
) -> tuple[float, int]:
    base_cap = participation_cap * volume_t
    vel_term = 1.0 + 10.0 * max(abs(ret), z_vel)
    cap = base_cap * vel_term
    if sigma_target and sigma_target > 0:
        cap *= sigma_target / max(sigma, 1e-9)
        if sigma > veto_sigma * sigma_target:
            cap *= 0.2
    equity = cash + pos * price_t
    if risk_frac:
        risk_cap = (equity * risk_frac) / (price_t * contract_mult + 1e-9)
        cap = min(cap, risk_cap)
    cap = max(0.0, min(cap, cap_hard_max))
    if edge_gate and edge_t == -1 and plane_index <= 0:
        cap *= edge_decay
    risk_headroom = cap / max(cap_hard_max, 1e-9)
    if risk_headroom > risk_headroom_high:
        risk_budget = 1
        cap *= 1.0
    elif risk_headroom < risk_headroom_low:
        risk_budget = -1
        cap *= 0.2
    else:
        risk_budget = 0
        cap *= 0.5
    cap = max(0.0, min(cap, cap_hard_max))
    return cap, risk_budget

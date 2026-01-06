from __future__ import annotations

import numpy as np


def compute_fill(
    action_t: int,
    pos: float,
    cap: float,
    hold_decay: float,
    z_vel: float,
    vel_exit: float,
    thesis_hold: bool,
    persist_ramp: float,
    align_age: int,
) -> tuple[float, float]:
    cap_used = cap
    if action_t == 0:
        cap_used = cap * (1.0 - hold_decay) + 1e-9
        fill = np.clip(-pos, -cap_used, cap_used)
    else:
        if z_vel > vel_exit and pos != 0:
            fill = np.clip(-pos, -cap_used, cap_used)
        else:
            if thesis_hold:
                step = 0.0
            else:
                target = action_t * cap_used
                ramp = persist_ramp * (1.0 + align_age * 0.01)
                step = ramp * (target - pos)
            fill = np.clip(step, -cap_used, cap_used)
    return fill, cap_used


def apply_execution(
    price_t: float,
    fill: float,
    cap: float,
    cash: float,
    pos: float,
    cost: float,
    impact_coeff: float,
) -> tuple[float, float, float, float, float, float]:
    slippage = impact_coeff * abs(fill / max(cap, 1e-9))
    price_exec = price_t * (1 + slippage * np.sign(fill))
    cash -= fill * price_exec
    pos += fill
    fee = cost * abs(fill)
    slippage_cost = abs(fill) * abs(price_exec - price_t)
    capital_at_risk = abs(fill * price_exec)
    return price_exec, cash, pos, fee, slippage_cost, capital_at_risk

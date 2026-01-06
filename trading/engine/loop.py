import pathlib
import time
from collections import deque

import numpy as np
import pandas as pd

from execution.accounting import build_run_summary, compute_step_accounting
from execution.fills import apply_execution, compute_fill
from execution.sizing import compute_cap
from policy.thesis import (
    ThesisDerived,
    ThesisEvent,
    ThesisInputs,
    ThesisParams,
    ThesisState,
    apply_thesis_constraints,
    step_thesis_memory,
)
from signals.stress import compute_structural_stress
from signals.triadic import compute_triadic_state
from ternary import clip_ternary_sum, ternary_controller, ternary_permission, ternary_sign
from trading_io.logs import (
    emit_progress_print,
    emit_run_summary,
    emit_step_row,
    emit_trade_close_print,
    emit_trade_print,
    emit_trade_row,
)
from utils.stats import norm_cdf, norm_pdf, norm_ppf

LOG = pathlib.Path("logs/trading_log.csv")
LOG.parent.mkdir(parents=True, exist_ok=True)
RUN_HISTORY = pathlib.Path("data/run_history.csv")
RUN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
PARTICIPATION_CAP = 0.05   # max fraction of bar volume we can trade
IMPACT_COEFF = 0.0001      # slippage per unit participation
SIGMA_TARGET = 0.01        # target vol for basic risk parity sizing
DEFAULT_RISK_FRAC = None   # optional: fraction of equity to risk per trade (futures-style)
CONTRACT_MULT = 1.0        # notional per contract; leave at 1 unless using real futures
CAP_HARD_MAX = 100.0       # absolute cap on per-step size (after all scaling)
START_CASH = 100000.0      # starting cash to give non-zero risk budget
EST_TAX_RATE = 0.25        # estimated tax on realized PnL
GOAL_CASH_X = START_CASH + 50_000.0  # spendable cash target
GOAL_EPS = 0.05            # tail fraction for expected shortfall
MDL_NOISE_MULT = 2.0       # sigma multiplier for active-trit threshold
MDL_SWITCH_PENALTY = 1.0   # side cost for state switches
MDL_TRADE_PENALTY = 1.0    # side cost for executed trades
SHADOW_REFIT_WINDOW = 64   # refit window size for shadow MDL split diagnostic
PLANE_FLIP_WINDOW = SHADOW_REFIT_WINDOW  # rolling window for plane sign flips
SHADOW_SPLIT_PENALTY_MULT = 1.0  # split penalty multiplier (log n)
SHADOW_MDL_EPS_MULT = 1e-12  # scale-aware epsilon multiplier for promote/tie/reject
PLANE_BASE = 3.0           # base for surprise planes
PLANE_COUNT = 4            # number of surprise planes to track (0..PLANE_COUNT-1)
PLANE_SIGMA_SLOW_ALPHA = 0.002  # slow sigma EMA for plane normalization
EDGE_EMA_ALPHA = 0.002     # slow EMA for exposure-normalized edge
THESIS_DEPTH_MAX = 6       # max depth for thesis memory counter

# Controls
HOLD_DECAY = 0.6           # exposure decay factor when action -> HOLD
VEL_EXIT = 3.0             # exit if latent velocity exceeds this while in position
PERSIST_RAMP = 0.05        # ramp factor for size in new regime
VETO_SIGMA = 5.0           # if realized sigma > VETO_SIGMA * sigma_target -> shrink size
PBAD_CAUTION = 0.4         # caution threshold for ternary permission
PBAD_BAN = 0.7             # ban threshold for ternary permission
K_LATENT_TAU = 0.25        # capital pressure dead-zone
RISK_HEADROOM_LOW = 0.2    # ternary risk budget low threshold
RISK_HEADROOM_HIGH = 0.5   # ternary risk budget high threshold

# Thesis memory (FSM) defaults
THESIS_MEMORY_DEFAULT = False
THESIS_A_MAX = 200
THESIS_COOLDOWN = 5
THESIS_PBAD_LO = PBAD_CAUTION
THESIS_PBAD_HI = PBAD_BAN
THESIS_STRESS_LO = 0.2
THESIS_STRESS_HI = 0.5
THESIS_TC_K = 0.0
THESIS_BENCHMARK_X = 1.0

# --- State helper ----------------------------------------------------------


def compute_regret_reward(prev_action: int, log_ret: float, benchmark_x: float, tc_step: float) -> float:
    return (prev_action - benchmark_x) * log_ret - tc_step


BELIEF_UNKNOWN = -1


def clip_belief(val: int) -> int:
    return max(0, min(2, val))


def update_belief(belief: int, delta: int, alpha: int, beta: int, rho: int, perm: int) -> int:
    if perm == -1:
        return 0
    if alpha == 0 and beta == -1:
        return BELIEF_UNKNOWN
    if belief == BELIEF_UNKNOWN:
        return max(0, delta)
    return clip_belief(belief + delta)


def belief_state_label(b_plus: int, b_minus: int) -> str:
    if b_plus == BELIEF_UNKNOWN and b_minus == BELIEF_UNKNOWN:
        return "unk"
    if b_plus == 0 and b_minus == 0:
        return "flat"
    if b_plus == 2 and b_minus == 2:
        return "conflict"
    if b_plus == 2 and b_plus > b_minus:
        return "l2"
    if b_minus == 2 and b_minus > b_plus:
        return "s2"
    if b_plus == 1 and b_minus <= 1:
        return "l1"
    if b_minus == 1 and b_plus <= 1:
        return "s1"
    return "unk"


def fmt_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_trading_loop(
    price: np.ndarray,
    volume: np.ndarray,
    source: str,
    time_index=None,
    max_steps=None,
    max_trades=None,
    max_seconds=None,
    sleep_s=0.0,
    risk_frac: float = DEFAULT_RISK_FRAC,
    contract_mult: float = CONTRACT_MULT,
    sigma_target: float = SIGMA_TARGET,
    log_path: pathlib.Path = LOG,
    trade_log_path: pathlib.Path | None = None,
    progress_every: int = 0,
    est_tax_rate: float = EST_TAX_RATE,
    goal_cash_x: float = GOAL_CASH_X,
    goal_eps: float = GOAL_EPS,
    mdl_noise_mult: float = MDL_NOISE_MULT,
    mdl_switch_penalty: float = MDL_SWITCH_PENALTY,
    mdl_trade_penalty: float = MDL_TRADE_PENALTY,
    log_level: str = "info",
    log_append: bool = False,
    tape_id: str | None = None,
    edge_ema_alpha: float = EDGE_EMA_ALPHA,
    edge_gate: bool = False,
    edge_decay: float = 0.9,
    thesis_depth_max: int = THESIS_DEPTH_MAX,
    thesis_memory: bool = THESIS_MEMORY_DEFAULT,
    thesis_a_max: int = THESIS_A_MAX,
    thesis_cooldown: int = THESIS_COOLDOWN,
    thesis_pbad_lo: float = THESIS_PBAD_LO,
    thesis_pbad_hi: float = THESIS_PBAD_HI,
    thesis_stress_lo: float = THESIS_STRESS_LO,
    thesis_stress_hi: float = THESIS_STRESS_HI,
    tc_k: float = THESIS_TC_K,
    benchmark_x: float = THESIS_BENCHMARK_X,
):
    def shadow_mdl_for_window(ret_window):
        n = len(ret_window)
        if n <= 1:
            return np.nan
        mu = float(np.mean(ret_window))
        var = float(np.var(ret_window))
        var = max(var, 1e-12)
        residual = ret_window - mu
        nll = 0.5 * n * np.log(var) + 0.5 * float(np.sum((residual ** 2) / var))
        param_cost = np.log(max(n, 1.0))
        return nll + param_cost
    """
    Core trading loop extracted so multiple markets can be evaluated.
    """
    price = np.asarray(price, dtype=float)
    volume = np.asarray(volume, dtype=float)
    if time_index is not None:
        time_index = np.asarray(time_index)
        if len(time_index) != len(price):
            raise ValueError("time_index length must match price length")
    log_path = pathlib.Path(log_path) if log_path is not None else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_append:
            log_path.unlink(missing_ok=True)
    if trade_log_path:
        trade_log_path = pathlib.Path(trade_log_path)
        trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_append:
            trade_log_path.unlink(missing_ok=True)
    # Precompute triadic states and structural stress (for bad_day signal)
    pre_states = compute_triadic_state(price)
    p_bad, bad_flag = compute_structural_stress(price, pre_states)

    cash = START_CASH
    prev_cash = cash
    prev_pnl = START_CASH
    fees_accrued = 0.0
    c_spend_prev = cash
    pos = 0.0
    z_prev = 0.0
    prev_action = 0
    prev_goal_prob = 0.0
    active_trit_count = 0.0
    plane_counts = [0.0 for _ in range(PLANE_COUNT)]
    plane_k_prev = 0
    plane_sign_prev = 0
    plane_flip_flags = deque(maxlen=max(1, PLANE_FLIP_WINDOW))
    sigma_slow = 0.0
    edge_ema = 0.0
    side_cost = 0.0
    action_run_length = 0
    time_since_last_switch = 0
    thesis_age = 0      # how long we've held a non-zero thesis
    thesis_depth = 0    # bounded ordinal memory counter
    thesis_depth_peak = 0  # max depth reached during current trade
    state_age = 0       # how long the field state has persisted
    align_age = 0       # how long state and thesis have been aligned
    prev_state = 0
    capital_pressure = 0
    thesis_d = 0
    thesis_s = 0
    thesis_a = 0
    thesis_c = 0
    thesis_v = 0
    belief_plus = BELIEF_UNKNOWN
    belief_minus = BELIEF_UNKNOWN
    trade_id = 0
    trade_entry_step = None
    trade_entry_price = 0.0
    trade_entry_notional = 0.0
    trade_realized_pnl = 0.0
    avg_entry_price = 0.0
    realized_pnl_total = 0.0
    dz_min = 5e-5  # minimum dead-zone
    cost = 0.0005
    rows = []
    recent_rets = []
    fill_count = 0
    closed_trade_count = 0
    start_ts = time.time()
    stop_reason = ""
    total_steps = len(price) - 1
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    pnl_delta_window = []
    window = 200
    t0_ts = time_index[0] if time_index is not None and len(time_index) else None
    returns = np.diff(price) / price[:-1]
    for t in range(1, total_steps + 1):
        if max_seconds is not None and (time.time() - start_ts) >= max_seconds:
            stop_reason = "max_seconds"
            break
        if max_trades is not None and closed_trade_count >= max_trades:
            stop_reason = "max_trades"
            break
        ret = price[t] / price[t - 1] - 1.0
        price_change = price[t] - price[t - 1]
        recent_rets.append(ret)
        if len(recent_rets) > 200:
            recent_rets.pop(0)
        sigma = np.std(recent_rets) if recent_rets else dz_min
        sigma_slow = (
            (1.0 - PLANE_SIGMA_SLOW_ALPHA) * sigma_slow
            + PLANE_SIGMA_SLOW_ALPHA * abs(ret)
        )
        # latent update (EWMA of volatility-normalized returns)
        z_update = ret / (sigma + 1e-9)
        z_update = np.clip(z_update, -5.0, 5.0)
        z = 0.95 * z_prev + 0.05 * z_update
        z_vel = abs(z - z_prev)
        # dead-zone in the same (dimensionless) space as z
        dz = max(0.25, dz_min)
        if abs(z) < dz:
            desired = 0
        elif z > 0:
            desired = 1
        else:
            desired = -1
        direction = desired
        thesis = thesis_d if thesis_memory else int(np.sign(pos))
        # Plane classification for this step (used for logging/gating)
        abs_ret = abs(ret)
        noise_floor = max(sigma_slow, 1e-9)
        noise_threshold = mdl_noise_mult * noise_floor
        if abs_ret > noise_threshold:
            plane_k = int(np.floor(np.log(abs_ret / noise_threshold) / np.log(PLANE_BASE)))
            plane_index = max(0, min(plane_k, PLANE_COUNT - 1))
        else:
            plane_index = -1
            plane_k = 0
        delta_plane = plane_k - plane_k_prev
        plane_sign = int(np.sign(delta_plane))
        flip = int(plane_sign != 0 and plane_sign_prev != 0 and plane_sign != plane_sign_prev)
        plane_flip_flags.append(flip)
        plane_sign_flips_w = int(sum(plane_flip_flags))
        plane_would_veto = int(plane_sign_flips_w > 1)

        active_trit = 1.0 if plane_index >= 0 else 0.0
        active_trit_count += active_trit
        plane_hits = []
        plane_rates = []
        for k in range(PLANE_COUNT):
            hit = 1.0 if plane_index == k else 0.0
            plane_counts[k] += hit
            plane_hits.append(hit)
            plane_rates.append(plane_counts[k] / max(t, 1))
        stress = active_trit_count / max(t, 1)

        shadow_delta_mdl = np.nan
        shadow_would_promote = 0
        shadow_is_tie = 0
        shadow_reject = 0
        w = SHADOW_REFIT_WINDOW
        if t - w >= 1 and t + w - 1 <= total_steps:
            start = t - w - 1
            mid = t - 1
            end = t + w - 1
            left = returns[start:mid]
            right = returns[mid:end]
            both = returns[start:end]
            mdl_current = shadow_mdl_for_window(both)
            if not np.isnan(mdl_current):
                mdl_left = shadow_mdl_for_window(left)
                mdl_right = shadow_mdl_for_window(right)
                split_penalty = SHADOW_SPLIT_PENALTY_MULT * np.log(max(len(both), 1.0))
                mdl_split = mdl_left + mdl_right + split_penalty
                shadow_delta_mdl = mdl_split - mdl_current
                eps = SHADOW_MDL_EPS_MULT * max(1.0, abs(mdl_current))
                shadow_would_promote = int(shadow_delta_mdl < -eps)
                shadow_is_tie = int(abs(shadow_delta_mdl) <= eps)
                shadow_reject = int(shadow_delta_mdl > eps)

        p_bad_t = float(p_bad[t]) if t < len(p_bad) else 0.0
        stress_veto = bool(bad_flag[t]) if t < len(bad_flag) else False
        permission = ternary_permission(p_bad_t, caution=PBAD_CAUTION, ban=PBAD_BAN)
        belief_alpha_plus = 0
        belief_alpha_minus = 0
        if plane_sign != 0:
            belief_alpha_plus = 1 if plane_sign == 1 else -1
            belief_alpha_minus = 1 if plane_sign == -1 else -1
        if plane_would_veto == 1 or plane_sign_flips_w > 1:
            belief_beta = -1
        elif plane_sign_flips_w <= 1:
            belief_beta = 1
        else:
            belief_beta = 0
        if p_bad_t >= thesis_pbad_hi or stress >= thesis_stress_hi:
            belief_rho = -1
        elif p_bad_t <= thesis_pbad_lo and stress <= thesis_stress_lo:
            belief_rho = 1
        else:
            belief_rho = 0
        belief_gate = int(permission == 1 and belief_beta != -1 and belief_rho != -1)
        belief_delta_plus = belief_alpha_plus * belief_gate
        belief_delta_minus = belief_alpha_minus * belief_gate
        belief_plus = update_belief(
            belief_plus, belief_delta_plus, belief_alpha_plus, belief_beta, belief_rho, permission
        )
        belief_minus = update_belief(
            belief_minus, belief_delta_minus, belief_alpha_minus, belief_beta, belief_rho, permission
        )
        belief_state = belief_state_label(belief_plus, belief_minus)
        if fill_count > 0 or pos != 0:
            edge_t = ternary_sign(edge_ema)
        else:
            edge_t = ternary_sign(z, tau=dz)
        action_signal = ternary_controller(
            direction=direction,
            edge=edge_t,
            permission=permission,
            capital_pressure=capital_pressure,
            thesis=thesis,
        )
        thesis_hold = False
        thesis_depth_prev = thesis_depth
        hard_veto = permission == -1 or stress_veto
        thesis_event = "thesis_none"
        thesis_reason = ""
        thesis_override = ""
        thesis_alpha = 0
        thesis_beta = 0
        thesis_rho = 0
        thesis_ds = 0
        thesis_sum = 0
        exit_trigger = False
        if thesis_memory:
            state = ThesisState(d=thesis_d, s=thesis_s, a=thesis_a, c=thesis_c, v=thesis_v)
            inputs = ThesisInputs(
                plane_sign=plane_sign,
                plane_sign_flips_w=plane_sign_flips_w,
                plane_would_veto=plane_would_veto,
                stress=stress,
                p_bad=p_bad_t,
                shadow_would_promote=shadow_would_promote,
            )
            params = ThesisParams(
                a_max=thesis_a_max,
                cooldown=thesis_cooldown,
                p_bad_lo=thesis_pbad_lo,
                p_bad_hi=thesis_pbad_hi,
                stress_lo=thesis_stress_lo,
                stress_hi=thesis_stress_hi,
            )
            new_state, derived, event = step_thesis_memory(state, inputs, params)
            action_proposed = 0 if hard_veto else action_signal
            if action_proposed == 0 and pos != 0 and not hard_veto:
                action_proposed = int(np.sign(pos))
            action_t, thesis_override = apply_thesis_constraints(
                state.d, derived, action_proposed, hard_veto, event.exit_trigger
            )
            thesis_event = event.event
            thesis_reason = event.reason
            thesis_alpha = derived.alpha
            thesis_beta = derived.beta
            thesis_rho = derived.rho
            thesis_ds = derived.ds
            thesis_sum = derived.sum
            thesis_d = new_state.d
            thesis_s = new_state.s
            thesis_a = new_state.a
            thesis_c = new_state.c
            thesis_v = new_state.v
            thesis_depth = 0
        else:
            if hard_veto:
                action_t = 0
                thesis_depth = 0
            else:
                if action_signal != 0 and action_signal != prev_action:
                    thesis_depth = 1
                elif action_signal != 0 and action_signal == prev_action and permission == 1:
                    thesis_depth = min(thesis_depth_prev + 1, thesis_depth_max)
                elif action_signal == 0 and thesis_depth_prev > 0:
                    thesis_depth = thesis_depth_prev - 1
                else:
                    thesis_depth = thesis_depth_prev

                if action_signal == 0 and thesis_depth_prev > 1:
                    action_t = prev_action
                    thesis_hold = True
                elif action_signal == 0:
                    action_t = 0
                else:
                    action_t = action_signal
        if hard_veto or permission == -1 or stress_veto:
            decision_kind = "flat"
        elif exit_trigger:
            decision_kind = "flat"
        elif permission == 0:
            decision_kind = "unknown"
        elif direction == 0 or edge_t == 0 or direction != edge_t:
            decision_kind = "unknown"
        elif action_t > 0:
            decision_kind = "long"
        elif action_t < 0:
            decision_kind = "short"
        else:
            decision_kind = "unknown"
        equity = cash + pos * price[t]
        cap, risk_budget = compute_cap(
            volume_t=volume[t],
            ret=ret,
            z_vel=z_vel,
            sigma_target=sigma_target,
            sigma=sigma,
            veto_sigma=VETO_SIGMA,
            cash=cash,
            pos=pos,
            price_t=price[t],
            risk_frac=risk_frac,
            contract_mult=contract_mult,
            participation_cap=PARTICIPATION_CAP,
            cap_hard_max=CAP_HARD_MAX,
            edge_gate=edge_gate,
            edge_t=edge_t,
            plane_index=plane_index,
            edge_decay=edge_decay,
            risk_headroom_low=RISK_HEADROOM_LOW,
            risk_headroom_high=RISK_HEADROOM_HIGH,
        )

        if action_t == prev_action:
            action_run_length += 1
            time_since_last_switch += 1
        else:
            action_run_length = 1
            time_since_last_switch = 0

        # Update persistence clocks
        state_age = state_age + 1 if direction == prev_state else 0 if direction == 0 else 1
        if pos != 0:
            thesis_age += 1
        else:
            thesis_age = 0
        if pos != 0 and direction != 0 and np.sign(pos) == np.sign(direction):
            align_age += 1
        else:
            align_age = 0

        # Triadic control: ternary action drives fills
        fill, cap = compute_fill(
            action_t=action_t,
            pos=pos,
            cap=cap,
            hold_decay=HOLD_DECAY,
            z_vel=z_vel,
            vel_exit=VEL_EXIT,
            thesis_hold=thesis_hold,
            persist_ramp=PERSIST_RAMP,
            align_age=align_age,
        )

        price_exec, cash, pos, fee, slippage_cost, capital_at_risk = apply_execution(
            price_t=price[t],
            fill=fill,
            cap=cap,
            cash=cash,
            pos=pos,
            cost=cost,
            impact_coeff=IMPACT_COEFF,
        )
        pos_prev = pos - fill
        fees_accrued += fee
        step_metrics, edge_ema = compute_step_accounting(
            price_t=price[t],
            cash=cash,
            pos=pos,
            fee=fee,
            slippage_cost=slippage_cost,
            capital_at_risk=capital_at_risk,
            prev_pnl=prev_pnl,
            pos_prev=pos_prev,
            edge_ema=edge_ema,
            edge_ema_alpha=edge_ema_alpha,
            fees_accrued=fees_accrued,
            est_tax_rate=est_tax_rate,
            start_cash=START_CASH,
            t=t,
        )
        pnl = step_metrics["pnl"]
        edge_raw = step_metrics["edge_raw"]
        cash_eff = step_metrics["cash_eff"]
        exec_eff = step_metrics["exec_eff"]
        cash_vel = step_metrics["cash_vel"]
        c_spend = step_metrics["c_spend"]
        realized_pnl = step_metrics["realized_pnl"]

        realized_pnl_step = 0.0
        trade_pnl = 0.0
        trade_pnl_pct = 0.0
        trade_duration = 0
        trade_closed = False
        trade_close_reason = ""
        close_trade_id = None
        close_entry_step = None
        close_entry_price = None
        close_entry_notional = 0.0
        entry_price = trade_entry_price if trade_entry_step is not None else np.nan
        price_move_entry = price[t] - entry_price if trade_entry_step is not None else np.nan
        if fill != 0:
            if pos_prev == 0:
                trade_id += 1
                trade_entry_step = t
                trade_entry_price = price_exec
                trade_entry_notional = abs(fill * price_exec)
                trade_realized_pnl = 0.0
                avg_entry_price = price_exec
                thesis_depth_peak = thesis_depth
            elif pos == 0:
                closed_qty = abs(pos_prev)
                if pos_prev > 0:
                    realized_pnl_step = (price_exec - avg_entry_price) * closed_qty
                else:
                    realized_pnl_step = (avg_entry_price - price_exec) * closed_qty
                trade_realized_pnl += realized_pnl_step
                trade_pnl = trade_realized_pnl
                trade_duration = t - (trade_entry_step or t)
                if trade_entry_notional > 0:
                    trade_pnl_pct = trade_pnl / trade_entry_notional
                trade_closed = True
                trade_close_reason = "flat"
                close_trade_id = trade_id
                close_entry_step = trade_entry_step
                close_entry_price = trade_entry_price
                close_entry_notional = trade_entry_notional
                trade_entry_step = None
                trade_entry_price = 0.0
                trade_entry_notional = 0.0
                trade_realized_pnl = 0.0
                avg_entry_price = 0.0
                thesis_depth_peak = 0
            elif np.sign(pos_prev) == np.sign(pos):
                if np.sign(fill) == np.sign(pos_prev):
                    total_qty = abs(pos_prev) + abs(fill)
                    if total_qty > 0:
                        avg_entry_price = (
                            avg_entry_price * abs(pos_prev) + price_exec * abs(fill)
                        ) / total_qty
                else:
                    closed_qty = min(abs(fill), abs(pos_prev))
                    if pos_prev > 0:
                        realized_pnl_step = (price_exec - avg_entry_price) * closed_qty
                    else:
                        realized_pnl_step = (avg_entry_price - price_exec) * closed_qty
                    trade_realized_pnl += realized_pnl_step
            else:
                closed_qty = abs(pos_prev)
                if pos_prev > 0:
                    realized_pnl_step = (price_exec - avg_entry_price) * closed_qty
                else:
                    realized_pnl_step = (avg_entry_price - price_exec) * closed_qty
                trade_realized_pnl += realized_pnl_step
                trade_pnl = trade_realized_pnl
                trade_duration = t - (trade_entry_step or t)
                if trade_entry_notional > 0:
                    trade_pnl_pct = trade_pnl / trade_entry_notional
                trade_closed = True
                trade_close_reason = "flip"
                close_trade_id = trade_id
                close_entry_step = trade_entry_step
                close_entry_price = trade_entry_price
                close_entry_notional = trade_entry_notional
                trade_id += 1
                trade_entry_step = t
                trade_entry_price = price_exec
                trade_entry_notional = abs(pos * price_exec)
                trade_realized_pnl = 0.0
                avg_entry_price = price_exec
                thesis_depth_peak = thesis_depth

        realized_pnl_total += realized_pnl_step

        if trade_entry_step is not None:
            thesis_depth_peak = max(thesis_depth_peak, thesis_depth)

        # Proxy MDL and stress (plane-aware surprise)
        if action_t != prev_action:
            side_cost += mdl_switch_penalty
        if fill != 0:
            side_cost += mdl_trade_penalty
        mdl_rate = (side_cost + active_trit_count) / max(t, 1)

        # Goal probability + shortfall via normal approximation
        pnl_delta_window.append(c_spend - c_spend_prev)
        if len(pnl_delta_window) > window:
            pnl_delta_window.pop(0)
        mu = float(np.mean(pnl_delta_window)) if pnl_delta_window else 0.0
        sigma_spend = float(np.std(pnl_delta_window)) if len(pnl_delta_window) > 1 else 0.0
        if time_index is not None and t0_ts is not None:
            try:
                t_ts = time_index[t]
                remaining = max((pd.to_datetime(t_ts) - pd.to_datetime(t0_ts)).total_seconds(), 1.0)
                total = max((pd.to_datetime(time_index[-1]) - pd.to_datetime(t0_ts)).total_seconds(), 1.0)
                remaining_steps = max(int((total - remaining) / max(total / total_steps, 1.0)), 0)
            except Exception:
                remaining_steps = max(total_steps - t, 0)
        else:
            remaining_steps = max(total_steps - t, 0)
        mean_ct = c_spend + mu * remaining_steps
        std_ct = sigma_spend * np.sqrt(remaining_steps)
        if std_ct <= 0:
            goal_prob = 1.0 if mean_ct >= goal_cash_x else 0.0
            es_shortfall = max(0.0, goal_cash_x - mean_ct)
        else:
            z = (goal_cash_x - mean_ct) / std_ct
            goal_prob = 1.0 - norm_cdf(z)
            z_eps = norm_ppf(goal_eps)
            mean_tail = mean_ct - std_ct * (norm_pdf(z_eps) / max(goal_eps, 1e-9))
            es_shortfall = max(0.0, goal_cash_x - mean_tail)
        goal_align = goal_prob - prev_goal_prob
        goal_pressure = np.clip(
            ((goal_cash_x - c_spend) / max(goal_cash_x, 1e-9)) * (total_steps / max(remaining_steps, 1)),
            0.0,
            1.0,
        )
        regret = (START_CASH - fees_accrued) - mean_ct
        log_ret = np.log(price[t] / price[t - 1]) if price[t - 1] > 0 else 0.0
        tc_step = tc_k * abs(action_t - prev_action)
        reward_regret = compute_regret_reward(prev_action, log_ret, benchmark_x, tc_step)
        risk_penalty = 0.0
        sigma_ref = (
            (sigma_target or 0.0) * price[t] * max(abs(pos), 1.0)
            if sigma_target
            else max(price[t] * abs(pos), 1.0)
        )
        k_latent = (c_spend - c_spend_prev) / (sigma_ref + 1e-9)
        capital_pressure = ternary_sign(k_latent, tau=K_LATENT_TAU)
        row = {
            "t": t,
            "ts": time_index[t] if time_index is not None and t < len(time_index) else np.nan,
            "price": price[t],
            "price_exec": price_exec,
            "price_change": price_change,
            "price_ret": ret,
            "volume": volume[t] if t < len(volume) else np.nan,
            "pnl": pnl,
            "cash": cash,
            "cash_eff": cash_eff,
            "exec_eff": exec_eff,
            "cash_vel": cash_vel,
            "edge_raw": edge_raw,
            "edge_ema": edge_ema,
            "c_spend": c_spend,
            "goal_prob": goal_prob,
            "goal_align": goal_align,
            "goal_pressure": goal_pressure,
            "regret": regret,
            "r_step": log_ret,
            "tc_step": tc_step,
            "benchmark_x": benchmark_x,
            "reward_regret": reward_regret,
            "risk_penalty": risk_penalty,
            "es_shortfall": es_shortfall,
            "mdl_rate": mdl_rate,
            "stress": stress,
            "active_trit": active_trit,
            "plane_index": plane_index,
            "plane_k": plane_k,
            "delta_plane": delta_plane,
            "plane_abs": abs(delta_plane),
            "plane_sign": plane_sign,
            "plane_sign_flips_W": plane_sign_flips_w,
            "plane_would_veto": plane_would_veto,
            "sigma_slow": sigma_slow,
            "plane0": plane_hits[0] if PLANE_COUNT > 0 else 0.0,
            "plane1": plane_hits[1] if PLANE_COUNT > 1 else 0.0,
            "plane2": plane_hits[2] if PLANE_COUNT > 2 else 0.0,
            "plane3": plane_hits[3] if PLANE_COUNT > 3 else 0.0,
            "plane_rate0": plane_rates[0] if PLANE_COUNT > 0 else 0.0,
            "plane_rate1": plane_rates[1] if PLANE_COUNT > 1 else 0.0,
            "plane_rate2": plane_rates[2] if PLANE_COUNT > 2 else 0.0,
            "plane_rate3": plane_rates[3] if PLANE_COUNT > 3 else 0.0,
            "p_bad": p_bad_t,
            "bad_flag": int(bad_flag[t]) if t < len(bad_flag) else 0,
            "ban": int(permission == -1),
            "can_trade": int(permission == 1),
            "direction": direction,
            "belief_state": belief_state,
            "belief_plus": belief_plus,
            "belief_minus": belief_minus,
            "belief_alpha_plus": belief_alpha_plus,
            "belief_alpha_minus": belief_alpha_minus,
            "belief_delta_plus": belief_delta_plus,
            "belief_delta_minus": belief_delta_minus,
            "decision_kind": decision_kind,
            "edge_t": edge_t,
            "permission": permission,
            "capital_pressure": capital_pressure,
            "risk_budget": risk_budget,
            "z_norm": abs(z),
            "z_vel": z_vel,
            "hold": int(action_t == 0),
            "entropy": 0.0,
            "regime": 0,
            "shadow_delta_mdl": shadow_delta_mdl,
            "shadow_would_promote": shadow_would_promote,
            "shadow_is_tie": shadow_is_tie,
            "shadow_reject": shadow_reject,
            "action": np.sign(fill) if fill != 0 else 0,
            "action_signal": action_signal,
            "action_t": action_t,
            "action_run_length": action_run_length,
            "time_since_last_switch": time_since_last_switch,
            "source": source,
            "tape_id": tape_id if tape_id is not None else "",
            "pos": pos,
            "fill": fill,
            "fill_units": fill,
            "fill_value": fill * price_exec,
            "cap": cap,
            "equity": equity,
            "avg_entry_price": avg_entry_price if avg_entry_price else np.nan,
            "entry_price": entry_price,
            "entry_step": trade_entry_step if trade_entry_step is not None else np.nan,
            "trade_id": trade_id if trade_entry_step is not None else np.nan,
            "trade_open": int(trade_entry_step is not None),
            "trade_closed": int(trade_closed),
            "trade_duration": trade_duration,
            "trade_pnl": trade_pnl,
            "trade_pnl_pct": trade_pnl_pct,
            "realized_pnl_step": realized_pnl_step,
            "realized_pnl_total": realized_pnl_total,
            "unrealized_pnl": (price[t] - avg_entry_price) * pos if pos != 0 else 0.0,
            "price_move_entry": price_move_entry,
            "prev_action": prev_action,
            "thesis_age": thesis_age,
            "thesis_depth": thesis_depth,
            "thesis_hold": int(thesis_hold),
            "thesis_d": thesis_d,
            "thesis_s": thesis_s,
            "thesis_a": thesis_a,
            "thesis_c": thesis_c,
            "thesis_v": thesis_v,
            "thesis_alpha": thesis_alpha,
            "thesis_beta": thesis_beta,
            "thesis_rho": thesis_rho,
            "thesis_ds": thesis_ds,
            "thesis_sum": thesis_sum,
            "thesis_event": thesis_event,
            "thesis_reason": thesis_reason,
            "thesis_override": thesis_override,
            "state_age": state_age,
            "align_age": align_age,
        }
        rows.append(row)
        emit_step_row(row, log_path)
        if trade_closed and trade_log_path:
            trade_row = {
                "t": t,
                "ts": time_index[t] if time_index is not None and t < len(time_index) else np.nan,
                "trade_id": close_trade_id,
                "source": source,
                "close_reason": trade_close_reason,
                "entry_step": close_entry_step if close_entry_step is not None else np.nan,
                "exit_step": t,
                "entry_price": close_entry_price,
                "exit_price": price_exec,
                "trade_duration": trade_duration,
                "trade_pnl": trade_pnl,
                "trade_pnl_pct": trade_pnl_pct,
                "price_move": price_exec - (close_entry_price or 0.0),
                "price_move_pct": (
                    (price_exec / close_entry_price - 1.0) if close_entry_price else np.nan
                ),
                "thesis_depth_exit": thesis_depth,
                "thesis_depth_prev": thesis_depth_prev,
                "thesis_depth_peak": thesis_depth_peak,
            }
            emit_trade_row(trade_row, trade_log_path)
        if log_level in {"trades", "verbose"} and fill != 0:
            trade_throttle = progress_every if progress_every else 1
            if t % trade_throttle != 0 and t != total_steps:
                pass
            else:
                unrealized = (price[t] - avg_entry_price) * pos if pos != 0 else 0.0
                emit_trade_print(
                    fmt_ts(),
                    t=t,
                    price=price[t],
                    fill=fill,
                    pos=pos,
                    cap=cap,
                    action=int(row["action"]),
                    permission=permission,
                    cash_eff=row["cash_eff"],
                    exec_eff=row["exec_eff"],
                    c_spend=row["c_spend"],
                    goal_prob=row["goal_prob"],
                    mdl_rate=row["mdl_rate"],
                    stress=row["stress"],
                    plane_index=row["plane_index"],
                    can_trade=row["can_trade"],
                    regret=row["regret"],
                    unrealized=unrealized,
                    realized_pnl_step=realized_pnl_step,
                    avg_entry_price=avg_entry_price,
                )
        if log_level in {"trades", "verbose"} and trade_closed:
            emit_trade_close_print(
                fmt_ts(),
                trade_id=close_trade_id,
                close_reason=trade_close_reason,
                trade_pnl=trade_pnl,
                trade_pnl_pct=trade_pnl_pct,
                trade_duration=trade_duration,
                entry_price=close_entry_price,
                exit_price=price_exec,
            )
        if log_level in {"info", "verbose"} and progress_every and (t % progress_every == 0 or t == total_steps):
            emit_progress_print(
                fmt_ts(),
                source=source,
                t=t,
                total_steps=total_steps,
                pnl=row["pnl"],
                pos=row["pos"],
                fill=row["fill"],
                action=int(row["action"]),
                p_bad=row["p_bad"],
                bad_flag=row["bad_flag"],
                cash_eff=row["cash_eff"],
                exec_eff=row["exec_eff"],
                mdl_rate=row["mdl_rate"],
                stress=row["stress"],
                goal_prob=row["goal_prob"],
            )
        z_prev = z
        prev_action = action_t
        prev_state = direction
        prev_cash = cash
        prev_pnl = pnl
        prev_goal_prob = goal_prob
        c_spend_prev = c_spend
        plane_k_prev = plane_k
        plane_sign_prev = plane_sign
        if fill != 0:
            fill_count += 1
        if trade_closed:
            closed_trade_count += 1
            if max_trades is not None and closed_trade_count >= max_trades:
                stop_reason = "max_trades"
                break
        time.sleep(sleep_s)  # slow enough to watch in dashboard

    elapsed_s = time.time() - start_ts
    reason = f", stop={stop_reason}" if stop_reason else ""
    summary = build_run_summary(
        rows=rows,
        p_bad=p_bad,
        bad_flag=bad_flag,
        start_cash=START_CASH,
        stop_reason=stop_reason,
        elapsed_s=elapsed_s,
    )
    print(
        f"[{fmt_ts()}] Run complete: source={source}, steps={summary['steps']}, "
        f"trades={summary['trades']}, pnl={summary['pnl']:.4f}, "
        f"elapsed={elapsed_s:.2f}s{reason}"
    )
    emit_run_summary(summary, RUN_HISTORY)
    return summary, rows

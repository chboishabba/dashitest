"""
run_trader.py
--------------
Streams a simple ternary trading loop over cached Stooq data and logs
to logs/trading_log.csv for the training_dashboard to visualize.

Usage:
  python trading/run_trader.py
Then in another terminal:
  python trading/training_dashboard.py --log logs/trading_log.csv --refresh 0.5
"""

import time
import pathlib
import math
import os
import sys
import subprocess
from dataclasses import dataclass
from collections import deque
import pandas as pd
import numpy as np

from ternary import ternary_controller, ternary_permission, ternary_sign

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


def compute_triadic_state(prices, dz_min=5e-5, window=200):
    """
    Reproduce the triadic state logic from run_trader:
    - rolling std over returns (window)
    - EWMA of volatility-normalized returns
    - dead-zone around zero
    Returns an array of ints in {-1,0,+1} with the same length as prices.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    state = np.zeros(n, dtype=int)
    z_prev = 0.0
    recent_rets = []
    shadow_mdl_window = []
    for t in range(1, n):
        ret = prices[t] / prices[t - 1] - 1.0
        recent_rets.append(ret)
        if len(recent_rets) > window:
            recent_rets.pop(0)
        sigma = np.std(recent_rets) if recent_rets else dz_min
        z_update = ret / (sigma + 1e-9)
        z_update = np.clip(z_update, -5.0, 5.0)
        z = 0.95 * z_prev + 0.05 * z_update
        dz = max(dz_min, 0.5 * sigma)
        if abs(z) < dz:
            desired = 0
        elif z > 0:
            desired = 1
        else:
            desired = -1
        state[t] = desired
        z_prev = z
    return state


def compute_structural_stress(prices, states, window=100, vol_z_thr=1.5, flip_thr=0.4):
    """
    Derive a crude structural stress score from price/triadic state history:
    - rolling vol z-score (median/MAD)
    - rolling jump z-score (abs return / rolling vol)
    - rolling flip rate of the triadic state

    Returns p_bad in [0,1] and a bad_flag bool (p_bad > 0.7).
    """
    prices = pd.Series(prices, dtype=float)
    states = pd.Series(states, dtype=float)
    rets = prices.pct_change().fillna(0.0)
    vol = rets.rolling(window).std().bfill().fillna(0.0)
    med_vol = vol.median()
    mad_vol = (vol - med_vol).abs().median() + 1e-9
    vol_z = (vol - med_vol) / mad_vol
    jump_z = (rets.abs() / (vol + 1e-9)).fillna(0.0)
    flips = (states != states.shift(1)).astype(float)
    flip_rate = flips.rolling(window).mean().fillna(0.0)
    score = (
        (vol_z / vol_z_thr).clip(lower=0)
        + (jump_z).clip(lower=0)
        + (flip_rate / flip_thr).clip(lower=0)
    )
    # squash to [0,1] with a smooth cap
    p_bad = (score / (1.0 + score)).clip(0.0, 1.0)
    bad_flag = p_bad > 0.7
    return p_bad.to_numpy(), bad_flag.to_numpy()


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)


def norm_ppf(p: float) -> float:
    # Acklam's approximation for inverse normal CDF
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def clip_ternary_sum(val: int | float) -> int:
    if val >= 1:
        return 1
    if val <= -1:
        return -1
    return 0


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


def fmt_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def find_btc_csv():
    raw = pathlib.Path("data/raw/stooq")
    if not raw.exists():
        return None
    # prefer intraday then daily
    files = (
        sorted(raw.glob("btc_intraday_1s*.csv"))
        + sorted(raw.glob("btc_intraday*.csv"))
        + sorted(raw.glob("btc*.csv"))
    )
    for f in files:
        try:
            df_head = pd.read_csv(f, nrows=5)
            cols = [c.lower() for c in df_head.columns]
            if any(k in cols for k in ("close", "zamkniecie")) and not df_head.empty:
                df_full = pd.read_csv(f)
                if len(df_full) >= 1000:
                    return f
        except Exception:
            continue
    return None


def find_stooq_csv():
    raw = pathlib.Path("data/raw/stooq")
    if not raw.exists():
        raise FileNotFoundError("data/raw/stooq not found; run trading/data_downloader.py first.")
    files = sorted(raw.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No Stooq CSVs found; run trading/data_downloader.py.")
    # pick first valid CSV with Close column and data
    for f in files:
        try:
            df_head = pd.read_csv(f, nrows=5)
            cols = [c.lower() for c in df_head.columns]
            if any(k in cols for k in ("close", "zamkniecie")) and not df_head.empty:
                return f
        except Exception:
            continue
    raise FileNotFoundError(
        "No valid Stooq CSV with Close/Zamkniecie column; re-run trading/data_downloader.py with .us symbols."
    )


def list_price_csvs(raw_root: pathlib.Path) -> list[pathlib.Path]:
    if not raw_root.exists():
        return []
    return sorted(p for p in raw_root.rglob("*.csv") if p.is_file())


def load_prices(path: pathlib.Path, return_time: bool = False):
    def read_basic(p):
        return pd.read_csv(p)

    def read_skip(p):
        # for yfinance multi-index CSV: skip ticker header rows
        return pd.read_csv(p, skiprows=2)

    def parse_df(df):
        col_map = {c.lower(): c for c in df.columns}
        close_key = None
        for key in ("close", "zamkniecie"):
            if key in col_map:
                close_key = col_map[key]
                break
        if close_key is None:
            raise ValueError(f"Close/Zamkniecie column not found in {path}")

        date_key = None
        for dk in ("date", "data", "datetime"):
            if dk in col_map:
                date_key = col_map[dk]
                break
        # yfinance exports sometimes put dates under "Price" after skipping header rows
        if date_key is None and "price" in col_map:
            maybe_dates = pd.to_datetime(df[col_map["price"]], errors="coerce")
            if maybe_dates.notna().mean() > 0.8:
                date_key = col_map["price"]

        close_series = pd.to_numeric(df[close_key], errors="coerce")

        vol_key = None
        for vk in ("volume", "wolumen"):
            if vk in col_map:
                vol_key = col_map[vk]
                break
        if vol_key is not None:
            vol_series = pd.to_numeric(df[vol_key], errors="coerce")
        else:
            vol_series = pd.Series(np.ones(len(close_series)) * 1e6)

        if date_key is not None:
            dates = pd.to_datetime(df[date_key], errors="coerce")
            valid = (~close_series.isna()) & (~dates.isna())
            close_series = close_series[valid]
            vol_series = vol_series[valid]
            dates = dates[valid]
            order = np.argsort(dates.to_numpy())
            close_series = close_series.iloc[order]
            vol_series = vol_series.iloc[order]
        else:
            close_series = close_series.dropna()
            vol_series = vol_series.loc[close_series.index]

        time_index = dates.to_numpy() if date_key is not None else None
        return close_series.to_numpy(), vol_series.to_numpy(), time_index

    # try basic read, then fallback to skiprows if too many non-numeric
    for reader in (read_basic, read_skip):
        try:
            df = reader(path)
            # handle yfinance multi-index export (ticker row, then date row)
            cols_lower = [c.lower() for c in df.columns]
            if "price" in cols_lower:
                price_col = cols_lower.index("price")
                head_vals = df.iloc[:2, price_col].astype(str).str.lower()
                if head_vals.str.contains("ticker").any():
                    df = df.iloc[2:].reset_index(drop=True)
            close, vol, ts = parse_df(df)
            # replace nonpositive/NaN volume
            pos_vol = vol[np.isfinite(vol) & (vol > 0)]
            fallback_vol = np.median(pos_vol) if pos_vol.size else 1e6
            vol = np.where((~np.isfinite(vol)) | (vol <= 0), fallback_vol, vol)
            # require reasonable length
            if len(close) >= 10 and np.isfinite(close).any():
                if return_time:
                    return close, vol, ts
                return close, vol
        except Exception:
            continue
    raise ValueError(f"Could not parse prices from {path}")


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
    trade_count = 0
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
        if max_trades is not None and trade_count >= max_trades:
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
        belief_delta_plus = clip_ternary_sum(belief_alpha_plus + belief_beta + belief_rho + permission)
        belief_delta_minus = clip_ternary_sum(belief_alpha_minus + belief_beta + belief_rho + permission)
        belief_plus = update_belief(
            belief_plus, belief_delta_plus, belief_alpha_plus, belief_beta, belief_rho, permission
        )
        belief_minus = update_belief(
            belief_minus, belief_delta_minus, belief_alpha_minus, belief_beta, belief_rho, permission
        )
        belief_state = belief_state_label(belief_plus, belief_minus)
        if trade_count > 0 or pos != 0:
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
        base_cap = PARTICIPATION_CAP * volume[t]
        vel_term = 1.0 + 10.0 * max(abs(ret), z_vel)
        cap = base_cap * vel_term
        # volatility targeting (risk parity style)
        if sigma_target and sigma_target > 0:
            cap *= sigma_target / max(sigma, 1e-9)
            if sigma > VETO_SIGMA * sigma_target:
                cap *= 0.2  # shrink size aggressively in chaotic regimes
        # optional risk budget (futures-style)
        equity = cash + pos * price[t]
        if risk_frac:
            risk_cap = (equity * risk_frac) / (price[t] * contract_mult + 1e-9)
            cap = min(cap, risk_cap)
        cap = max(0.0, min(cap, CAP_HARD_MAX))
        if edge_gate and edge_t == -1 and plane_index <= 0:
            cap *= edge_decay
        risk_headroom = cap / max(CAP_HARD_MAX, 1e-9)
        if risk_headroom > RISK_HEADROOM_HIGH:
            risk_budget = 1
            cap *= 1.0
        elif risk_headroom < RISK_HEADROOM_LOW:
            risk_budget = -1
            cap *= 0.2
        else:
            risk_budget = 0
            cap *= 0.5
        cap = max(0.0, min(cap, CAP_HARD_MAX))

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
        if action_t == 0:
            # decay exposure toward zero
            cap = cap * (1.0 - HOLD_DECAY) + 1e-9  # keep a tiny ability to act
            fill = np.clip(-pos, -cap, cap)
        else:
            # exit if latent velocity too high while in position
            if z_vel > VEL_EXIT and pos != 0:
                fill = np.clip(-pos, -cap, cap)
            else:
                if thesis_hold:
                    step = 0.0
                else:
                    # ramp toward target with persistence factor; faster if align_age large
                    target = action_t * cap
                    ramp = PERSIST_RAMP * (1.0 + align_age * 0.01)
                    step = ramp * (target - pos)
                fill = np.clip(step, -cap, cap)

        slippage = IMPACT_COEFF * abs(fill / max(cap, 1e-9))
        price_exec = price[t] * (1 + slippage * np.sign(fill))
        cash -= fill * price_exec
        pos_prev = pos
        pos += fill
        fee = cost * abs(fill)
        fees_accrued += fee
        slippage_cost = abs(fill) * abs(price_exec - price[t])
        pnl = cash + pos * price[t] - fee
        capital_at_risk = abs(fill * price_exec)
        delta_pnl = pnl - prev_pnl
        edge_raw = delta_pnl / (abs(pos_prev) + 1e-9)
        edge_ema = (1.0 - edge_ema_alpha) * edge_ema + edge_ema_alpha * edge_raw
        cash_eff = delta_pnl / (capital_at_risk + 1e-9) if capital_at_risk > 0 else np.nan
        exec_eff = 1.0 - (slippage_cost + fee) / (capital_at_risk + 1e-9) if capital_at_risk > 0 else np.nan
        cash_vel = (cash - START_CASH) / max(t, 1)
        realized_pnl = cash - START_CASH
        tax_est = est_tax_rate * max(realized_pnl, 0.0)
        c_spend = cash - tax_est - fees_accrued

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
        if log_path:
            pd.DataFrame([row]).to_csv(
                log_path, mode="a", header=not log_path.exists(), index=False
            )
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
            pd.DataFrame([trade_row]).to_csv(
                trade_log_path, mode="a", header=not trade_log_path.exists(), index=False
            )
        if log_level in {"trades", "verbose"} and fill != 0:
            trade_throttle = progress_every if progress_every else 1
            if t % trade_throttle != 0 and t != total_steps:
                pass
            else:
                unrealized = (price[t] - avg_entry_price) * pos if pos != 0 else 0.0
                print(
                    f"[{fmt_ts()}] [trade] t={t:6d} px={price[t]:.2f} fill={fill:.4f} pos={pos:.4f} "
                    f"cap={cap:.4f} act={int(row['action'])} banned={int(permission == -1)} "
                    f"cash_eff={row['cash_eff']:.5f} exec_eff={row['exec_eff']:.5f} "
                    f"c_spend={row['c_spend']:.2f} goal_p={row['goal_prob']:.3f} "
                    f"mdl={row['mdl_rate']:.4f} stress={row['stress']:.4f} "
                    f"plane={row['plane_index']} can_trade={row['can_trade']} "
                    f"regret={row['regret']:.2f} u_pnl={unrealized:.2f} "
                    f"r_pnl={realized_pnl_step:.2f} entry={avg_entry_price:.2f}"
                )
        if log_level in {"trades", "verbose"} and trade_closed:
            print(
                f"[{fmt_ts()}] [trade] close id={close_trade_id} reason={trade_close_reason} "
                f"pnl={trade_pnl:.4f} pct={trade_pnl_pct:.4f} dur={trade_duration} "
                f"entry={close_entry_price:.4f} exit={price_exec:.4f}"
            )
        if log_level in {"info", "verbose"} and progress_every and (t % progress_every == 0 or t == total_steps):
            print(
                f"[{fmt_ts()}] [{source}] t={t:6d}/{total_steps:6d} pnl={row['pnl']:.4f} pos={row['pos']:.4f} "
                f"fill={row['fill']:.4f} act={int(row['action'])} p_bad={row['p_bad']:.3f} "
                f"bad={row['bad_flag']} cash_eff={row['cash_eff']:.5f} exec_eff={row['exec_eff']:.5f} "
                f"mdl_rate={row['mdl_rate']:.4f} stress={row['stress']:.4f} goal_prob={row['goal_prob']:.3f}"
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
            trade_count += 1
            if max_trades is not None and trade_count >= max_trades:
                stop_reason = "max_trades"
                break
        time.sleep(sleep_s)  # slow enough to watch in dashboard

    # print summary stats
    total_pnl = rows[-1]["pnl"] if rows else 0.0
    trades = sum(1 for r in rows if r["fill"] != 0)
    hold_pct = sum(1 for r in rows if r["action"] == 0) / len(rows) if rows else 0.0
    max_drawdown = 0.0
    if rows:
        equity_curve = pd.Series([r["pnl"] for r in rows])
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max).min()
        max_drawdown = float(drawdown)
    elapsed_s = time.time() - start_ts
    reason = f", stop={stop_reason}" if stop_reason else ""
    print(
        f"[{fmt_ts()}] Run complete: source={source}, steps={len(rows)}, trades={trades}, "
        f"pnl={total_pnl:.4f}, elapsed={elapsed_s:.2f}s{reason}"
    )

    # Conditional returns to validate the bad-day classifier
    ret_all = ret_good = ret_bad = float("nan")
    if rows:
        equity_series = pd.Series([r["pnl"] for r in rows])
        eq_prev = equity_series.shift(1).fillna(START_CASH)
        eq_ret = equity_series.diff() / eq_prev.replace(0, np.nan)
        bad_mask = pd.Series([bool(r["bad_flag"]) for r in rows])
        ret_all = float(eq_ret.mean(skipna=True))
        ret_good = float(eq_ret[~bad_mask].mean(skipna=True))
        ret_bad = float(eq_ret[bad_mask].mean(skipna=True))

    summary = {
        "timestamp": pd.Timestamp.utcnow(),
        "source": source,
        "steps": len(rows),
        "trades": trades,
        "pnl": total_pnl,
        "elapsed_s": elapsed_s,
        "stop_reason": stop_reason,
        "hold_pct": hold_pct,
        "max_drawdown": max_drawdown,
        "p_bad_mean": float(np.mean(p_bad[: len(rows)])) if len(rows) else float("nan"),
        "bad_rate": float(np.mean(bad_flag[: len(rows)])) if len(rows) else float("nan"),
        "ret_all": ret_all,
        "ret_good": ret_good,
        "ret_bad": ret_bad,
    }
    pd.DataFrame([summary]).to_csv(
        RUN_HISTORY, mode="a", header=not RUN_HISTORY.exists(), index=False
    )
    return summary, rows


def main(
    max_steps=None,
    max_trades=None,
    max_seconds=None,
    sleep_s=0.0,
    risk_frac: float = DEFAULT_RISK_FRAC,
    contract_mult: float = CONTRACT_MULT,
    sigma_target: float = SIGMA_TARGET,
    est_tax_rate: float = EST_TAX_RATE,
    goal_cash_x: float = GOAL_CASH_X,
    goal_eps: float = GOAL_EPS,
    mdl_noise_mult: float = MDL_NOISE_MULT,
    mdl_switch_penalty: float = MDL_SWITCH_PENALTY,
    mdl_trade_penalty: float = MDL_TRADE_PENALTY,
    progress_every: int = 0,
    log_level: str = "info",
    run_all: bool = False,
    raw_root: str = "data/raw",
    log_prefix: str = "logs/trading_log",
    inter_run_sleep: float = 0.5,
    log_combined: bool = False,
    edge_gate: bool = False,
    edge_decay: float = 0.9,
    edge_alpha: float = EDGE_EMA_ALPHA,
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
    geometry_plots: bool = True,
    geometry_overlay: bool = True,
    geometry_simplex: bool = True,
    geometry_dir: str = "logs/geometry",
):
    run_ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def slugify(value: str) -> str:
        keep = []
        for ch in value:
            if ch.isalnum() or ch in ("-", "_"):
                keep.append(ch)
            else:
                keep.append("_")
        return "".join(keep)

    def emit_geometry(log_path: pathlib.Path, source_name: str):
        if not geometry_plots:
            return
        geometry_root = pathlib.Path(geometry_dir)
        geometry_root.mkdir(parents=True, exist_ok=True)
        prefix = geometry_root / f"{slugify(source_name)}_{run_ts}"
        cmd = [
            sys.executable,
            "scripts/plot_decision_geometry.py",
            "--csv",
            str(log_path),
            "--save-prefix",
            str(prefix),
            "--no-show",
        ]
        if geometry_overlay:
            cmd.append("--overlay")
        if geometry_simplex:
            cmd.append("--simplex")
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f".{os.pathsep}{existing}" if existing else "."
        try:
            subprocess.run(cmd, check=False, env=env)
        except Exception as exc:
            print(f"[warn] geometry plots failed for {log_path}: {exc}")

    source = "stooq"
    try:
        if run_all:
            raw_root_path = pathlib.Path(raw_root)
            csv_paths = list_price_csvs(raw_root_path)
            if not csv_paths:
                raise FileNotFoundError(f"No CSVs found under {raw_root_path}.")
            combined_path = None
            if log_combined:
                combined_path = pathlib.Path(f"{log_prefix}_all.csv")
                combined_path.parent.mkdir(parents=True, exist_ok=True)
                combined_path.unlink(missing_ok=True)
            for idx, csv_path in enumerate(csv_paths, 1):
                try:
                    price, volume, ts = load_prices(csv_path, return_time=True)
                except Exception as exc:
                    print(f"[skip] {csv_path} ({exc})")
                    continue
                source = f"{csv_path.parent.name}:{csv_path.stem}"
                if log_combined and combined_path is not None:
                    log_path = combined_path
                    trade_log_path = pathlib.Path(f"{log_prefix}_trades_all.csv")
                else:
                    log_path = pathlib.Path(f"{log_prefix}_{csv_path.stem}.csv")
                    trade_log_path = pathlib.Path(f"{log_prefix}_trades_{csv_path.stem}.csv")
                print(f"[run {idx}/{len(csv_paths)}] {csv_path} -> {log_path}")
                run_trading_loop(
                    price=price,
                    volume=volume,
                    source=source,
                    time_index=ts,
                    max_steps=max_steps,
                    max_trades=max_trades,
                    max_seconds=max_seconds,
                    sleep_s=sleep_s,
                    risk_frac=risk_frac,
                    contract_mult=contract_mult,
                    sigma_target=sigma_target,
                    est_tax_rate=est_tax_rate,
                    goal_cash_x=goal_cash_x,
                    goal_eps=goal_eps,
                    mdl_noise_mult=mdl_noise_mult,
                    mdl_switch_penalty=mdl_switch_penalty,
                    mdl_trade_penalty=mdl_trade_penalty,
                    log_path=log_path,
                    trade_log_path=trade_log_path,
                    progress_every=progress_every,
                    log_level=log_level,
                    log_append=bool(log_combined),
                    tape_id=source if log_combined else None,
                    edge_gate=edge_gate,
                    edge_decay=edge_decay,
                    edge_ema_alpha=edge_alpha,
                    thesis_depth_max=thesis_depth_max,
                    thesis_memory=thesis_memory,
                    thesis_a_max=thesis_a_max,
                    thesis_cooldown=thesis_cooldown,
                    thesis_pbad_lo=thesis_pbad_lo,
                    thesis_pbad_hi=thesis_pbad_hi,
                    thesis_stress_lo=thesis_stress_lo,
                    thesis_stress_hi=thesis_stress_hi,
                    tc_k=tc_k,
                    benchmark_x=benchmark_x,
                )
                if not log_combined:
                    emit_geometry(log_path, source)
                if inter_run_sleep > 0:
                    time.sleep(inter_run_sleep)
            if log_combined and combined_path is not None:
                emit_geometry(combined_path, "combined")
            return

        csv_path = find_btc_csv()
        if csv_path is not None:
            price, volume, ts = load_prices(csv_path, return_time=True)
            source = "btc"
            print(f"Using BTC data: {csv_path}")
        else:
            csv_path = find_stooq_csv()
            price, volume, ts = load_prices(csv_path, return_time=True)
            print(f"Using Stooq data: {csv_path}")
    except Exception as e:
        print(f"Falling back to synthetic prices: {e}")
        source = "synthetic"
        rng = np.random.default_rng(0)
        steps = rng.normal(loc=0.0, scale=0.01, size=1000)
        price = 100 + np.cumsum(steps)
        volume = np.ones_like(price) * 1e6
        ts = None

    run_trading_loop(
        price=price,
        volume=volume,
        source=source,
        time_index=ts,
        max_steps=max_steps,
        max_trades=max_trades,
        max_seconds=max_seconds,
        sleep_s=sleep_s,
        risk_frac=risk_frac,
        contract_mult=contract_mult,
        sigma_target=sigma_target,
        est_tax_rate=est_tax_rate,
        goal_cash_x=goal_cash_x,
        goal_eps=goal_eps,
        mdl_noise_mult=mdl_noise_mult,
        mdl_switch_penalty=mdl_switch_penalty,
        mdl_trade_penalty=mdl_trade_penalty,
        log_path=LOG,
        trade_log_path=pathlib.Path("logs/trade_log.csv"),
        progress_every=progress_every,
        log_level=log_level,
        edge_gate=edge_gate,
        edge_decay=edge_decay,
        edge_ema_alpha=edge_alpha,
        thesis_depth_max=thesis_depth_max,
        thesis_memory=thesis_memory,
        thesis_a_max=thesis_a_max,
        thesis_cooldown=thesis_cooldown,
        thesis_pbad_lo=thesis_pbad_lo,
        thesis_pbad_hi=thesis_pbad_hi,
        thesis_stress_lo=thesis_stress_lo,
        thesis_stress_hi=thesis_stress_hi,
        tc_k=tc_k,
        benchmark_x=benchmark_x,
    )
    emit_geometry(LOG, source)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None, help="Limit steps (debug).")
    ap.add_argument("--max-trades", type=int, default=None, help="Stop after N trades.")
    ap.add_argument("--max-seconds", type=float, default=None, help="Stop after N seconds per asset.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep per step.")
    ap.add_argument("--risk-frac", type=float, default=DEFAULT_RISK_FRAC, help="Risk fraction.")
    ap.add_argument("--sigma-target", type=float, default=SIGMA_TARGET, help="Target vol.")
    ap.add_argument("--est-tax-rate", type=float, default=EST_TAX_RATE, help="Estimated tax rate on realized PnL.")
    ap.add_argument("--goal-cash-x", type=float, default=GOAL_CASH_X, help="Spendable cash target.")
    ap.add_argument("--goal-eps", type=float, default=GOAL_EPS, help="Tail fraction for expected shortfall.")
    ap.add_argument("--mdl-noise-mult", type=float, default=MDL_NOISE_MULT, help="Sigma multiplier for active-trit threshold.")
    ap.add_argument("--mdl-switch-penalty", type=float, default=MDL_SWITCH_PENALTY, help="Side cost for state switches.")
    ap.add_argument("--mdl-trade-penalty", type=float, default=MDL_TRADE_PENALTY, help="Side cost for executed trades.")
    ap.add_argument("--progress-every", type=int, default=0, help="Print progress every N steps.")
    ap.add_argument("--all", action="store_true", help="Run over all CSVs under data/raw.")
    ap.add_argument("--raw-root", type=str, default="data/raw", help="Root directory to scan when --all set.")
    ap.add_argument(
        "--log-prefix",
        type=str,
        default="logs/trading_log",
        help="Log prefix when --all set (per-file log suffix added).",
    )
    ap.add_argument("--inter-run-sleep", type=float, default=0.5, help="Pause between runs when --all set.")
    ap.add_argument(
        "--log-combined",
        action="store_true",
        help="Write all tapes into a single log file when --all set.",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["quiet", "info", "trades", "verbose"],
        help="Output verbosity.",
    )
    ap.add_argument("--edge-gate", action="store_true", help="Enable edge-based cap decay gate.")
    ap.add_argument("--edge-decay", type=float, default=0.9, help="Cap multiplier when edge gate triggers.")
    ap.add_argument("--edge-alpha", type=float, default=EDGE_EMA_ALPHA, help="EMA alpha for edge metric.")
    ap.add_argument("--thesis-depth-max", type=int, default=THESIS_DEPTH_MAX, help="Max thesis memory depth.")
    ap.add_argument("--thesis-memory", action="store_true", help="Enable thesis memory FSM.")
    ap.add_argument("--thesis-a-max", type=int, default=THESIS_A_MAX, help="Max thesis age.")
    ap.add_argument("--thesis-cooldown", type=int, default=THESIS_COOLDOWN, help="Cooldown steps after thesis exit.")
    ap.add_argument("--thesis-pbad-lo", type=float, default=THESIS_PBAD_LO, help="Low p_bad threshold for thesis risk trit.")
    ap.add_argument("--thesis-pbad-hi", type=float, default=THESIS_PBAD_HI, help="High p_bad threshold for thesis risk trit.")
    ap.add_argument("--thesis-stress-lo", type=float, default=THESIS_STRESS_LO, help="Low stress threshold for thesis risk trit.")
    ap.add_argument("--thesis-stress-hi", type=float, default=THESIS_STRESS_HI, help="High stress threshold for thesis risk trit.")
    ap.add_argument("--tc-k", type=float, default=THESIS_TC_K, help="Transaction cost coefficient for reward_regret.")
    ap.add_argument("--benchmark-x", type=float, default=THESIS_BENCHMARK_X, help="Benchmark exposure for reward_regret.")
    ap.add_argument("--geometry-dir", type=str, default="logs/geometry", help="Output directory for geometry plots.")
    ap.add_argument("--geometry-simplex", action="store_true", help="Render simplex plot in geometry output.")
    ap.add_argument("--no-geometry-simplex", dest="geometry_simplex", action="store_false")
    ap.add_argument("--geometry-overlay", action="store_true", help="Render overlay plot in geometry output.")
    ap.add_argument("--no-geometry-overlay", dest="geometry_overlay", action="store_false")
    ap.add_argument("--geometry-plots", dest="geometry_plots", action="store_true", help="Emit geometry plots.")
    ap.add_argument("--no-geometry-plots", dest="geometry_plots", action="store_false", help="Skip geometry plots.")
    ap.set_defaults(geometry_plots=True, geometry_overlay=True, geometry_simplex=True)
    args = ap.parse_args()
    main(
        max_steps=args.max_steps,
        max_trades=args.max_trades,
        max_seconds=args.max_seconds,
        sleep_s=args.sleep,
        risk_frac=args.risk_frac,
        sigma_target=args.sigma_target,
        est_tax_rate=args.est_tax_rate,
        goal_cash_x=args.goal_cash_x,
        goal_eps=args.goal_eps,
        mdl_noise_mult=args.mdl_noise_mult,
        mdl_switch_penalty=args.mdl_switch_penalty,
        mdl_trade_penalty=args.mdl_trade_penalty,
        progress_every=args.progress_every,
        log_level=args.log_level,
        run_all=args.all,
        raw_root=args.raw_root,
        log_prefix=args.log_prefix,
        inter_run_sleep=args.inter_run_sleep,
        log_combined=args.log_combined,
        edge_gate=args.edge_gate,
        edge_decay=args.edge_decay,
        edge_alpha=args.edge_alpha,
        thesis_depth_max=args.thesis_depth_max,
        thesis_memory=args.thesis_memory,
        thesis_a_max=args.thesis_a_max,
        thesis_cooldown=args.thesis_cooldown,
        thesis_pbad_lo=args.thesis_pbad_lo,
        thesis_pbad_hi=args.thesis_pbad_hi,
        thesis_stress_lo=args.thesis_stress_lo,
        thesis_stress_hi=args.thesis_stress_hi,
        tc_k=args.tc_k,
        benchmark_x=args.benchmark_x,
        geometry_plots=args.geometry_plots,
        geometry_overlay=args.geometry_overlay,
        geometry_simplex=args.geometry_simplex,
        geometry_dir=args.geometry_dir,
    )

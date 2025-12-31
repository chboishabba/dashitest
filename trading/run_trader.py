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
import pandas as pd
import numpy as np

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
PLANE_BASE = 3.0           # base for surprise planes
PLANE_COUNT = 4            # number of surprise planes to track (0..PLANE_COUNT-1)

# Controls
HOLD_DECAY = 0.6           # exposure decay factor when action -> HOLD
VEL_EXIT = 3.0             # exit if latent velocity exceeds this while in position
PERSIST_RAMP = 0.05        # ramp factor for size in new regime
VETO_SIGMA = 5.0           # if realized sigma > VETO_SIGMA * sigma_target -> shrink size

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
    sleep_s=0.0,
    risk_frac: float = DEFAULT_RISK_FRAC,
    contract_mult: float = CONTRACT_MULT,
    sigma_target: float = SIGMA_TARGET,
    log_path: pathlib.Path = LOG,
    progress_every: int = 0,
    est_tax_rate: float = EST_TAX_RATE,
    goal_cash_x: float = GOAL_CASH_X,
    goal_eps: float = GOAL_EPS,
    mdl_noise_mult: float = MDL_NOISE_MULT,
    mdl_switch_penalty: float = MDL_SWITCH_PENALTY,
    mdl_trade_penalty: float = MDL_TRADE_PENALTY,
    log_level: str = "info",
):
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
        log_path.unlink(missing_ok=True)
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
    side_cost = 0.0
    thesis_age = 0      # how long we've held a non-zero thesis
    state_age = 0       # how long the field state has persisted
    align_age = 0       # how long state and thesis have been aligned
    dz_min = 5e-5  # minimum dead-zone
    cost = 0.0005
    rows = []
    recent_rets = []
    total_steps = len(price) - 1
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    pnl_delta_window = []
    window = 200
    t0_ts = time_index[0] if time_index is not None and len(time_index) else None
    for t in range(1, total_steps + 1):
        ret = price[t] / price[t - 1] - 1.0
        recent_rets.append(ret)
        if len(recent_rets) > 200:
            recent_rets.pop(0)
        sigma = np.std(recent_rets) if recent_rets else dz_min
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
        banned = bool(t < len(bad_flag) and bad_flag[t])
        order = desired - pos
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

        # Update persistence clocks
        state_age = state_age + 1 if desired == prev_action else 0 if desired == 0 else 1
        if pos != 0:
            thesis_age += 1
        else:
            thesis_age = 0
        if pos != 0 and desired != 0 and np.sign(pos) == np.sign(desired):
            align_age += 1
        else:
            align_age = 0

        # Triadic control: BAN is sovereign, then HOLD decay, then normal ramp
        if banned:
            # hard flatten; BAN overrides any directional intent
            cap = max(cap, abs(pos))
            fill = np.clip(-pos, -cap, cap)
            desired = 0
        elif desired == 0:
            # decay exposure toward zero
            cap = cap * (1.0 - HOLD_DECAY) + 1e-9  # keep a tiny ability to act
            fill = np.clip(-pos, -cap, cap)
        else:
            # exit if latent velocity too high while in position
            if z_vel > VEL_EXIT and pos != 0:
                fill = np.clip(-pos, -cap, cap)
            else:
                # ramp toward target with persistence factor; faster if align_age large
                target = desired * cap
                ramp = PERSIST_RAMP * (1.0 + align_age * 0.01)
                step = ramp * (target - pos)
                fill = np.clip(step, -cap, cap)

        slippage = IMPACT_COEFF * abs(fill / max(cap, 1e-9))
        price_exec = price[t] * (1 + slippage * np.sign(fill))
        cash -= fill * price_exec
        pos += fill
        fee = cost * abs(fill)
        fees_accrued += fee
        slippage_cost = abs(fill) * abs(price_exec - price[t])
        pnl = cash + pos * price[t] - fee
        capital_at_risk = abs(fill * price_exec)
        delta_pnl = pnl - prev_pnl
        cash_eff = delta_pnl / (capital_at_risk + 1e-9) if capital_at_risk > 0 else np.nan
        exec_eff = 1.0 - (slippage_cost + fee) / (capital_at_risk + 1e-9) if capital_at_risk > 0 else np.nan
        cash_vel = (cash - START_CASH) / max(t, 1)
        realized_pnl = cash - START_CASH
        tax_est = est_tax_rate * max(realized_pnl, 0.0)
        c_spend = cash - tax_est - fees_accrued

        # Proxy MDL and stress (plane-aware surprise)
        noise_threshold = mdl_noise_mult * max(sigma, 1e-9)
        abs_ret = abs(ret)
        if abs_ret > noise_threshold:
            plane_index = int(np.floor(np.log(abs_ret / noise_threshold) / np.log(PLANE_BASE)))
            plane_index = max(0, min(plane_index, PLANE_COUNT - 1))
        else:
            plane_index = -1
        active_trit = 1.0 if plane_index >= 0 else 0.0
        active_trit_count += active_trit
        plane_hits = []
        plane_rates = []
        for k in range(PLANE_COUNT):
            hit = 1.0 if plane_index == k else 0.0
            plane_counts[k] += hit
            plane_hits.append(hit)
            plane_rates.append(plane_counts[k] / max(t, 1))
        if desired != prev_action:
            side_cost += mdl_switch_penalty
        if fill != 0:
            side_cost += mdl_trade_penalty
        mdl_rate = (side_cost + active_trit_count) / max(t, 1)
        stress = active_trit_count / max(t, 1)

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
        can_trade = int(not banned)
        row = {
            "t": t,
            "ts": time_index[t] if time_index is not None and t < len(time_index) else np.nan,
            "price": price[t],
            "volume": volume[t] if t < len(volume) else np.nan,
            "pnl": pnl,
            "cash": cash,
            "cash_eff": cash_eff,
            "exec_eff": exec_eff,
            "cash_vel": cash_vel,
            "c_spend": c_spend,
            "goal_prob": goal_prob,
            "goal_align": goal_align,
            "goal_pressure": goal_pressure,
            "regret": regret,
            "es_shortfall": es_shortfall,
            "mdl_rate": mdl_rate,
            "stress": stress,
            "active_trit": active_trit,
            "plane_index": plane_index,
            "plane0": plane_hits[0] if PLANE_COUNT > 0 else 0.0,
            "plane1": plane_hits[1] if PLANE_COUNT > 1 else 0.0,
            "plane2": plane_hits[2] if PLANE_COUNT > 2 else 0.0,
            "plane3": plane_hits[3] if PLANE_COUNT > 3 else 0.0,
            "plane_rate0": plane_rates[0] if PLANE_COUNT > 0 else 0.0,
            "plane_rate1": plane_rates[1] if PLANE_COUNT > 1 else 0.0,
            "plane_rate2": plane_rates[2] if PLANE_COUNT > 2 else 0.0,
            "plane_rate3": plane_rates[3] if PLANE_COUNT > 3 else 0.0,
            "p_bad": p_bad[t] if t < len(p_bad) else np.nan,
            "bad_flag": int(bad_flag[t]) if t < len(bad_flag) else 0,
            "ban": int(banned),
            "can_trade": can_trade,
            "z_norm": abs(z),
            "z_vel": z_vel,
            "hold": int(desired == 0),
            "entropy": 0.0,
            "regime": 0,
            "action": np.sign(fill) if fill != 0 else 0,
            "source": source,
            "pos": pos,
            "fill": fill,
            "cap": cap,
            "equity": equity,
            "prev_action": prev_action,
            "thesis_age": thesis_age,
            "state_age": state_age,
            "align_age": align_age,
        }
        rows.append(row)
        if log_path:
            pd.DataFrame([row]).to_csv(
                log_path, mode="a", header=not log_path.exists(), index=False
            )
        if log_level in {"trades", "verbose"} and fill != 0:
            print(
                f"[trade] t={t:6d} px={price[t]:.2f} fill={fill:.4f} pos={pos:.4f} "
                f"cap={cap:.4f} act={int(row['action'])} banned={int(banned)} "
                f"cash_eff={row['cash_eff']:.5f} exec_eff={row['exec_eff']:.5f} "
                f"c_spend={row['c_spend']:.2f} goal_p={row['goal_prob']:.3f} "
                f"mdl={row['mdl_rate']:.4f} stress={row['stress']:.4f} "
                f"plane={row['plane_index']} can_trade={row['can_trade']} "
                f"regret={row['regret']:.2f}"
            )
        if log_level != "quiet" and progress_every and (t % progress_every == 0 or t == total_steps):
            print(
                f"[{source}] t={t:6d}/{total_steps:6d} pnl={row['pnl']:.4f} pos={row['pos']:.4f} "
                f"fill={row['fill']:.4f} act={int(row['action'])} p_bad={row['p_bad']:.3f} "
                f"bad={row['bad_flag']} cash_eff={row['cash_eff']:.5f} exec_eff={row['exec_eff']:.5f} "
                f"mdl_rate={row['mdl_rate']:.4f} stress={row['stress']:.4f} goal_prob={row['goal_prob']:.3f}"
            )
        z_prev = z
        prev_action = desired
        prev_cash = cash
        prev_pnl = pnl
        prev_goal_prob = goal_prob
        c_spend_prev = c_spend
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
    print(f"Run complete: source={source}, steps={len(rows)}, trades={trades}, pnl={total_pnl:.4f}")

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
    sleep_s=0.0,
    risk_frac: float = DEFAULT_RISK_FRAC,
    contract_mult: float = CONTRACT_MULT,
    sigma_target: float = SIGMA_TARGET,
    progress_every: int = 0,
    log_level: str = "info",
):
    source = "stooq"
    try:
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
        sleep_s=sleep_s,
        risk_frac=risk_frac,
        contract_mult=contract_mult,
        sigma_target=sigma_target,
        log_path=LOG,
        progress_every=progress_every,
        log_level=log_level,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None, help="Limit steps (debug).")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep per step.")
    ap.add_argument("--risk-frac", type=float, default=DEFAULT_RISK_FRAC, help="Risk fraction.")
    ap.add_argument("--sigma-target", type=float, default=SIGMA_TARGET, help="Target vol.")
    ap.add_argument("--progress-every", type=int, default=0, help="Print progress every N steps.")
    ap.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["quiet", "info", "trades", "verbose"],
        help="Output verbosity.",
    )
    args = ap.parse_args()
    main(
        max_steps=args.max_steps,
        sleep_s=args.sleep,
        risk_frac=args.risk_frac,
        sigma_target=args.sigma_target,
        progress_every=args.progress_every,
        log_level=args.log_level,
    )

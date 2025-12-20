"""
run_trader.py
--------------
Streams a simple ternary trading loop over cached Stooq data and logs
to logs/trading_log.csv for the training_dashboard to visualize.

Usage:
  python run_trader.py
Then in another terminal:
  python training_dashboard.py --log logs/trading_log.csv --refresh 0.5
"""

import time
import pathlib
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

# Controls
HOLD_DECAY = 0.6           # exposure decay factor when action -> HOLD
VEL_EXIT = 3.0             # exit if latent velocity exceeds this while in position
PERSIST_RAMP = 0.05        # ramp factor for size in new regime
VETO_SIGMA = 5.0           # if realized sigma > VETO_SIGMA * sigma_target -> shrink size


def find_btc_csv():
    raw = pathlib.Path("data/raw/stooq")
    if not raw.exists():
        return None
    # prefer intraday then daily
    files = sorted(list(raw.glob("btc_intraday*.csv")) + list(raw.glob("btc*.csv")))
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
        raise FileNotFoundError("data/raw/stooq not found; run data_downloader.py first.")
    files = sorted(raw.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No Stooq CSVs found; run data_downloader.py.")
    # pick first valid CSV with Close column and data
    for f in files:
        try:
            df_head = pd.read_csv(f, nrows=5)
            cols = [c.lower() for c in df_head.columns]
            if any(k in cols for k in ("close", "zamkniecie")) and not df_head.empty:
                return f
        except Exception:
            continue
    raise FileNotFoundError("No valid Stooq CSV with Close/Zamkniecie column; re-run data_downloader.py with .us symbols.")


def load_prices(path: pathlib.Path):
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

        return close_series.to_numpy(), vol_series.to_numpy()

    # try basic read, then fallback to skiprows if too many non-numeric
    for reader in (read_basic, read_skip):
        try:
            df = reader(path)
            close, vol = parse_df(df)
            # replace nonpositive/NaN volume
            pos_vol = vol[np.isfinite(vol) & (vol > 0)]
            fallback_vol = np.median(pos_vol) if pos_vol.size else 1e6
            vol = np.where((~np.isfinite(vol)) | (vol <= 0), fallback_vol, vol)
            # require reasonable length
            if len(close) >= 10 and np.isfinite(close).any():
                return close, vol
        except Exception:
            continue
    raise ValueError(f"Could not parse prices from {path}")


def main(
    max_steps=None,
    sleep_s=0.0,
    risk_frac: float = DEFAULT_RISK_FRAC,
    contract_mult: float = CONTRACT_MULT,
    sigma_target: float = SIGMA_TARGET,
):
    source = "stooq"
    try:
        csv_path = find_btc_csv()
        if csv_path is not None:
            price, volume = load_prices(csv_path)
            source = "btc"
            print(f"Using BTC data: {csv_path}")
        else:
            csv_path = find_stooq_csv()
            price, volume = load_prices(csv_path)
            print(f"Using Stooq data: {csv_path}")
    except Exception as e:
        print(f"Falling back to synthetic prices: {e}")
        source = "synthetic"
        rng = np.random.default_rng(0)
        steps = rng.normal(loc=0.0, scale=0.01, size=1000)
        price = 100 + np.cumsum(steps)
        volume = np.ones_like(price) * 1e6
    LOG.unlink(missing_ok=True)
    cash = START_CASH
    pos = 0.0
    z_prev = 0.0
    prev_action = 0
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

        # Triadic control: decay on HOLD, exit on high velocity, ramp size when aligned persists
        if desired == 0:
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
        pnl = cash + pos * price[t] - cost * abs(fill)
        row = {
            "t": t,
            "price": price[t],
            "pnl": pnl,
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
        pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        z_prev = z
        prev_action = desired
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

    # append run history
    summary = {
        "timestamp": pd.Timestamp.utcnow(),
        "source": source,
        "steps": len(rows),
        "trades": trades,
        "pnl": total_pnl,
        "hold_pct": hold_pct,
        "max_drawdown": max_drawdown,
    }
    pd.DataFrame([summary]).to_csv(RUN_HISTORY, mode="a", header=not RUN_HISTORY.exists(), index=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_step_accounting(
    price_t: float,
    cash: float,
    pos: float,
    fee: float,
    slippage_cost: float,
    capital_at_risk: float,
    prev_pnl: float,
    pos_prev: float,
    edge_ema: float,
    edge_ema_alpha: float,
    fees_accrued: float,
    est_tax_rate: float,
    start_cash: float,
    t: int,
) -> tuple[dict, float]:
    pnl = cash + pos * price_t - fee
    delta_pnl = pnl - prev_pnl
    edge_raw = delta_pnl / (abs(pos_prev) + 1e-9)
    edge_ema = (1.0 - edge_ema_alpha) * edge_ema + edge_ema_alpha * edge_raw
    cash_eff = delta_pnl / (capital_at_risk + 1e-9) if capital_at_risk > 0 else np.nan
    exec_eff = (
        1.0 - (slippage_cost + fee) / (capital_at_risk + 1e-9)
        if capital_at_risk > 0
        else np.nan
    )
    cash_vel = (cash - start_cash) / max(t, 1)
    realized_pnl = cash - start_cash
    tax_est = est_tax_rate * max(realized_pnl, 0.0)
    c_spend = cash - tax_est - fees_accrued
    metrics = {
        "pnl": pnl,
        "delta_pnl": delta_pnl,
        "edge_raw": edge_raw,
        "edge_ema": edge_ema,
        "cash_eff": cash_eff,
        "exec_eff": exec_eff,
        "cash_vel": cash_vel,
        "realized_pnl": realized_pnl,
        "tax_est": tax_est,
        "c_spend": c_spend,
    }
    return metrics, edge_ema


def build_run_summary(
    rows: list[dict],
    p_bad: np.ndarray,
    bad_flag: np.ndarray,
    start_cash: float,
    stop_reason: str,
    elapsed_s: float,
) -> dict:
    total_pnl = rows[-1]["pnl"] if rows else 0.0
    trades = sum(1 for r in rows if r.get("trade_closed"))
    hold_pct = sum(1 for r in rows if r["action"] == 0) / len(rows) if rows else 0.0
    max_drawdown = 0.0
    if rows:
        equity_curve = pd.Series([r["pnl"] for r in rows])
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max).min()
        max_drawdown = float(drawdown)

    ret_all = ret_good = ret_bad = float("nan")
    if rows:
        equity_series = pd.Series([r["pnl"] for r in rows])
        eq_prev = equity_series.shift(1).fillna(start_cash)
        eq_ret = equity_series.diff() / eq_prev.replace(0, np.nan)
        bad_mask = pd.Series([bool(r["bad_flag"]) for r in rows])
        ret_all = float(eq_ret.mean(skipna=True))
        ret_good = float(eq_ret[~bad_mask].mean(skipna=True))
        ret_bad = float(eq_ret[bad_mask].mean(skipna=True))

    return {
        "timestamp": pd.Timestamp.utcnow(),
        "source": rows[-1]["source"] if rows else "",
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

from __future__ import annotations

import pandas as pd


def emit_step_row(row: dict, log_path) -> None:
    if log_path is None:
        return
    pd.DataFrame([row]).to_csv(
        log_path, mode="a", header=not log_path.exists(), index=False
    )


def emit_trade_row(trade_row: dict, trade_log_path) -> None:
    if trade_log_path is None:
        return
    pd.DataFrame([trade_row]).to_csv(
        trade_log_path, mode="a", header=not trade_log_path.exists(), index=False
    )


def emit_run_summary(summary: dict, run_history_path) -> None:
    pd.DataFrame([summary]).to_csv(
        run_history_path, mode="a", header=not run_history_path.exists(), index=False
    )


def emit_trade_print(
    ts: str,
    t: int,
    price: float,
    fill: float,
    pos: float,
    cap: float,
    action: int,
    permission: int,
    cash_eff: float,
    exec_eff: float,
    c_spend: float,
    goal_prob: float,
    mdl_rate: float,
    stress: float,
    plane_index: int,
    can_trade: int,
    regret: float,
    unrealized: float,
    realized_pnl_step: float,
    avg_entry_price: float,
) -> None:
    print(
        f"[{ts}] [trade] t={t:6d} px={price:.2f} fill={fill:.4f} pos={pos:.4f} "
        f"cap={cap:.4f} act={action} banned={int(permission == -1)} "
        f"cash_eff={cash_eff:.5f} exec_eff={exec_eff:.5f} "
        f"c_spend={c_spend:.2f} goal_p={goal_prob:.3f} "
        f"mdl={mdl_rate:.4f} stress={stress:.4f} "
        f"plane={plane_index} can_trade={can_trade} "
        f"regret={regret:.2f} u_pnl={unrealized:.2f} "
        f"r_pnl={realized_pnl_step:.2f} entry={avg_entry_price:.2f}"
    )


def emit_trade_close_print(
    ts: str,
    trade_id: int,
    close_reason: str,
    trade_pnl: float,
    trade_pnl_pct: float,
    trade_duration: int,
    entry_price: float,
    exit_price: float,
) -> None:
    print(
        f"[{ts}] [trade] close id={trade_id} reason={close_reason} "
        f"pnl={trade_pnl:.4f} pct={trade_pnl_pct:.4f} dur={trade_duration} "
        f"entry={entry_price:.4f} exit={exit_price:.4f}"
    )


def emit_progress_print(
    ts: str,
    source: str,
    t: int,
    total_steps: int,
    pnl: float,
    pos: float,
    fill: float,
    action: int,
    p_bad: float,
    bad_flag: bool,
    cash_eff: float,
    exec_eff: float,
    mdl_rate: float,
    stress: float,
    goal_prob: float,
) -> None:
    print(
        f"[{ts}] [{source}] t={t:6d}/{total_steps:6d} pnl={pnl:.4f} pos={pos:.4f} "
        f"fill={fill:.4f} act={action} p_bad={p_bad:.3f} "
        f"bad={bad_flag} cash_eff={cash_eff:.5f} exec_eff={exec_eff:.5f} "
        f"mdl_rate={mdl_rate:.4f} stress={stress:.4f} goal_prob={goal_prob:.3f}"
    )

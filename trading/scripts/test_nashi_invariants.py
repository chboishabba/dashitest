from __future__ import annotations

from pathlib import Path
import sys
import json
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from nashi.certification import certification_census, summarize_census
from nashi.invariants import (
    check_artifact_parity,
    load_duckdb_family_rows,
    check_step_rows,
    load_duckdb_step_rows,
    load_ndjson_rows,
    summarize_parity_violations,
    summarize_violations,
)

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover
    duckdb = None


def _base_row() -> dict[str, object]:
    return {
        "ts": 1710000000000,
        "symbol": "BTCUSDT",
        "action": 1,
        "fill": 0.25,
        "exposure": 0.25,
        "intent_direction": 1,
        "intent_target": 0.25,
        "justification_chain": "positive_edge -> trade_normal -> bar_exec -> phase9_capital_kernel_v1 -> expected_surplus=2.0 -> realized_surplus=1.0",
        "just_regime": "positive_edge",
        "just_posture": "trade_normal",
        "just_actuator": "bar_exec",
        "just_cost_model": "phase9_capital_kernel_v1",
        "nashi_family_certified": 1,
        "nashi_family_trade_certified": 1,
        "nashi_family_preserve_certified": 0,
        "nashi_family_tail_localized": 0,
        "nashi_family_spread_dominated": 0,
        "nashi_family_hostile_regime": 0,
        "nashi_family_reasons": "",
        "mw_refusal_level": "normal",
        "mw_forced_hold": 0,
        "mw_forced_ban": 0,
        "mw_max_exposure": 1.0,
        "proposed_expected_surplus": 1.0,
        "proposed_expected_gross_surplus": 2.0,
        "proposed_expected_cost": 1.0,
        "expected_surplus": 1.0,
        "expected_gross_surplus": 2.0,
        "expected_cost": 1.0,
        "executed_expected_surplus": 1.0,
        "executed_expected_gross_surplus": 2.0,
        "executed_expected_cost": 1.0,
        "executable_opportunity": 1,
        "realized_surplus": 0.5,
        "realized_efficiency": 0.5,
        "aligned_expected_surplus": 1.0,
        "aligned_realized_surplus": 0.5,
        "aligned_realized_efficiency": 0.5,
        "phase9_cost_survival_ratio": 2.0,
        "phase9_microstructure_kills_edge": 0,
        "hazard_source": "none",
        "hazard_tightened_source": "none",
    }


def main() -> int:
    valid = _base_row()
    violations = check_step_rows([valid])
    assert violations == [], violations

    hazard_preserve = _base_row()
    hazard_preserve["action"] = 0
    hazard_preserve["fill"] = 0.0
    hazard_preserve["exposure"] = 0.0
    hazard_preserve["intent_direction"] = 0
    hazard_preserve["intent_target"] = 0.0
    hazard_preserve["mw_refusal_level"] = "hold"
    hazard_preserve["mw_forced_hold"] = 1
    hazard_preserve["mw_max_exposure"] = 0.0
    hazard_preserve["nashi_family_trade_certified"] = 0
    hazard_preserve["nashi_family_preserve_certified"] = 1
    hazard_preserve["expected_surplus"] = 0.0
    hazard_preserve["expected_gross_surplus"] = 0.0
    hazard_preserve["expected_cost"] = 0.0
    hazard_preserve["executed_expected_surplus"] = 0.0
    hazard_preserve["executed_expected_gross_surplus"] = 0.0
    hazard_preserve["executed_expected_cost"] = 0.0
    hazard_preserve["executable_opportunity"] = 0
    hazard_preserve["realized_surplus"] = 0.0
    hazard_preserve["realized_efficiency"] = 0.0
    hazard_preserve["hazard_active"] = 1
    hazard_preserve["hazard_name"] = "vol_spike"
    hazard_preserve["hazard_reason"] = "volatility_shock"
    hazard_preserve["hazard_forced_hold"] = 1
    hazard_preserve["hazard_forced_ban"] = 0
    hazard_preserve["hazard_contextual_pressure"] = 0.7
    hazard_preserve["hazard_contextual_active"] = 1
    hazard_preserve["hazard_contextual_label"] = "vol_spike"
    hazard_preserve["hazard_source"] = "contextual"
    hazard_preserve["hazard_tightened_source"] = "contextual"
    violations = check_step_rows([hazard_preserve])
    assert violations == [], violations

    invalid = _base_row()
    invalid["justification_chain"] = ""
    invalid["nashi_family_trade_certified"] = 1
    invalid["nashi_family_preserve_certified"] = 1
    invalid["nashi_family_spread_dominated"] = 1
    invalid["nashi_family_reasons"] = ""
    invalid["mw_refusal_level"] = "ban"
    invalid["mw_forced_ban"] = 1
    invalid["mw_forced_hold"] = 0
    invalid["mw_max_exposure"] = 0.5
    invalid["action"] = 1
    invalid["intent_direction"] = 1
    invalid["intent_target"] = 0.25
    invalid["exposure"] = 0.25
    invalid["expected_surplus"] = 1.1
    invalid["proposed_expected_surplus"] = 1.1
    invalid["executed_expected_surplus"] = 1.1
    invalid["executable_opportunity"] = 1
    invalid["realized_surplus"] = 0.3
    invalid["realized_efficiency"] = 0.1
    invalid["aligned_realized_efficiency"] = 0.1
    invalid["phase9_cost_survival_ratio"] = 0.5
    invalid["phase9_microstructure_kills_edge"] = 0

    violations = check_step_rows([invalid])
    summary = summarize_violations(violations)

    assert summary["trade_requires_justification"] == 1
    assert summary["family_dual_certification"] == 1
    assert summary["spread_dominated_blocks_trade_certification"] == 1
    assert summary["family_reason_missing_microstructure"] == 1
    assert summary["ban_forces_flat_action"] == 1
    assert summary["ban_sets_zero_max_exposure"] == 1
    assert summary["ban_implies_hold"] == 1
    assert summary["phase9_expected_surplus_arithmetic"] == 1
    assert summary["phase9_proposed_expected_surplus_arithmetic"] == 1
    assert summary["phase9_executed_expected_surplus_arithmetic"] == 1
    assert summary["phase9_executable_opportunity_blocking_consistency"] == 1
    assert summary["phase9_realized_efficiency_consistency"] == 1
    assert summary["phase9_aligned_realized_efficiency_consistency"] == 1

    micro_invalid = _base_row()
    micro_invalid["mw_refusal_level"] = "normal"
    micro_invalid["mw_forced_hold"] = 0
    micro_invalid["mw_forced_ban"] = 0
    micro_invalid["hold"] = 0
    micro_invalid["action"] = 1
    micro_invalid["intent_direction"] = 1
    micro_invalid["intent_target"] = 0.25
    micro_invalid["expected_surplus"] = 1.1
    micro_invalid["executed_expected_surplus"] = 1.1
    micro_invalid["expected_gross_surplus"] = 1.6
    micro_invalid["executed_expected_gross_surplus"] = 1.6
    micro_invalid["expected_cost"] = 0.5
    micro_invalid["executed_expected_cost"] = 0.5
    micro_invalid["phase9_cost_survival_ratio"] = 0.5
    micro_invalid["phase9_microstructure_kills_edge"] = 0
    micro_summary = summarize_violations(check_step_rows([micro_invalid]))
    assert micro_summary["phase9_microstructure_flag_consistency"] == 1

    hazard_invalid = _base_row()
    hazard_invalid["mw_refusal_level"] = "normal"
    hazard_invalid["mw_forced_hold"] = 0
    hazard_invalid["mw_forced_ban"] = 0
    hazard_invalid["hazard_active"] = 1
    hazard_invalid["hazard_name"] = ""
    hazard_invalid["hazard_reason"] = ""
    hazard_invalid["hazard_forced_hold"] = 1
    hazard_invalid["hazard_forced_ban"] = 1
    hazard_invalid["hazard_contextual_pressure"] = 0.6
    hazard_invalid["hazard_contextual_active"] = 1
    hazard_invalid["hazard_contextual_label"] = "macro_news"
    hazard_invalid["hazard_source"] = "synthetic_only"
    hazard_invalid["hazard_tightened_source"] = "synthetic_only"
    hazard_invalid["nashi_family_trade_certified"] = 1
    hazard_invalid["nashi_family_preserve_certified"] = 0
    hazard_invalid["action"] = 1
    hazard_invalid["fill"] = 0.25
    hazard_invalid["exposure"] = 0.25
    hazard_invalid["intent_direction"] = 1
    hazard_invalid["intent_target"] = 0.25
    hazard_invalid["mw_max_exposure"] = 1.0
    violations = check_step_rows([hazard_invalid])
    summary = summarize_violations(violations)
    assert summary["hazard_requires_explanation"] == 1
    assert summary["hazard_hold_aligns_meta_witness"] == 1
    assert summary["hazard_ban_aligns_meta_witness"] == 1
    assert summary["hazard_ban_implies_hold"] == 1
    assert summary["hazard_ban_forces_flat_action"] == 1
    assert summary["hazard_source_provenance_consistency"] == 1
    assert summary["hazard_tightened_source_consistency"] == 1

    zero_exec = _base_row()
    zero_exec["executed_expected_surplus"] = 0.0
    zero_exec["executed_expected_gross_surplus"] = 0.0
    zero_exec["executed_expected_cost"] = 0.0
    zero_exec["executable_opportunity"] = 0
    zero_exec["realized_surplus"] = 0.0
    zero_exec["realized_efficiency"] = 0.0
    violations = check_step_rows([zero_exec])
    assert violations == [], violations

    hazard_ban = _base_row()
    hazard_ban["ts"] = hazard_ban["ts"] + 1000
    hazard_ban["action"] = 0
    hazard_ban["fill"] = 0.0
    hazard_ban["exposure"] = 0.0
    hazard_ban["intent_direction"] = 0
    hazard_ban["intent_target"] = 0.0
    hazard_ban["mw_refusal_level"] = "ban"
    hazard_ban["mw_forced_hold"] = 1
    hazard_ban["mw_forced_ban"] = 1
    hazard_ban["mw_max_exposure"] = 0.0
    hazard_ban["nashi_family_trade_certified"] = 0
    hazard_ban["nashi_family_preserve_certified"] = 0
    hazard_ban["expected_surplus"] = 0.0
    hazard_ban["expected_gross_surplus"] = 0.0
    hazard_ban["expected_cost"] = 0.0
    hazard_ban["executed_expected_surplus"] = 0.0
    hazard_ban["executed_expected_gross_surplus"] = 0.0
    hazard_ban["executed_expected_cost"] = 0.0
    hazard_ban["executable_opportunity"] = 0
    hazard_ban["realized_surplus"] = 0.0
    hazard_ban["realized_efficiency"] = 0.0
    hazard_ban["hazard_active"] = 1
    hazard_ban["hazard_name"] = "session_break"
    hazard_ban["hazard_reason"] = "market_halt"
    hazard_ban["hazard_forced_hold"] = 1
    hazard_ban["hazard_forced_ban"] = 1
    census = certification_census(pd.DataFrame([hazard_preserve, hazard_ban]), window_rows=8)
    cert_summary = summarize_census(census)
    assert cert_summary["hazard_active_windows"] == 1
    assert cert_summary["hazard_ban_windows"] == 1
    assert cert_summary["hazard_preserve_windows"] == 1
    assert cert_summary["hazard_trade_windows"] == 0
    assert cert_summary["hazard_active_steps"] == 2
    assert cert_summary["hazard_ban_steps"] == 1
    assert cert_summary["hazard_preserve_steps"] == 1
    assert cert_summary["hazard_summary"]["hazard_name_modes"]
    assert cert_summary["hazard_summary"]["hazard_reason_modes"]

    if duckdb is not None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "steps.csv"
            duckdb_path = Path(tmp) / "steps.duckdb"
            decision_path = Path(tmp) / "decisions.ndjson"
            ohlc_path = Path(tmp) / "ohlc.ndjson"

            parity_row = _base_row() | {
                "t": 0,
                "price": 50000.0,
                "hold": 0,
                "pnl": 12.0,
                "mw_reason": "normal",
                "nashi_status": "interior",
                "realized_surplus": 0.25,
                "hazard_active": 1,
                "hazard_name": "vol_spike",
                "hazard_reason": "volatility_shock",
            }
            pd.DataFrame([parity_row]).to_csv(csv_path, index=False)

            con = duckdb.connect(str(duckdb_path))
            con.execute(
                """
                create table nashi_steps as
                select
                    to_timestamp(? / 1000.0) as timestamp,
                    ?::bigint as t,
                    ?::varchar as symbol,
                    ?::double as price,
                    ?::integer as action,
                    ?::boolean as hold,
                    ?::double as exposure,
                    ?::double as pnl,
                    ?::double as fill,
                    ?::double as fee,
                    ?::double as expected_surplus,
                    ?::double as realized_surplus,
                    ?::varchar as mw_reason,
                    ?::varchar as nashi_status,
                    ?::boolean as hazard_active,
                    ?::varchar as hazard_source,
                    ?::varchar as hazard_tightened_source,
                    ?::varchar as hazard_name,
                    ?::varchar as hazard_reason
                """,
                [
                    parity_row["ts"],
                    parity_row["t"],
                    parity_row["symbol"],
                    parity_row["price"],
                    parity_row["action"],
                    bool(parity_row["hold"]),
                    parity_row["exposure"],
                    parity_row["pnl"],
                    parity_row["fill"],
                    0.0,
                    parity_row["expected_surplus"],
                    parity_row["realized_surplus"],
                    parity_row["mw_reason"],
                    parity_row["nashi_status"],
                    bool(parity_row["hazard_active"]),
                    parity_row["hazard_source"],
                    parity_row["hazard_tightened_source"],
                    parity_row["hazard_name"],
                    parity_row["hazard_reason"],
                ],
            )
            con.execute(
                """
                create table nashi_family_certifications as
                select
                    to_timestamp(? / 1000.0) as timestamp,
                    ?::varchar as symbol,
                    ?::varchar as family_class,
                    ?::varchar as family_constructor,
                    ?::boolean as certified,
                    ?::boolean as trade_certified,
                    ?::boolean as preserve_certified,
                    ?::boolean as tail_localized,
                    ?::boolean as spread_dominated,
                    ?::boolean as hostile_regime,
                    ?::double as arrow_boundary_share,
                    ?::double as microstructure_kill_share,
                    ?::bigint as window_size,
                    ?::varchar as reasons,
                    ?::varchar as source_file
                """,
                [
                    parity_row["ts"],
                    parity_row["symbol"],
                    "interior_family",
                    "InteriorFamily",
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    0.0,
                    0.0,
                    1,
                    "",
                    "test",
                ],
            )
            con.close()

            with decision_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "timestamp": parity_row["ts"],
                    "symbol": parity_row["symbol"],
                    "direction": parity_row["action"],
                    "hazard_active": parity_row["hazard_active"],
                    "hazard_reason": parity_row["hazard_reason"],
                }))
                fh.write("\n")
            with ohlc_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "timestamp": parity_row["ts"],
                    "close": parity_row["price"],
                }))
                fh.write("\n")

            parity = check_artifact_parity(
                [parity_row],
                duckdb_rows=load_duckdb_step_rows(duckdb_path),
                family_rows=load_duckdb_family_rows(duckdb_path),
                decision_rows=load_ndjson_rows(decision_path),
                ohlc_rows=load_ndjson_rows(ohlc_path),
            )
            assert parity == [], parity

            bad_decisions = [{"timestamp": parity_row["ts"] + 1, "symbol": parity_row["symbol"], "direction": parity_row["action"]}]
            parity = check_artifact_parity(
                [parity_row],
                decision_rows=bad_decisions,
            )
            parity_summary = summarize_parity_violations(parity)
            assert parity_summary["csv_decision_timestamp_mismatch"] == 1

            bad_hazard_decisions = [{
                "timestamp": parity_row["ts"],
                "symbol": parity_row["symbol"],
                "direction": parity_row["action"],
                "hazard_active": parity_row["hazard_active"],
                "hazard_reason": "different_reason",
            }]
            parity = check_artifact_parity(
                [parity_row],
                decision_rows=bad_hazard_decisions,
            )
            parity_summary = summarize_parity_violations(parity)
            assert parity_summary["csv_decision_hazard_reason_mismatch"] == 1

            con = duckdb.connect(str(duckdb_path))
            con.execute("delete from nashi_steps")
            con.execute(
                """
                insert into nashi_steps
                select
                    to_timestamp(? / 1000.0) as timestamp,
                    ?::bigint as t,
                    ?::varchar as symbol,
                    ?::double as price,
                    ?::integer as action,
                    ?::boolean as hold,
                    ?::double as exposure,
                    ?::double as pnl,
                    ?::double as fill,
                    ?::double as fee,
                    ?::double as expected_surplus,
                    ?::double as realized_surplus,
                    ?::varchar as mw_reason,
                    ?::varchar as nashi_status,
                    ?::boolean as hazard_active,
                    ?::varchar as hazard_source,
                    ?::varchar as hazard_tightened_source,
                    ?::varchar as hazard_name,
                    ?::varchar as hazard_reason
                """,
                [
                    parity_row["ts"],
                    parity_row["t"],
                    parity_row["symbol"],
                    parity_row["price"],
                    parity_row["action"],
                    bool(parity_row["hold"]),
                    parity_row["exposure"],
                    parity_row["pnl"],
                    parity_row["fill"],
                    0.0,
                    parity_row["expected_surplus"],
                    parity_row["realized_surplus"],
                    parity_row["mw_reason"],
                    parity_row["nashi_status"],
                    bool(parity_row["hazard_active"]),
                    "contextual",
                    parity_row["hazard_tightened_source"],
                    parity_row["hazard_name"],
                    parity_row["hazard_reason"],
                ],
            )
            con.close()
            parity = check_artifact_parity(
                [parity_row],
                duckdb_rows=load_duckdb_step_rows(duckdb_path),
            )
            parity_summary = summarize_parity_violations(parity)
            assert parity_summary["csv_duckdb_value_mismatch"] == 1

            con = duckdb.connect(str(duckdb_path))
            con.execute("delete from nashi_steps")
            con.execute(
                """
                insert into nashi_steps
                select
                    to_timestamp(? / 1000.0) as timestamp,
                    ?::bigint as t,
                    ?::varchar as symbol,
                    ?::double as price,
                    ?::integer as action,
                    ?::boolean as hold,
                    ?::double as exposure,
                    ?::double as pnl,
                    ?::double as fill,
                    ?::double as fee,
                    ?::double as expected_surplus,
                    ?::double as realized_surplus,
                    ?::varchar as mw_reason,
                    ?::varchar as nashi_status,
                    ?::boolean as hazard_active,
                    ?::varchar as hazard_source,
                    ?::varchar as hazard_tightened_source,
                    ?::varchar as hazard_name,
                    ?::varchar as hazard_reason
                """,
                [
                    parity_row["ts"],
                    parity_row["t"],
                    parity_row["symbol"],
                    parity_row["price"],
                    parity_row["action"],
                    bool(parity_row["hold"]),
                    parity_row["exposure"],
                    parity_row["pnl"],
                    parity_row["fill"],
                    0.0,
                    parity_row["expected_surplus"],
                    parity_row["realized_surplus"],
                    parity_row["mw_reason"],
                    parity_row["nashi_status"],
                    bool(parity_row["hazard_active"]),
                    parity_row["hazard_source"],
                    parity_row["hazard_tightened_source"],
                    parity_row["hazard_name"],
                    parity_row["hazard_reason"],
                ],
            )
            con.execute("delete from nashi_family_certifications")
            con.execute(
                """
                insert into nashi_family_certifications
                select
                    to_timestamp(? / 1000.0) as timestamp,
                    ?::varchar as symbol,
                    ?::varchar as family_class,
                    ?::varchar as family_constructor,
                    ?::boolean as certified,
                    ?::boolean as trade_certified,
                    ?::boolean as preserve_certified,
                    ?::boolean as tail_localized,
                    ?::boolean as spread_dominated,
                    ?::boolean as hostile_regime,
                    ?::double as arrow_boundary_share,
                    ?::double as microstructure_kill_share,
                    ?::bigint as window_size,
                    ?::varchar as reasons,
                    ?::varchar as source_file
                """,
                [
                    parity_row["ts"],
                    parity_row["symbol"],
                    "interior_family",
                    "InteriorFamily",
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    0.0,
                    1.0,
                    1,
                    "microstructure_kills_edge",
                    "test",
                ],
            )
            con.close()
            parity = check_artifact_parity(
                [parity_row],
                family_rows=load_duckdb_family_rows(duckdb_path),
            )
            parity_summary = summarize_parity_violations(parity)
            assert parity_summary["csv_family_value_mismatch"] == 1

    print("test_nashi_invariants=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

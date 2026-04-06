from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from trading_io.logs import emit_step_row
except ModuleNotFoundError:  # pragma: no cover
    from trading.trading_io.logs import emit_step_row

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    duckdb = None


class NashiTelemetry:
    """
    Emit Nashi outputs into the repo's existing visualization surfaces.

    CSV step logs keep `training_dashboard*.py` compatibility.
    Decision/OHLC NDJSON plus DuckDB tables keep the stream dashboard path usable.
    """

    def __init__(
        self,
        *,
        step_log_path: Path | None,
        decision_ndjson_path: Path | None,
        ohlc_ndjson_path: Path | None,
        duckdb_path: Path | None,
        family_csv_path: Path | None,
        family_ndjson_path: Path | None,
        source_label: str,
        reset: bool = True,
    ) -> None:
        self.step_log_path = step_log_path
        self.decision_ndjson_path = decision_ndjson_path
        self.ohlc_ndjson_path = ohlc_ndjson_path
        self.duckdb_path = duckdb_path
        self.family_csv_path = family_csv_path
        self.family_ndjson_path = family_ndjson_path
        self.source_label = source_label
        self.duckdb_enabled = bool(duckdb_path is not None and duckdb is not None)

        for path in (
            step_log_path,
            decision_ndjson_path,
            ohlc_ndjson_path,
            duckdb_path,
            family_csv_path,
            family_ndjson_path,
        ):
            if path is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if reset:
                path.unlink(missing_ok=True)

        self.con = duckdb.connect(str(duckdb_path)) if self.duckdb_enabled else None
        if self.con is not None:
            self._ensure_tables()

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None

    def emit(
        self,
        *,
        step_row: dict[str, Any],
        decision_row: dict[str, Any],
        ohlc_row: dict[str, Any],
        family_row: dict[str, Any] | None = None,
    ) -> None:
        if self.step_log_path is not None:
            emit_step_row(step_row, self.step_log_path)
        if self.decision_ndjson_path is not None:
            self._append_ndjson(self.decision_ndjson_path, decision_row)
        if self.ohlc_ndjson_path is not None:
            self._append_ndjson(self.ohlc_ndjson_path, ohlc_row)
        if family_row is not None:
            family_artifact_row = self._family_artifact_row(family_row)
            if self.family_csv_path is not None:
                emit_step_row(family_artifact_row, self.family_csv_path)
            if self.family_ndjson_path is not None:
                self._append_ndjson(self.family_ndjson_path, family_artifact_row)
        if self.con is not None:
            self._append_duckdb(step_row, decision_row, ohlc_row, family_row)

    def _ensure_tables(self) -> None:
        assert self.con is not None
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS stream_actions (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                state INTEGER,
                direction INTEGER,
                target_exposure DOUBLE,
                urgency DOUBLE,
                hold BOOLEAN,
                actionability DOUBLE,
                reason VARCHAR,
                gate_open BOOLEAN,
                posture INTEGER,
                phase6_gate VARCHAR,
                source_file VARCHAR
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlc_1s (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                trades BIGINT,
                source_file VARCHAR
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS nashi_steps (
                timestamp TIMESTAMP,
                t BIGINT,
                symbol VARCHAR,
                price DOUBLE,
                bid DOUBLE,
                ask DOUBLE,
                spread DOUBLE,
                spread_bps DOUBLE,
                volume DOUBLE,
                state INTEGER,
                action INTEGER,
                hold BOOLEAN,
                exposure DOUBLE,
                pnl DOUBLE,
                fill DOUBLE,
                fill_price DOUBLE,
                fee DOUBLE,
                slippage DOUBLE,
                actionability DOUBLE,
                edge DOUBLE,
                stress DOUBLE,
                microstructure_pressure DOUBLE,
                cost_survival_ratio DOUBLE,
                drawdown DOUBLE,
                acceptable BOOLEAN,
                nashi_status VARCHAR,
                nashi_refusal VARCHAR,
                nashi_reasons VARCHAR,
                nashi_candidate_id VARCHAR,
                nashi_candidate_reason VARCHAR,
                nashi_spread_regime VARCHAR,
                nashi_rejected_candidates VARCHAR,
                nashi_family_class VARCHAR,
                nashi_family_constructor VARCHAR,
                nashi_family_certified BOOLEAN,
                nashi_family_trade_certified BOOLEAN,
                nashi_family_preserve_certified BOOLEAN,
                nashi_family_tail_localized BOOLEAN,
                nashi_family_spread_dominated BOOLEAN,
                nashi_family_hostile_regime BOOLEAN,
                nashi_family_arrow_boundary_share DOUBLE,
                nashi_family_microstructure_kill_share DOUBLE,
                nashi_family_reasons VARCHAR,
                nashi_q_delta DOUBLE,
                nashi_eigen_overlap DOUBLE,
                nashi_mdl_prev DOUBLE,
                nashi_mdl_next DOUBLE,
                capital_C DOUBLE,
                capital_dd DOUBLE,
                kappa_t DOUBLE,
                mw_reason VARCHAR,
                mw_refusal_level VARCHAR,
                mw_forced_hold BOOLEAN,
                mw_forced_ban BOOLEAN,
                mw_max_exposure DOUBLE,
                proposed_executable_opportunity BOOLEAN,
                proposed_governance_viable BOOLEAN,
                proposed_cost_viable BOOLEAN,
                proposed_viability_reason VARCHAR,
                proposed_expected_surplus DOUBLE,
                proposed_expected_gross_surplus DOUBLE,
                proposed_expected_cost DOUBLE,
                proposed_cost_survival_ratio DOUBLE,
                expected_surplus DOUBLE,
                expected_gross_surplus DOUBLE,
                expected_cost DOUBLE,
                phase9_cost_survival_ratio DOUBLE,
                executed_expected_surplus DOUBLE,
                executed_expected_gross_surplus DOUBLE,
                executed_expected_cost DOUBLE,
                executed_cost_survival_ratio DOUBLE,
                executable_opportunity BOOLEAN,
                realized_efficiency DOUBLE,
                aligned_expected_surplus DOUBLE,
                aligned_realized_surplus DOUBLE,
                aligned_realized_efficiency DOUBLE,
                execution_cost_realized DOUBLE,
                execution_cost_gap DOUBLE,
                execution_fill_ratio DOUBLE,
                phase9_microstructure_kills_edge BOOLEAN,
                hazard DOUBLE,
                hazard_regime VARCHAR,
                hazard_p_bad DOUBLE,
                hazard_bad_flag BOOLEAN,
                hazard_contextual_pressure DOUBLE,
                hazard_contextual_active BOOLEAN,
                hazard_contextual_label VARCHAR,
                hazard_ema DOUBLE,
                hazard_persistence DOUBLE,
                hazard_trend DOUBLE,
                hazard_cooldown DOUBLE,
                hazard_active BOOLEAN,
                hazard_source VARCHAR,
                hazard_tightened_source VARCHAR,
                hazard_name VARCHAR,
                hazard_reason VARCHAR,
                hazard_forced_hold BOOLEAN,
                hazard_forced_ban BOOLEAN,
                phase9_cfg_hazard_clamp_threshold DOUBLE,
                phase9_cfg_hazard_hold_threshold DOUBLE,
                phase9_cfg_hazard_ban_threshold DOUBLE,
                phase9_cfg_hazard_survival_floor_add DOUBLE,
                phase9_cfg_hazard_exposure_tightening DOUBLE,
                phase9_cfg_hazard_min_exposure_scale DOUBLE,
                phase9_cfg_min_expected_surplus DOUBLE,
                phase9_cfg_min_actionability DOUBLE,
                phase9_cfg_min_edge DOUBLE,
                phase9_cfg_microstructure_floor DOUBLE,
                phase9_cfg_microstructure_floor_min DOUBLE,
                phase9_cfg_microstructure_relief DOUBLE,
                phase9_cfg_microstructure_min_turnover DOUBLE,
                phase9_cfg_microstructure_min_gross DOUBLE,
                realized_surplus DOUBLE,
                justification_chain VARCHAR,
                just_regime VARCHAR,
                just_posture VARCHAR,
                just_actuator VARCHAR,
                just_cost_model VARCHAR,
                source_file VARCHAR
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS nashi_family_certifications (
                t BIGINT,
                timestamp TIMESTAMP,
                symbol VARCHAR,
                family_class VARCHAR,
                family_constructor VARCHAR,
                certified BOOLEAN,
                trade_certified BOOLEAN,
                preserve_certified BOOLEAN,
                tail_localized BOOLEAN,
                spread_dominated BOOLEAN,
                hostile_regime BOOLEAN,
                arrow_boundary_share DOUBLE,
                microstructure_kill_share DOUBLE,
                window_size BIGINT,
                reasons VARCHAR,
                source_file VARCHAR
            )
            """
        )
        self.con.execute(
            """
            CREATE OR REPLACE VIEW stream_actions_latest AS
            SELECT timestamp, symbol, state, direction, target_exposure, urgency, hold,
                   actionability, reason, gate_open, posture, phase6_gate, source_file
            FROM (
                SELECT timestamp, symbol, state, direction, target_exposure, urgency, hold,
                       actionability, reason, gate_open, posture, phase6_gate, source_file,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM stream_actions
            )
            WHERE rn = 1
            """
        )

    @staticmethod
    def _append_ndjson(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=True, allow_nan=False))
            fh.write("\n")

    def _family_artifact_row(self, family_row: dict[str, Any]) -> dict[str, Any]:
        row = dict(family_row)
        row["source_file"] = self.source_label
        return row

    def _append_duckdb(
        self,
        step_row: dict[str, Any],
        decision_row: dict[str, Any],
        ohlc_row: dict[str, Any],
        family_row: dict[str, Any] | None,
    ) -> None:
        assert self.con is not None

        decision_df = pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(decision_row["timestamp"], unit="ms", utc=True),
                    "symbol": decision_row["symbol"],
                    "state": int(decision_row["state"]),
                    "direction": int(decision_row["direction"]),
                    "target_exposure": float(decision_row["target_exposure"]),
                    "urgency": float(decision_row["urgency"]),
                    "hold": bool(decision_row["hold"]),
                    "actionability": float(decision_row["actionability"]),
                    "reason": decision_row["reason"],
                    "gate_open": True,
                    "posture": 1,
                    "phase6_gate": "",
                    "source_file": self.source_label,
                }
            ]
        )
        self.con.register("nashi_decision_df", decision_df)
        self.con.execute(
            """
            INSERT INTO stream_actions
            SELECT timestamp, symbol, state, direction, target_exposure, urgency, hold,
                   actionability, reason, gate_open, posture, phase6_gate, source_file
            FROM nashi_decision_df
            """
        )

        ohlc_df = pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(ohlc_row["timestamp"], unit="ms", utc=True),
                    "symbol": ohlc_row["symbol"],
                    "open": float(ohlc_row["open"]),
                    "high": float(ohlc_row["high"]),
                    "low": float(ohlc_row["low"]),
                    "close": float(ohlc_row["close"]),
                    "volume": float(ohlc_row["volume"]),
                    "trades": int(ohlc_row["trades"]),
                    "source_file": self.source_label,
                }
            ]
        )
        self.con.register("nashi_ohlc_df", ohlc_df)
        self.con.execute(
            """
            INSERT INTO ohlc_1s
            SELECT timestamp, symbol, open, high, low, close, volume, trades, source_file
            FROM nashi_ohlc_df
            """
        )

        if family_row is not None:
            family_df = pd.DataFrame(
                [
                    {
                        "t": int(family_row["t"]),
                        "timestamp": pd.to_datetime(family_row["timestamp"], unit="ms", utc=True),
                        "symbol": family_row["symbol"],
                        "family_class": family_row["family_class"],
                        "family_constructor": family_row["family_constructor"],
                        "certified": bool(family_row["certified"]),
                        "trade_certified": bool(family_row["trade_certified"]),
                        "preserve_certified": bool(family_row["preserve_certified"]),
                        "tail_localized": bool(family_row["tail_localized"]),
                        "spread_dominated": bool(family_row["spread_dominated"]),
                        "hostile_regime": bool(family_row["hostile_regime"]),
                        "arrow_boundary_share": float(family_row["arrow_boundary_share"]),
                        "microstructure_kill_share": float(family_row["microstructure_kill_share"]),
                        "window_size": int(family_row["window_size"]),
                        "reasons": family_row["reasons"],
                        "source_file": self.source_label,
                    }
                ]
            )
            self.con.register("nashi_family_df", family_df)
            self.con.execute(
                """
                INSERT INTO nashi_family_certifications
                SELECT t, timestamp, symbol, family_class, family_constructor, certified,
                       trade_certified, preserve_certified, tail_localized, spread_dominated,
                       hostile_regime, arrow_boundary_share, microstructure_kill_share,
                       window_size, reasons, source_file
                FROM nashi_family_df
                """
            )

        step_df = pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(step_row["ts"], unit="ms", utc=True),
                    "t": int(step_row["t"]),
                    "symbol": step_row["symbol"],
                    "price": float(step_row["price"]),
                    "bid": float(step_row["bid"]),
                    "ask": float(step_row["ask"]),
                    "spread": float(step_row["spread"]),
                    "spread_bps": float(step_row["spread_bps"]),
                    "volume": float(step_row["volume"]),
                    "state": int(step_row["state"]),
                    "action": int(step_row["action"]),
                    "hold": bool(step_row["hold"]),
                    "exposure": float(step_row["exposure"]),
                    "pnl": float(step_row["pnl"]),
                    "fill": float(step_row["fill"]),
                    "fill_price": float(step_row["fill_price"]),
                    "fee": float(step_row["fee"]),
                    "slippage": float(step_row["slippage"]),
                    "actionability": float(step_row["actionability"]),
                    "edge": float(step_row["edge"]),
                    "stress": float(step_row["stress"]),
                    "microstructure_pressure": float(step_row["microstructure_pressure"]),
                    "cost_survival_ratio": float(step_row["cost_survival_ratio"]),
                    "drawdown": float(step_row["drawdown"]),
                    "acceptable": bool(step_row["acceptable"]),
                    "nashi_status": step_row["nashi_status"],
                    "nashi_refusal": step_row["nashi_refusal"],
                    "nashi_reasons": step_row["nashi_reasons"],
                    "nashi_candidate_id": step_row["nashi_candidate_id"],
                    "nashi_candidate_reason": step_row["nashi_candidate_reason"],
                    "nashi_spread_regime": step_row["nashi_spread_regime"],
                    "nashi_rejected_candidates": step_row["nashi_rejected_candidates"],
                    "nashi_family_class": step_row["nashi_family_class"],
                    "nashi_family_constructor": step_row["nashi_family_constructor"],
                    "nashi_family_certified": bool(step_row["nashi_family_certified"]),
                    "nashi_family_trade_certified": bool(step_row["nashi_family_trade_certified"]),
                    "nashi_family_preserve_certified": bool(step_row["nashi_family_preserve_certified"]),
                    "nashi_family_tail_localized": bool(step_row["nashi_family_tail_localized"]),
                    "nashi_family_spread_dominated": bool(step_row["nashi_family_spread_dominated"]),
                    "nashi_family_hostile_regime": bool(step_row["nashi_family_hostile_regime"]),
                    "nashi_family_arrow_boundary_share": float(step_row["nashi_family_arrow_boundary_share"]),
                    "nashi_family_microstructure_kill_share": float(step_row["nashi_family_microstructure_kill_share"]),
                    "nashi_family_reasons": step_row["nashi_family_reasons"],
                    "nashi_q_delta": float(step_row["nashi_q_delta"]),
                    "nashi_eigen_overlap": float(step_row["nashi_eigen_overlap"]),
                    "nashi_mdl_prev": float(step_row["nashi_mdl_prev"]),
                    "nashi_mdl_next": float(step_row["nashi_mdl_next"]),
                    "capital_C": float(step_row["capital_C"]),
                    "capital_dd": float(step_row["capital_dd"]),
                    "kappa_t": float(step_row["kappa_t"]),
                    "mw_reason": step_row["mw_reason"],
                    "mw_refusal_level": step_row["mw_refusal_level"],
                    "mw_forced_hold": bool(step_row["mw_forced_hold"]),
                    "mw_forced_ban": bool(step_row["mw_forced_ban"]),
                    "mw_max_exposure": float(step_row["mw_max_exposure"]),
                    "proposed_executable_opportunity": bool(step_row["proposed_executable_opportunity"]),
                    "proposed_governance_viable": bool(step_row["proposed_governance_viable"]),
                    "proposed_cost_viable": bool(step_row["proposed_cost_viable"]),
                    "proposed_viability_reason": step_row["proposed_viability_reason"],
                    "proposed_expected_surplus": float(step_row["proposed_expected_surplus"]),
                    "proposed_expected_gross_surplus": float(step_row["proposed_expected_gross_surplus"]),
                    "proposed_expected_cost": float(step_row["proposed_expected_cost"]),
                    "proposed_cost_survival_ratio": float(step_row["proposed_cost_survival_ratio"]),
                    "expected_surplus": float(step_row["expected_surplus"]),
                    "expected_gross_surplus": float(step_row["expected_gross_surplus"]),
                    "expected_cost": float(step_row["expected_cost"]),
                    "phase9_cost_survival_ratio": float(step_row["phase9_cost_survival_ratio"]),
                    "executed_expected_surplus": float(step_row["executed_expected_surplus"]),
                    "executed_expected_gross_surplus": float(step_row["executed_expected_gross_surplus"]),
                    "executed_expected_cost": float(step_row["executed_expected_cost"]),
                    "executed_cost_survival_ratio": float(step_row["executed_cost_survival_ratio"]),
                    "executable_opportunity": bool(step_row["executable_opportunity"]),
                    "realized_efficiency": float(step_row["realized_efficiency"]),
                    "aligned_expected_surplus": float(step_row.get("aligned_expected_surplus", step_row["expected_surplus"])),
                    "aligned_realized_surplus": float(step_row.get("aligned_realized_surplus", step_row["realized_surplus"])),
                    "aligned_realized_efficiency": float(step_row.get("aligned_realized_efficiency", step_row["realized_efficiency"])),
                    "execution_cost_realized": float(step_row.get("execution_cost_realized", step_row["fee"])),
                    "execution_cost_gap": float(step_row.get("execution_cost_gap", 0.0)),
                    "execution_fill_ratio": float(step_row.get("execution_fill_ratio", 0.0)),
                    "phase9_microstructure_kills_edge": bool(step_row["phase9_microstructure_kills_edge"]),
                    "hazard": float(step_row["hazard"]),
                    "hazard_regime": step_row["hazard_regime"],
                    "hazard_p_bad": float(step_row["hazard_p_bad"]),
                    "hazard_bad_flag": bool(step_row["hazard_bad_flag"]),
                    "hazard_contextual_pressure": float(step_row.get("hazard_contextual_pressure", 0.0)),
                    "hazard_contextual_active": bool(step_row.get("hazard_contextual_active", False)),
                    "hazard_contextual_label": step_row.get("hazard_contextual_label", ""),
                    "hazard_ema": float(step_row.get("hazard_ema", step_row["hazard"])),
                    "hazard_persistence": float(step_row.get("hazard_persistence", 0.0)),
                    "hazard_trend": float(step_row.get("hazard_trend", 0.0)),
                    "hazard_cooldown": float(step_row.get("hazard_cooldown", 0.0)),
                    "hazard_active": bool(step_row["hazard_active"]),
                    "hazard_source": step_row.get("hazard_source", "none"),
                    "hazard_tightened_source": step_row.get("hazard_tightened_source", "none"),
                    "hazard_name": step_row["hazard_name"],
                    "hazard_reason": step_row["hazard_reason"],
                    "hazard_forced_hold": bool(step_row["hazard_forced_hold"]),
                    "hazard_forced_ban": bool(step_row["hazard_forced_ban"]),
                    "phase9_cfg_hazard_clamp_threshold": float(step_row.get("phase9_cfg_hazard_clamp_threshold", 0.45)),
                    "phase9_cfg_hazard_hold_threshold": float(step_row.get("phase9_cfg_hazard_hold_threshold", 0.72)),
                    "phase9_cfg_hazard_ban_threshold": float(step_row.get("phase9_cfg_hazard_ban_threshold", 0.94)),
                    "phase9_cfg_hazard_survival_floor_add": float(step_row.get("phase9_cfg_hazard_survival_floor_add", 0.40)),
                    "phase9_cfg_hazard_exposure_tightening": float(step_row.get("phase9_cfg_hazard_exposure_tightening", 0.60)),
                    "phase9_cfg_hazard_min_exposure_scale": float(step_row.get("phase9_cfg_hazard_min_exposure_scale", 0.15)),
                    "phase9_cfg_min_expected_surplus": float(step_row["phase9_cfg_min_expected_surplus"]),
                    "phase9_cfg_min_actionability": float(step_row["phase9_cfg_min_actionability"]),
                    "phase9_cfg_min_edge": float(step_row["phase9_cfg_min_edge"]),
                    "phase9_cfg_microstructure_floor": float(step_row["phase9_cfg_microstructure_floor"]),
                    "phase9_cfg_microstructure_floor_min": float(step_row["phase9_cfg_microstructure_floor_min"]),
                    "phase9_cfg_microstructure_relief": float(step_row["phase9_cfg_microstructure_relief"]),
                    "phase9_cfg_microstructure_min_turnover": float(step_row["phase9_cfg_microstructure_min_turnover"]),
                    "phase9_cfg_microstructure_min_gross": float(step_row["phase9_cfg_microstructure_min_gross"]),
                    "realized_surplus": float(step_row["realized_surplus"]),
                    "justification_chain": step_row["justification_chain"],
                    "just_regime": step_row["just_regime"],
                    "just_posture": step_row["just_posture"],
                    "just_actuator": step_row["just_actuator"],
                    "just_cost_model": step_row["just_cost_model"],
                    "source_file": self.source_label,
                }
            ]
        )
        self.con.register("nashi_step_df", step_df)
        self.con.execute(
            """
            INSERT INTO nashi_steps
            SELECT timestamp, t, symbol, price, bid, ask, spread, spread_bps, volume, state, action, hold, exposure, pnl,
                   fill, fill_price, fee, slippage, actionability, edge, stress, microstructure_pressure, cost_survival_ratio, drawdown,
                   acceptable, nashi_status, nashi_refusal, nashi_reasons, nashi_candidate_id,
                   nashi_candidate_reason, nashi_spread_regime, nashi_rejected_candidates,
                   nashi_family_class, nashi_family_constructor, nashi_family_certified, nashi_family_trade_certified,
                   nashi_family_preserve_certified, nashi_family_tail_localized, nashi_family_spread_dominated,
                   nashi_family_hostile_regime, nashi_family_arrow_boundary_share, nashi_family_microstructure_kill_share,
                   nashi_family_reasons, nashi_q_delta,
                   nashi_eigen_overlap, nashi_mdl_prev, nashi_mdl_next, capital_C, capital_dd,
                   kappa_t, mw_reason, mw_refusal_level, mw_forced_hold, mw_forced_ban,
                   mw_max_exposure, proposed_executable_opportunity, proposed_governance_viable,
                   proposed_cost_viable, proposed_viability_reason,
                   proposed_expected_surplus, proposed_expected_gross_surplus,
                   proposed_expected_cost, proposed_cost_survival_ratio,
                   expected_surplus, expected_gross_surplus, expected_cost,
                   phase9_cost_survival_ratio, executed_expected_surplus,
                   executed_expected_gross_surplus, executed_expected_cost,
                   executed_cost_survival_ratio, executable_opportunity, realized_efficiency,
                   aligned_expected_surplus, aligned_realized_surplus, aligned_realized_efficiency,
                   execution_cost_realized, execution_cost_gap, execution_fill_ratio,
                   phase9_microstructure_kills_edge,
                   hazard, hazard_regime, hazard_p_bad, hazard_bad_flag,
                   hazard_contextual_pressure, hazard_contextual_active, hazard_contextual_label,
                   hazard_ema, hazard_persistence, hazard_trend, hazard_cooldown, hazard_active,
                   hazard_source, hazard_tightened_source, hazard_name, hazard_reason, hazard_forced_hold, hazard_forced_ban,
                   phase9_cfg_hazard_clamp_threshold, phase9_cfg_hazard_hold_threshold, phase9_cfg_hazard_ban_threshold,
                   phase9_cfg_hazard_survival_floor_add, phase9_cfg_hazard_exposure_tightening, phase9_cfg_hazard_min_exposure_scale,
                   phase9_cfg_min_expected_surplus, phase9_cfg_min_actionability, phase9_cfg_min_edge,
                   phase9_cfg_microstructure_floor, phase9_cfg_microstructure_floor_min, phase9_cfg_microstructure_relief,
                   phase9_cfg_microstructure_min_turnover, phase9_cfg_microstructure_min_gross,
                   realized_surplus, justification_chain,
                   just_regime, just_posture, just_actuator, just_cost_model, source_file
            FROM nashi_step_df
            """
        )

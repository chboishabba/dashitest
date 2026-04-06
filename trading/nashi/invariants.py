from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .phase9 import CapitalParams, microstructure_survival_floor as phase9_microstructure_survival_floor

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover
    duckdb = None


EPS = 1e-9


@dataclass(frozen=True)
class InvariantViolation:
    rule: str
    row_index: int
    ts: int | None
    symbol: str
    detail: str


@dataclass(frozen=True)
class ArtifactParityViolation:
    rule: str
    detail: str


def _as_bool(row: Mapping[str, object], key: str) -> bool:
    value = row.get(key)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _as_float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(row: Mapping[str, object], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _split_reasons(value: str) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split("|") if part.strip()}


def load_step_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def load_ndjson_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_duckdb_step_rows(path: Path, *, table: str = "nashi_steps") -> list[dict[str, object]]:
    if duckdb is None:
        raise RuntimeError("duckdb is not installed in this environment")
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("show tables").fetchall()}
        if table not in tables:
            raise ValueError(f"table '{table}' not found in {path}")
        frame = con.execute(f"select * from {table} order by t, symbol").fetchdf()
        if "timestamp" in frame.columns and "ts" not in frame.columns:
            frame["ts"] = [
                int(pd.Timestamp(value).value // 1_000_000) if pd.notna(value) else None
                for value in frame["timestamp"]
            ]
        return frame.to_dict(orient="records")
    finally:
        con.close()


def duckdb_has_table(path: Path, *, table: str) -> bool:
    if duckdb is None:
        raise RuntimeError("duckdb is not installed in this environment")
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("show tables").fetchall()}
        return table in tables
    finally:
        con.close()


def load_duckdb_family_rows(
    path: Path,
    *,
    table: str = "nashi_family_certifications",
) -> list[dict[str, object]]:
    if duckdb is None:
        raise RuntimeError("duckdb is not installed in this environment")
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("show tables").fetchall()}
        if table not in tables:
            raise ValueError(f"table '{table}' not found in {path}")
        frame = con.execute(f"select * from {table} order by timestamp, symbol").fetchdf()
        if "timestamp" in frame.columns and "ts" not in frame.columns:
            frame["ts"] = [
                int(pd.Timestamp(value).value // 1_000_000) if pd.notna(value) else None
                for value in frame["timestamp"]
            ]
        return frame.to_dict(orient="records")
    finally:
        con.close()


def summarize_violations(violations: Iterable[InvariantViolation]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for violation in violations:
        summary[violation.rule] = summary.get(violation.rule, 0) + 1
    return dict(sorted(summary.items()))


def summarize_parity_violations(violations: Iterable[ArtifactParityViolation]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for violation in violations:
        summary[violation.rule] = summary.get(violation.rule, 0) + 1
    return dict(sorted(summary.items()))


def _build_index(rows: Iterable[Mapping[str, object]]) -> dict[tuple[int | None, str], Mapping[str, object]]:
    index: dict[tuple[int | None, str], Mapping[str, object]] = {}
    for offset, row in enumerate(rows):
        key = (
            _as_int(row, "t", default=offset),
            _as_text(row, "symbol"),
        )
        index[key] = row
    return index


def _build_family_index(rows: Iterable[Mapping[str, object]]) -> dict[tuple[int | None, str], Mapping[str, object]]:
    index: dict[tuple[int | None, str], Mapping[str, object]] = {}
    for offset, row in enumerate(rows):
        has_t = row.get("t") not in (None, "")
        key = (
            _as_int(row, "t", default=offset) if has_t else offset,
            _as_text(row, "symbol"),
        )
        index[key] = row
    return index


def check_artifact_parity(
    csv_rows: list[Mapping[str, object]],
    *,
    duckdb_rows: list[Mapping[str, object]] | None = None,
    family_rows: list[Mapping[str, object]] | None = None,
    decision_rows: list[Mapping[str, object]] | None = None,
    ohlc_rows: list[Mapping[str, object]] | None = None,
) -> list[ArtifactParityViolation]:
    violations: list[ArtifactParityViolation] = []

    if duckdb_rows is not None:
        if len(csv_rows) != len(duckdb_rows):
            violations.append(
                ArtifactParityViolation(
                    "csv_duckdb_row_count_mismatch",
                    f"csv_rows={len(csv_rows)} duckdb_rows={len(duckdb_rows)}",
                )
            )
        csv_index = _build_index(csv_rows)
        duckdb_index = _build_index(duckdb_rows)
        if set(csv_index) != set(duckdb_index):
            missing_in_duckdb = sorted(set(csv_index) - set(duckdb_index))[:3]
            missing_in_csv = sorted(set(duckdb_index) - set(csv_index))[:3]
            violations.append(
                ArtifactParityViolation(
                    "csv_duckdb_key_mismatch",
                    f"missing_in_duckdb={missing_in_duckdb} missing_in_csv={missing_in_csv}",
                )
            )
        compare_fields = (
            "price",
            "action",
            "hold",
            "exposure",
            "pnl",
            "fill",
            "fee",
            "expected_surplus",
            "realized_surplus",
            "mw_reason",
            "nashi_status",
            "hazard_active",
            "hazard_source",
            "hazard_tightened_source",
            "hazard_name",
            "hazard_reason",
            "hazard_forced_hold",
            "hazard_forced_ban",
        )
        for key in sorted(set(csv_index) & set(duckdb_index)):
            left = csv_index[key]
            right = duckdb_index[key]
            for field in compare_fields:
                if field not in left or field not in right:
                    continue
                l_text = _as_text(left, field)
                r_text = _as_text(right, field)
                if field in {
                    "mw_reason",
                    "nashi_status",
                    "hazard_source",
                    "hazard_tightened_source",
                    "hazard_name",
                    "hazard_reason",
                }:
                    if l_text != r_text:
                        violations.append(
                            ArtifactParityViolation(
                                "csv_duckdb_value_mismatch",
                                f"key={key} field={field} csv={l_text!r} duckdb={r_text!r}",
                            )
                        )
                        return violations
                else:
                    if abs(_as_float(left, field) - _as_float(right, field)) > 1e-6:
                        violations.append(
                            ArtifactParityViolation(
                                "csv_duckdb_value_mismatch",
                                f"key={key} field={field} csv={_as_float(left, field)} duckdb={_as_float(right, field)}",
                            )
                        )
                        return violations

    if family_rows is not None:
        if len(csv_rows) != len(family_rows):
            violations.append(
                ArtifactParityViolation(
                    "csv_family_row_count_mismatch",
                    f"csv_rows={len(csv_rows)} family_rows={len(family_rows)}",
                )
            )
        csv_family_index = _build_index(csv_rows)
        family_index = _build_family_index(family_rows)
        if set(csv_family_index) != set(family_index):
            missing_in_family = sorted(set(csv_family_index) - set(family_index))[:3]
            missing_in_csv = sorted(set(family_index) - set(csv_family_index))[:3]
            violations.append(
                ArtifactParityViolation(
                    "csv_family_key_mismatch",
                    f"missing_in_family={missing_in_family} missing_in_csv={missing_in_csv}",
                )
            )
            return violations
        field_pairs = (
            ("nashi_family_class", "family_class"),
            ("nashi_family_constructor", "family_constructor"),
            ("nashi_family_certified", "certified"),
            ("nashi_family_trade_certified", "trade_certified"),
            ("nashi_family_preserve_certified", "preserve_certified"),
            ("nashi_family_tail_localized", "tail_localized"),
            ("nashi_family_spread_dominated", "spread_dominated"),
            ("nashi_family_hostile_regime", "hostile_regime"),
            ("nashi_family_arrow_boundary_share", "arrow_boundary_share"),
            ("nashi_family_microstructure_kill_share", "microstructure_kill_share"),
            ("nashi_family_reasons", "reasons"),
        )
        for idx, key in enumerate(sorted(csv_family_index)):
            left = csv_family_index[key]
            right = family_index[key]
            for csv_field, family_field in field_pairs:
                if csv_field not in left or family_field not in right:
                    continue
                if csv_field in {
                    "nashi_family_class",
                    "nashi_family_constructor",
                    "nashi_family_reasons",
                }:
                    l_text = _as_text(left, csv_field)
                    r_text = _as_text(right, family_field)
                    if l_text != r_text:
                        violations.append(
                            ArtifactParityViolation(
                                "csv_family_value_mismatch",
                                f"row={idx} field={csv_field}->{family_field} csv={l_text!r} family={r_text!r}",
                            )
                        )
                        return violations
                elif csv_field in {
                    "nashi_family_arrow_boundary_share",
                    "nashi_family_microstructure_kill_share",
                }:
                    if abs(_as_float(left, csv_field) - _as_float(right, family_field)) > 1e-6:
                        violations.append(
                            ArtifactParityViolation(
                                "csv_family_value_mismatch",
                                f"row={idx} field={csv_field}->{family_field} csv={_as_float(left, csv_field)} family={_as_float(right, family_field)}",
                            )
                        )
                        return violations
                else:
                    if _as_bool(left, csv_field) != _as_bool(right, family_field):
                        violations.append(
                            ArtifactParityViolation(
                                "csv_family_value_mismatch",
                                f"row={idx} field={csv_field}->{family_field} csv={_as_bool(left, csv_field)} family={_as_bool(right, family_field)}",
                            )
                        )
                        return violations

    if decision_rows is not None and len(decision_rows) != len(csv_rows):
        violations.append(
            ArtifactParityViolation(
                "csv_decision_row_count_mismatch",
                f"csv_rows={len(csv_rows)} decision_rows={len(decision_rows)}",
            )
        )

    if ohlc_rows is not None and len(ohlc_rows) != len(csv_rows):
        violations.append(
            ArtifactParityViolation(
                "csv_ohlc_row_count_mismatch",
                f"csv_rows={len(csv_rows)} ohlc_rows={len(ohlc_rows)}",
            )
        )

    if decision_rows is not None:
        for idx, (csv_row, decision_row) in enumerate(zip(csv_rows, decision_rows)):
            if _as_int(csv_row, "ts", default=-1) != _as_int(decision_row, "timestamp", default=-1):
                violations.append(
                    ArtifactParityViolation(
                        "csv_decision_timestamp_mismatch",
                        f"row={idx} csv_ts={csv_row.get('ts')} decision_ts={decision_row.get('timestamp')}",
                    )
                )
                break
            if _as_text(csv_row, "symbol") != _as_text(decision_row, "symbol"):
                violations.append(
                    ArtifactParityViolation(
                        "csv_decision_symbol_mismatch",
                        f"row={idx} csv_symbol={csv_row.get('symbol')} decision_symbol={decision_row.get('symbol')}",
                    )
                )
                break
            if _as_int(csv_row, "action", default=0) != _as_int(decision_row, "direction", default=0):
                violations.append(
                    ArtifactParityViolation(
                        "csv_decision_direction_mismatch",
                        f"row={idx} csv_action={csv_row.get('action')} decision_direction={decision_row.get('direction')}",
                    )
                )
                break
            if "hazard_active" in csv_row and "hazard_active" in decision_row:
                if _as_bool(csv_row, "hazard_active") != _as_bool(decision_row, "hazard_active"):
                    violations.append(
                        ArtifactParityViolation(
                            "csv_decision_hazard_active_mismatch",
                            f"row={idx} csv_hazard_active={csv_row.get('hazard_active')} decision_hazard_active={decision_row.get('hazard_active')}",
                        )
                    )
                    break
            if "hazard_reason" in csv_row and "hazard_reason" in decision_row:
                if _as_text(csv_row, "hazard_reason") != _as_text(decision_row, "hazard_reason"):
                    violations.append(
                        ArtifactParityViolation(
                            "csv_decision_hazard_reason_mismatch",
                            f"row={idx} csv_hazard_reason={csv_row.get('hazard_reason')!r} decision_hazard_reason={decision_row.get('hazard_reason')!r}",
                        )
                    )
                    break

    if ohlc_rows is not None:
        for idx, (csv_row, ohlc_row) in enumerate(zip(csv_rows, ohlc_rows)):
            if _as_int(csv_row, "ts", default=-1) != _as_int(ohlc_row, "timestamp", default=-1):
                violations.append(
                    ArtifactParityViolation(
                        "csv_ohlc_timestamp_mismatch",
                        f"row={idx} csv_ts={csv_row.get('ts')} ohlc_ts={ohlc_row.get('timestamp')}",
                    )
                )
                break
            if abs(_as_float(csv_row, "price") - _as_float(ohlc_row, "close")) > 1e-6:
                violations.append(
                    ArtifactParityViolation(
                        "csv_ohlc_close_mismatch",
                        f"row={idx} csv_price={csv_row.get('price')} ohlc_close={ohlc_row.get('close')}",
                    )
                )
                break

    return violations


def check_step_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    microstructure_survival_floor: float | None = None,
) -> list[InvariantViolation]:
    base_params = CapitalParams()
    floor_override = (
        float(microstructure_survival_floor)
        if microstructure_survival_floor is not None
        else None
    )
    violations: list[InvariantViolation] = []

    for index, row in enumerate(rows):
        available = set(row.keys())
        ts_val = row.get("ts")
        try:
            ts = int(float(ts_val)) if ts_val is not None else None
        except (TypeError, ValueError):
            ts = None
        symbol = _as_text(row, "symbol")

        action = _as_int(row, "action")
        fill = _as_float(row, "fill")
        exposure = _as_float(row, "exposure")
        intent_direction = _as_int(row, "intent_direction")
        intent_target = _as_float(row, "intent_target")
        trade_active = abs(action) > 0 or abs(fill) > EPS

        justification_chain = _as_text(row, "justification_chain")
        just_regime = _as_text(row, "just_regime")
        just_posture = _as_text(row, "just_posture")
        just_actuator = _as_text(row, "just_actuator")
        just_cost_model = _as_text(row, "just_cost_model")

        family_certified = _as_bool(row, "nashi_family_certified")
        trade_certified = _as_bool(row, "nashi_family_trade_certified")
        preserve_certified = _as_bool(row, "nashi_family_preserve_certified")
        tail_localized = _as_bool(row, "nashi_family_tail_localized")
        spread_dominated = _as_bool(row, "nashi_family_spread_dominated")
        hostile_regime = _as_bool(row, "nashi_family_hostile_regime")
        family_reasons = _split_reasons(_as_text(row, "nashi_family_reasons"))
        hazard_active = _as_bool(row, "hazard_active")
        hazard_name = _as_text(row, "hazard_name")
        hazard_reason = _as_text(row, "hazard_reason")
        hazard_forced_hold = _as_bool(row, "hazard_forced_hold")
        hazard_forced_ban = _as_bool(row, "hazard_forced_ban")
        hazard_contextual_pressure = _as_float(row, "hazard_contextual_pressure")
        hazard_contextual_active = _as_bool(row, "hazard_contextual_active")
        hazard_contextual_label = _as_text(row, "hazard_contextual_label")
        hazard_source = _as_text(row, "hazard_source")
        hazard_tightened_source = _as_text(row, "hazard_tightened_source")

        mw_refusal_level = _as_text(row, "mw_refusal_level").lower()
        mw_forced_hold = _as_bool(row, "mw_forced_hold")
        mw_forced_ban = _as_bool(row, "mw_forced_ban")
        mw_max_exposure = _as_float(row, "mw_max_exposure")

        expected_surplus = _as_float(row, "expected_surplus")
        expected_gross = _as_float(row, "expected_gross_surplus")
        expected_cost = _as_float(row, "expected_cost")
        proposed_expected_surplus = _as_float(row, "proposed_expected_surplus", expected_surplus)
        proposed_expected_gross = _as_float(row, "proposed_expected_gross_surplus", expected_gross)
        proposed_expected_cost = _as_float(row, "proposed_expected_cost", expected_cost)
        executed_expected_surplus = _as_float(row, "executed_expected_surplus", expected_surplus)
        executed_expected_gross = _as_float(row, "executed_expected_gross_surplus", expected_gross)
        executed_expected_cost = _as_float(row, "executed_expected_cost", expected_cost)
        executable_opportunity = _as_bool(row, "executable_opportunity")
        realized_efficiency = _as_float(row, "realized_efficiency")
        realized_surplus = _as_float(row, "realized_surplus")
        aligned_expected_surplus = _as_float(row, "aligned_expected_surplus")
        aligned_realized_surplus = _as_float(row, "aligned_realized_surplus")
        aligned_realized_efficiency = _as_float(row, "aligned_realized_efficiency")
        cost_survival_ratio = _as_float(row, "phase9_cost_survival_ratio")
        micro_kills_edge = _as_bool(row, "phase9_microstructure_kills_edge")
        params = CapitalParams(
            microstructure_survival_floor=_as_float(
                row,
                "phase9_cfg_microstructure_floor",
                base_params.microstructure_survival_floor,
            ),
            microstructure_survival_floor_min=_as_float(
                row,
                "phase9_cfg_microstructure_floor_min",
                base_params.microstructure_survival_floor_min,
            ),
            microstructure_relief_strength=_as_float(
                row,
                "phase9_cfg_microstructure_relief",
                base_params.microstructure_relief_strength,
            ),
            microstructure_min_turnover=_as_float(
                row,
                "phase9_cfg_microstructure_min_turnover",
                base_params.microstructure_min_turnover,
            ),
            microstructure_min_expected_gross=_as_float(
                row,
                "phase9_cfg_microstructure_min_gross",
                base_params.microstructure_min_expected_gross,
            ),
        )
        effective_floor = (
            floor_override
            if floor_override is not None
            else phase9_microstructure_survival_floor(
                edge=_as_float(row, "edge"),
                actionability=_as_float(row, "actionability"),
                params=params,
            )
        )
        trade_attempt_active = (
            trade_active
            or abs(intent_target - exposure) > params.microstructure_min_turnover
        )
        gross_active = expected_gross > params.microstructure_min_expected_gross

        if trade_active and not (
            justification_chain and just_regime and just_posture and just_actuator and just_cost_model
        ):
            violations.append(
                InvariantViolation(
                    "trade_requires_justification",
                    index,
                    ts,
                    symbol,
                    "trade-active row is missing justification-chain fields",
                )
            )

        if trade_certified and preserve_certified:
            violations.append(
                InvariantViolation(
                    "family_dual_certification",
                    index,
                    ts,
                    symbol,
                    "trade_certified and preserve_certified cannot both be true",
                )
            )

        if spread_dominated and trade_certified:
            violations.append(
                InvariantViolation(
                    "spread_dominated_blocks_trade_certification",
                    index,
                    ts,
                    symbol,
                    "spread_dominated family cannot be trade_certified",
                )
            )

        ban_like = mw_forced_ban or mw_refusal_level in {"ban", "paradox"}
        if ban_like:
            if abs(exposure) > EPS or action != 0 or intent_direction != 0 or abs(intent_target) > EPS:
                violations.append(
                    InvariantViolation(
                        "ban_forces_flat_action",
                        index,
                        ts,
                        symbol,
                        "BAN-like refusal must leave zero action, zero target, and flat exposure",
                    )
                )
            if abs(mw_max_exposure) > EPS:
                violations.append(
                    InvariantViolation(
                        "ban_sets_zero_max_exposure",
                        index,
                        ts,
                        symbol,
                        "BAN-like refusal must clamp max exposure to zero",
                    )
                )

        if mw_forced_ban and not mw_forced_hold:
            violations.append(
                InvariantViolation(
                    "ban_implies_hold",
                    index,
                    ts,
                    symbol,
                    "forced BAN should also set forced HOLD",
                )
            )

        if not family_certified and (trade_certified or preserve_certified):
            violations.append(
                InvariantViolation(
                    "family_certification_flag_alignment",
                    index,
                    ts,
                    symbol,
                    "trade/preserve certification requires certified=true",
                )
            )

        hazard_fields_present = bool(
            {"hazard_active", "hazard_name", "hazard_reason", "hazard_forced_hold", "hazard_forced_ban"} & available
        )
        if hazard_fields_present and (hazard_forced_hold or hazard_forced_ban) and not hazard_active:
            violations.append(
                InvariantViolation(
                    "hazard_directive_requires_hazard",
                    index,
                    ts,
                    symbol,
                    "hazard-driven hold/ban requires hazard_active=true",
                )
            )

        if hazard_fields_present and hazard_active and not (hazard_name or hazard_reason):
            violations.append(
                InvariantViolation(
                    "hazard_requires_explanation",
                    index,
                    ts,
                    symbol,
                    "hazard_active rows must carry hazard_name or hazard_reason",
                )
            )

        contextual_fields_present = bool(
            {"hazard_contextual_pressure", "hazard_contextual_active", "hazard_contextual_label"} & available
        )
        if contextual_fields_present and hazard_contextual_active and hazard_contextual_pressure <= EPS:
            violations.append(
                InvariantViolation(
                    "hazard_contextual_active_requires_pressure",
                    index,
                    ts,
                    symbol,
                    "hazard_contextual_active requires hazard_contextual_pressure > 0",
                )
            )

        if contextual_fields_present and hazard_contextual_label and not hazard_contextual_active:
            violations.append(
                InvariantViolation(
                    "hazard_contextual_label_requires_active",
                    index,
                    ts,
                    symbol,
                    "hazard_contextual_label requires hazard_contextual_active=true",
                )
            )

        if contextual_fields_present and hazard_contextual_active and hazard_active and hazard_name and hazard_contextual_label and hazard_name != hazard_contextual_label:
            violations.append(
                InvariantViolation(
                    "hazard_contextual_name_alignment",
                    index,
                    ts,
                    symbol,
                    "contextual hazard rows should expose hazard_name equal to hazard_contextual_label",
                )
            )

        hazard_source_fields_present = bool({"hazard_source", "hazard_tightened_source"} & available)
        if hazard_source_fields_present and hazard_source:
            implied_hazard_source = "none"
            if hazard_contextual_active:
                implied_hazard_source = "contextual"
            elif hazard_active:
                implied_hazard_source = "synthetic_only"
            if hazard_source != implied_hazard_source:
                violations.append(
                    InvariantViolation(
                        "hazard_source_provenance_consistency",
                        index,
                        ts,
                        symbol,
                        "hazard_source must align with hazard_active and hazard_contextual_active",
                    )
                )

        if hazard_source_fields_present and hazard_tightened_source:
            implied_tightened_source = "none"
            if hazard_active:
                implied_tightened_source = "contextual" if hazard_contextual_active else "synthetic_only"
            if hazard_tightened_source != implied_tightened_source:
                violations.append(
                    InvariantViolation(
                        "hazard_tightened_source_consistency",
                        index,
                        ts,
                        symbol,
                        "hazard_tightened_source must align with hazard_active and hazard_contextual_active",
                    )
                )
            if hazard_tightened_source == "contextual" and not hazard_contextual_active:
                violations.append(
                    InvariantViolation(
                        "hazard_tightened_source_requires_contextual",
                        index,
                        ts,
                        symbol,
                        "contextual hazard_tightened_source requires hazard_contextual_active=true",
                    )
                )

        if hazard_forced_hold and not mw_forced_hold:
            violations.append(
                InvariantViolation(
                    "hazard_hold_aligns_meta_witness",
                    index,
                    ts,
                    symbol,
                    "hazard_forced_hold requires mw_forced_hold=true",
                )
            )

        if hazard_forced_ban and not mw_forced_ban:
            violations.append(
                InvariantViolation(
                    "hazard_ban_aligns_meta_witness",
                    index,
                    ts,
                    symbol,
                    "hazard_forced_ban requires mw_forced_ban=true",
                )
            )

        if hazard_active and hazard_forced_hold and not hazard_forced_ban and trade_active:
            violations.append(
                InvariantViolation(
                    "hazard_hold_blocks_trade",
                    index,
                    ts,
                    symbol,
                    "hazard preserve rows must not trade",
                )
            )

        if hazard_active and hazard_forced_hold and not hazard_forced_ban and trade_certified:
            violations.append(
                InvariantViolation(
                    "hazard_preserve_blocks_trade_certification",
                    index,
                    ts,
                    symbol,
                    "hazard preserve rows cannot be trade_certified",
                )
            )

        if hazard_active and hazard_forced_ban and not mw_forced_hold:
            violations.append(
                InvariantViolation(
                    "hazard_ban_implies_hold",
                    index,
                    ts,
                    symbol,
                    "hazard_forced_ban requires mw_forced_hold=true",
                )
            )

        if hazard_active and hazard_forced_ban:
            if abs(exposure) > EPS or action != 0 or intent_direction != 0 or abs(intent_target) > EPS:
                violations.append(
                    InvariantViolation(
                        "hazard_ban_forces_flat_action",
                        index,
                        ts,
                        symbol,
                        "hazard BAN rows must leave zero action, zero target, and flat exposure",
                    )
                )

        if spread_dominated and "microstructure_kills_edge" not in family_reasons:
            violations.append(
                InvariantViolation(
                    "family_reason_missing_microstructure",
                    index,
                    ts,
                    symbol,
                    "spread_dominated family must include microstructure_kills_edge reason",
                )
            )

        if hostile_regime and "hostile_regime" not in family_reasons:
            violations.append(
                InvariantViolation(
                    "family_reason_missing_hostile_regime",
                    index,
                    ts,
                    symbol,
                    "hostile_regime family must include hostile_regime reason",
                )
            )

        if "mdl_tail_boundary" in family_reasons and not tail_localized:
            violations.append(
                InvariantViolation(
                    "family_reason_missing_tail_localized",
                    index,
                    ts,
                    symbol,
                    "mdl_tail_boundary reason requires tail_localized=true",
                )
            )

        if {"expected_surplus", "expected_gross_surplus", "expected_cost"}.issubset(available) and abs(
            expected_surplus - (expected_gross - expected_cost)
        ) > 1e-6:
            violations.append(
                InvariantViolation(
                    "phase9_expected_surplus_arithmetic",
                    index,
                    ts,
                    symbol,
                    "expected_surplus must equal expected_gross_surplus - expected_cost",
                )
            )

        if {"proposed_expected_surplus", "proposed_expected_gross_surplus", "proposed_expected_cost"}.issubset(available) and abs(
            proposed_expected_surplus - (proposed_expected_gross - proposed_expected_cost)
        ) > 1e-6:
            violations.append(
                InvariantViolation(
                    "phase9_proposed_expected_surplus_arithmetic",
                    index,
                    ts,
                    symbol,
                    "proposed_expected_surplus must equal proposed_expected_gross_surplus - proposed_expected_cost",
                )
            )

        if {"executed_expected_surplus", "executed_expected_gross_surplus", "executed_expected_cost"}.issubset(available) and abs(
            executed_expected_surplus - (executed_expected_gross - executed_expected_cost)
        ) > 1e-6:
            violations.append(
                InvariantViolation(
                    "phase9_executed_expected_surplus_arithmetic",
                    index,
                    ts,
                    symbol,
                    "executed_expected_surplus must equal executed_expected_gross_surplus - executed_expected_cost",
                )
            )

        if {"executable_opportunity", "executed_expected_surplus"}.issubset(available):
            implied_opportunity = (
                executed_expected_surplus > EPS
                and not mw_forced_hold
                and not mw_forced_ban
            )
            if executable_opportunity != implied_opportunity:
                violations.append(
                    InvariantViolation(
                        "phase9_executable_opportunity_consistency",
                        index,
                        ts,
                        symbol,
                        "executable_opportunity must match executed_expected_surplus > 0",
                    )
                )

        if {"executable_opportunity", "mw_forced_hold", "mw_forced_ban"}.issubset(available):
            if executable_opportunity and (mw_forced_hold or mw_forced_ban):
                violations.append(
                    InvariantViolation(
                        "phase9_executable_opportunity_blocking_consistency",
                        index,
                        ts,
                        symbol,
                        "blocking directives cannot coexist with executable_opportunity=true",
                    )
                )

        if {"realized_efficiency", "executed_expected_surplus", "realized_surplus"}.issubset(available):
            if abs(executed_expected_surplus) > EPS:
                implied_efficiency = realized_surplus / executed_expected_surplus
            else:
                implied_efficiency = 0.0

            if abs(realized_efficiency - implied_efficiency) > 1e-6:
                violations.append(
                    InvariantViolation(
                        "phase9_realized_efficiency_consistency",
                        index,
                        ts,
                        symbol,
                        "realized_efficiency must equal realized_surplus / executed_expected_surplus",
                    )
                )

        if {"aligned_realized_efficiency", "aligned_expected_surplus", "aligned_realized_surplus"}.issubset(available):
            if abs(aligned_expected_surplus) > EPS:
                implied_aligned_efficiency = aligned_realized_surplus / aligned_expected_surplus
            else:
                implied_aligned_efficiency = 0.0

            if abs(aligned_realized_efficiency - implied_aligned_efficiency) > 1e-6:
                violations.append(
                    InvariantViolation(
                        "phase9_aligned_realized_efficiency_consistency",
                        index,
                        ts,
                        symbol,
                        "aligned_realized_efficiency must equal aligned_realized_surplus / aligned_expected_surplus",
                    )
                )

        if {"phase9_cost_survival_ratio", "phase9_microstructure_kills_edge"}.issubset(available) and micro_kills_edge and cost_survival_ratio >= effective_floor:
            violations.append(
                InvariantViolation(
                    "phase9_microstructure_flag_consistency",
                    index,
                    ts,
                    symbol,
                    f"microstructure kill flag requires cost_survival_ratio < {effective_floor}",
                )
            )
        if (
            {"phase9_cost_survival_ratio", "phase9_microstructure_kills_edge"}.issubset(available)
            and (not micro_kills_edge)
            and trade_attempt_active
            and gross_active
            and not mw_forced_hold
            and not mw_forced_ban
            and cost_survival_ratio < effective_floor
        ):
            violations.append(
                InvariantViolation(
                    "phase9_microstructure_flag_consistency",
                    index,
                    ts,
                    symbol,
                    f"active trade attempt with cost_survival_ratio < {effective_floor} requires microstructure kill flag",
                )
            )

    return violations

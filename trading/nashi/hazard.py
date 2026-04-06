from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from signals.stress import compute_structural_stress


def _clip_unit(values: pd.Series | np.ndarray | float) -> pd.Series | float:
    if isinstance(values, pd.Series):
        return values.clip(lower=0.0, upper=1.0)
    return float(max(0.0, min(1.0, float(values))))


def _normalize_optional_series(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    values = pd.Series(0.0, index=frame.index, dtype=float)
    for name in names:
        if name not in frame.columns:
            continue
        raw = pd.to_numeric(frame[name], errors="coerce").fillna(0.0).astype(float)
        if raw.empty:
            continue
        if raw.max() <= 1.0 and raw.min() >= 0.0:
            values = np.maximum(values, raw)
            continue
        scale = float(raw.abs().quantile(0.95)) if len(raw) > 1 else float(abs(raw.iloc[0]))
        scale = max(scale, 1e-9)
        values = np.maximum(values, (raw.abs() / scale).clip(lower=0.0, upper=1.0))
    return values


def _hazard_regime(score: float, *, bad_flag: bool, synthetic_bad: bool) -> str:
    if bad_flag or score >= 0.85:
        return "hostile"
    if synthetic_bad or score >= 0.60:
        return "elevated"
    if score >= 0.35:
        return "watch"
    return "calm"


@dataclass(frozen=True)
class HazardObservation:
    hazard_score: float
    hazard_regime: str
    hazard_p_bad: float
    hazard_bad_flag: bool
    hazard_synthetic_bad: bool
    hazard_drawdown_pressure: float
    hazard_vol_pressure: float
    hazard_spread_pressure: float
    hazard_contextual_pressure: float
    hazard_contextual_active: bool
    hazard_contextual_label: str
    hazard_ema: float
    hazard_persistence: float
    hazard_trend: float
    hazard_cooldown: float


def hazard_reentry_ready(observation: HazardObservation) -> bool:
    return bool(
        observation.hazard_score < 0.28
        and observation.hazard_ema < 0.24
        and observation.hazard_persistence < 0.18
        and observation.hazard_cooldown < 0.40
        and not observation.hazard_bad_flag
        and not observation.hazard_synthetic_bad
    )


def contextual_hazard_only(observation: HazardObservation) -> bool:
    return bool(
        observation.hazard_contextual_active
        and observation.hazard_contextual_pressure > 0.0
        and not observation.hazard_bad_flag
        and not observation.hazard_synthetic_bad
        and observation.hazard_drawdown_pressure < 0.35
        and observation.hazard_vol_pressure < 0.40
    )

def proposal_hazard_density(observation: HazardObservation) -> float:
    contextual = float(observation.hazard_contextual_pressure)
    if observation.hazard_ema < 0.30 and observation.hazard_persistence < 0.20:
        contextual = min(contextual, 0.45)
    density = (
        0.40 * float(observation.hazard_score)
        + 0.25 * float(observation.hazard_ema)
        + 0.10 * float(observation.hazard_persistence)
        + 0.10 * float(observation.hazard_cooldown)
        + 0.10 * contextual
        + 0.05 * float(observation.hazard_trend)
    )
    if contextual_hazard_only(observation):
        if contextual >= 0.85:
            density = max(density, 0.78)
        elif contextual >= 0.70:
            density = max(density, 0.56)
    if hazard_reentry_ready(observation):
        density *= 0.55
    elif contextual_hazard_only(observation) and contextual < 0.55 and observation.hazard_cooldown < 0.30:
        density *= 0.80
    return float(max(0.0, min(1.0, density)))


def governance_hazard_level(observation: HazardObservation) -> float:
    contextual = float(observation.hazard_contextual_pressure)
    corroborated = (
        observation.hazard_score >= 0.35
        or observation.hazard_ema >= 0.28
        or observation.hazard_persistence >= 0.20
        or observation.hazard_bad_flag
        or observation.hazard_synthetic_bad
    )
    contextual_term = contextual * (0.30 if corroborated else 0.12)
    smoothed = (
        0.45 * float(observation.hazard_score)
        + 0.25 * float(observation.hazard_ema)
        + 0.15 * float(observation.hazard_persistence)
        + 0.10 * float(observation.hazard_cooldown)
        + contextual_term
        + 0.05 * float(observation.hazard_trend)
    )
    level = max(float(observation.hazard_score), smoothed)
    if contextual_hazard_only(observation):
        if contextual >= 0.90 and (observation.hazard_ema >= 0.16 or observation.hazard_cooldown >= 0.12):
            level = max(level, 0.84)
        elif contextual >= 0.78 and (
            observation.hazard_ema >= 0.14
            or observation.hazard_cooldown >= 0.10
            or observation.hazard_persistence >= 0.08
        ):
            level = max(level, 0.52)
    if hazard_reentry_ready(observation):
        level = min(level * 0.50, 0.34)
    elif contextual_hazard_only(observation) and contextual < 0.55 and observation.hazard_cooldown < 0.25:
        level = min(level, 0.38)
    return float(max(0.0, min(1.0, level)))


def _normalize_series_unit(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    finite = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    scale = float(finite.abs().quantile(0.95)) if len(finite) > 1 else float(abs(finite.iloc[0]))
    scale = max(scale, 1e-9)
    return (finite / scale).clip(lower=0.0, upper=1.0)


def _cooldown_series(events: pd.Series, *, decay: float = 0.85) -> pd.Series:
    cooldown: list[float] = []
    level = 0.0
    for event in events.fillna(False).astype(bool).tolist():
        level = 1.0 if event else level * float(decay)
        cooldown.append(level)
    return pd.Series(cooldown, index=events.index, dtype=float).clip(lower=0.0, upper=1.0)


def _coerce_window_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.notna().any():
        return pd.Series((ts.astype("int64") // 1_000_000).astype("int64"), index=series.index)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")


def _normalize_pressure(values: pd.Series | float) -> pd.Series:
    if isinstance(values, (int, float)):
        values = pd.Series([float(values)])
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    if numeric.empty:
        return pd.Series(dtype=float)
    lo = float(numeric.min())
    hi = float(numeric.quantile(0.95))
    if hi <= lo + 1e-9:
        return pd.Series(np.where(numeric > lo, 1.0, 0.0), index=numeric.index, dtype=float)
    return ((numeric - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)


def load_contextual_hazard_windows(path: Path | str) -> pd.DataFrame:
    src = Path(path)
    frame = pd.read_csv(src)
    if frame.empty:
        return pd.DataFrame(columns=["start_ms", "end_ms", "contextual_hazard", "contextual_label", "contextual_source"])

    if {"start_ms", "end_ms"}.issubset(frame.columns):
        start_ms = pd.to_numeric(frame["start_ms"], errors="coerce").fillna(0).astype("int64")
        end_ms = pd.to_numeric(frame["end_ms"], errors="coerce").fillna(0).astype("int64")
    elif {"ts_start", "ts_end"}.issubset(frame.columns):
        start_ms = _coerce_window_ts(frame["ts_start"])
        end_ms = _coerce_window_ts(frame["ts_end"])
    elif {"start", "end"}.issubset(frame.columns):
        start_ms = _coerce_window_ts(frame["start"])
        end_ms = _coerce_window_ts(frame["end"])
    else:
        raise KeyError("contextual hazard CSV must include start/end or ts_start/ts_end columns")

    if "contextual_hazard" in frame.columns:
        contextual_hazard = pd.to_numeric(frame["contextual_hazard"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    elif "sev_sum_p_bad" in frame.columns:
        contextual_hazard = (
            0.50 * _normalize_pressure(frame["sev_sum_p_bad"])
            + 0.20 * _normalize_pressure(frame.get("p_bad_mean", 0.0))
            + 0.15 * _normalize_pressure(frame.get("p_bad_max", 0.0))
            + 0.10 * _normalize_pressure(frame.get("bad_rate", 0.0))
            + 0.05 * _normalize_pressure(frame.get("synthetic_bad_rate", 0.0))
        ).clip(lower=0.0, upper=1.0)
    elif "events" in frame.columns:
        negative_tone = (-pd.to_numeric(frame.get("mean_tone", 0.0), errors="coerce").fillna(0.0)).clip(lower=0.0)
        contextual_hazard = (
            0.55 * _normalize_pressure(frame["events"])
            + 0.25 * _normalize_pressure(frame.get("triggers", 0.0))
            + 0.20 * _normalize_pressure(negative_tone)
        ).clip(lower=0.0, upper=1.0)
    else:
        raise KeyError("contextual hazard CSV must include contextual_hazard, sev_sum_p_bad, or events")

    if "contextual_label" in frame.columns:
        label = frame["contextual_label"].fillna("").astype(str)
    elif "top_codes" in frame.columns:
        label = frame["top_codes"].fillna("").astype(str)
    elif "window_id" in frame.columns:
        label = "window_" + frame["window_id"].astype(str)
    else:
        label = pd.Series("contextual_window", index=frame.index, dtype=object)

    if "contextual_source" in frame.columns:
        source = frame["contextual_source"].fillna("").astype(str)
    elif "events" in frame.columns:
        source = pd.Series("news_events", index=frame.index, dtype=object)
    elif "sev_sum_p_bad" in frame.columns:
        source = pd.Series("bad_windows", index=frame.index, dtype=object)
    else:
        source = pd.Series(src.name, index=frame.index, dtype=object)

    return pd.DataFrame(
        {
            "start_ms": np.minimum(start_ms, end_ms),
            "end_ms": np.maximum(start_ms, end_ms),
            "contextual_hazard": contextual_hazard.astype(float),
            "contextual_label": label,
            "contextual_source": source,
        }
    ).sort_values(["start_ms", "end_ms"], kind="stable").reset_index(drop=True)


def attach_contextual_hazard_windows(
    frame: pd.DataFrame,
    windows: pd.DataFrame | Path | str | None,
) -> pd.DataFrame:
    enriched = frame.copy()
    if windows is None:
        return enriched
    window_frame = load_contextual_hazard_windows(windows) if not isinstance(windows, pd.DataFrame) else windows.copy()
    if window_frame.empty:
        return enriched

    ts = pd.to_numeric(enriched.get("ts", pd.Series(0, index=enriched.index)), errors="coerce").fillna(0).astype("int64")
    contextual = pd.Series(0.0, index=enriched.index, dtype=float)
    label = pd.Series("", index=enriched.index, dtype=object)
    source = pd.Series("", index=enriched.index, dtype=object)
    active = pd.Series(False, index=enriched.index, dtype=bool)

    for _, window in window_frame.iterrows():
        mask = (ts >= int(window["start_ms"])) & (ts <= int(window["end_ms"]))
        if not bool(mask.any()):
            continue
        pressure = float(window["contextual_hazard"])
        stronger = mask & (contextual <= pressure)
        contextual.loc[mask] = np.maximum(contextual.loc[mask], pressure)
        label.loc[stronger] = str(window.get("contextual_label", ""))
        source.loc[stronger] = str(window.get("contextual_source", ""))
        active.loc[mask] = True

    enriched["contextual_hazard"] = contextual.to_numpy(dtype=float)
    enriched["news_hazard"] = contextual.to_numpy(dtype=float)
    enriched["hazard_contextual_active"] = active.to_numpy(dtype=bool)
    enriched["hazard_contextual_label"] = label.fillna("").astype(str)
    enriched["hazard_contextual_source"] = source.fillna("").astype(str)
    return enriched


def attach_hazard_observables(
    frame: pd.DataFrame,
    *,
    stress_window: int = 100,
    sigma_window: int = 50,
    shock_sigma: float = 3.0,
    dd_slope_threshold: float = 0.001,
    contextual_windows: pd.DataFrame | Path | str | None = None,
) -> pd.DataFrame:
    enriched = attach_contextual_hazard_windows(frame, contextual_windows)
    prices = pd.to_numeric(enriched["close"], errors="coerce").ffill().bfill()
    states = pd.to_numeric(enriched.get("state", 0.0), errors="coerce").fillna(0.0)
    spreads = pd.to_numeric(enriched.get("spread", 0.0), errors="coerce").fillna(0.0)

    p_bad, bad_flag = compute_structural_stress(
        prices.to_numpy(dtype=float),
        states.to_numpy(dtype=float),
        window=max(10, int(stress_window)),
    )
    p_bad_series = pd.Series(p_bad, index=enriched.index, dtype=float).fillna(0.0).clip(lower=0.0, upper=1.0)
    bad_flag_series = pd.Series(bad_flag, index=enriched.index).fillna(False).astype(bool)

    returns = prices.pct_change().fillna(0.0)
    sigma = returns.rolling(max(5, int(sigma_window)), min_periods=5).std().bfill().fillna(0.0)
    shock = returns.abs() > (float(shock_sigma) * sigma)

    price_peak = prices.cummax().clip(lower=1e-9)
    price_drawdown = (1.0 - (prices / price_peak)).clip(lower=0.0)
    drawdown_slope = price_drawdown.diff().clip(lower=0.0).fillna(0.0)
    synthetic_bad = (shock | (drawdown_slope > float(dd_slope_threshold))).astype(bool)

    vol_ref = float(np.nanmedian(sigma[sigma > 0.0])) if np.any(sigma > 0.0) else 1e-6
    vol_ref = max(vol_ref, 1e-6)
    vol_pressure = _clip_unit((sigma / vol_ref - 1.0) / 2.0)

    spread_bps = 1e4 * spreads / prices.clip(lower=1e-9)
    spread_ref = spread_bps.rolling(max(5, int(sigma_window)), min_periods=5).median().bfill().fillna(0.0)
    spread_pressure = _clip_unit((spread_bps / spread_ref.clip(lower=1e-9) - 1.0) / 2.0)
    contextual_pressure = _normalize_optional_series(
        enriched,
        (
            "hazard",
            "p_bad",
            "bad_flag",
            "news_hazard",
            "contextual_hazard",
            "stress_proxy",
        ),
    )

    hazard_score = (
        0.45 * p_bad_series
        + 0.20 * synthetic_bad.astype(float)
        + 0.15 * _clip_unit(price_drawdown / 0.10)
        + 0.10 * vol_pressure.astype(float)
        + 0.05 * spread_pressure.astype(float)
        + 0.05 * contextual_pressure.astype(float)
    ).clip(lower=0.0, upper=1.0)
    fast_span = max(3, int(max(5, sigma_window) // 3))
    slow_span = max(fast_span + 1, int(max(10, stress_window) // 4))
    hazard_ema = hazard_score.ewm(span=fast_span, adjust=False).mean().clip(lower=0.0, upper=1.0)
    hazard_slow = hazard_score.ewm(span=slow_span, adjust=False).mean().clip(lower=0.0, upper=1.0)
    hazard_trend = _normalize_series_unit((hazard_ema - hazard_slow).clip(lower=0.0))
    hostile_event = bad_flag_series | synthetic_bad | (hazard_score >= 0.85)
    hazard_persistence = (
        hostile_event.astype(float)
        .rolling(max(5, int(sigma_window)), min_periods=1)
        .mean()
        .clip(lower=0.0, upper=1.0)
    )
    hazard_cooldown = _cooldown_series(hostile_event, decay=0.88)
    hazard_regime = [
        _hazard_regime(
            float(score),
            bad_flag=bool(flag),
            synthetic_bad=bool(synth),
        )
        for score, flag, synth in zip(hazard_score, bad_flag_series, synthetic_bad, strict=False)
    ]

    enriched["hazard_p_bad"] = p_bad_series.to_numpy(dtype=float)
    enriched["hazard_bad_flag"] = bad_flag_series.to_numpy(dtype=bool)
    enriched["hazard_synthetic_bad"] = synthetic_bad.to_numpy(dtype=bool)
    enriched["hazard_drawdown_pressure"] = _clip_unit(price_drawdown / 0.10).to_numpy(dtype=float)
    enriched["hazard_vol_pressure"] = vol_pressure.to_numpy(dtype=float)
    enriched["hazard_spread_pressure"] = spread_pressure.to_numpy(dtype=float)
    enriched["hazard_contextual_pressure"] = contextual_pressure.to_numpy(dtype=float)
    enriched["hazard_score"] = hazard_score.to_numpy(dtype=float)
    enriched["hazard_ema"] = hazard_ema.to_numpy(dtype=float)
    enriched["hazard_persistence"] = hazard_persistence.to_numpy(dtype=float)
    enriched["hazard_trend"] = hazard_trend.to_numpy(dtype=float)
    enriched["hazard_cooldown"] = hazard_cooldown.to_numpy(dtype=float)
    enriched["hazard_regime"] = hazard_regime
    return enriched


def row_hazard_observation(row: pd.Series | dict[str, object]) -> HazardObservation:
    getter = row.get if isinstance(row, dict) else row.get
    return HazardObservation(
        hazard_score=float(getter("hazard_score", getter("hazard", 0.0))),
        hazard_regime=str(getter("hazard_regime", "calm")),
        hazard_p_bad=float(getter("hazard_p_bad", 0.0)),
        hazard_bad_flag=bool(getter("hazard_bad_flag", False)),
        hazard_synthetic_bad=bool(getter("hazard_synthetic_bad", False)),
        hazard_drawdown_pressure=float(getter("hazard_drawdown_pressure", 0.0)),
        hazard_vol_pressure=float(getter("hazard_vol_pressure", 0.0)),
        hazard_spread_pressure=float(getter("hazard_spread_pressure", 0.0)),
        hazard_contextual_pressure=float(getter("hazard_contextual_pressure", 0.0)),
        hazard_contextual_active=bool(getter("hazard_contextual_active", False)),
        hazard_contextual_label=str(getter("hazard_contextual_label", "")),
        hazard_ema=float(getter("hazard_ema", getter("hazard_score", getter("hazard", 0.0)))),
        hazard_persistence=float(getter("hazard_persistence", 0.0)),
        hazard_trend=float(getter("hazard_trend", 0.0)),
        hazard_cooldown=float(getter("hazard_cooldown", 0.0)),
    )

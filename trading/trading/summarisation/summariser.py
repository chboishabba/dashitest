from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .quotient_map import QuotientMap, QuotientSpec
from .storage import DIMENSION_NAMES, SummaryStorage, quantile_label


@dataclass
class SummarySpec:
    window_seconds: int = 60
    persistence_threshold: float = 0.0
    quantile_points: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99)
    quantile_dims: tuple[int, ...] = (0, 1)
    event_dims: tuple[int, ...] = (1, 2)
    event_threshold: float = 1.0


class Summariser:
    """Summarises quotient representatives into non-destructive statistics."""

    def __init__(
        self,
        spec: SummarySpec | None = None,
        *,
        storage: SummaryStorage | None = None,
        db_path: Path | str = "logs/research/quotient_summaries.duckdb",
        parquet_out: Path | str | None = "logs/research/quotient_summaries.parquet",
    ) -> None:
        self.spec = spec or SummarySpec()
        self.qmap = QuotientMap(QuotientSpec())
        if storage is not None:
            self.storage = storage
        else:
            self.storage = SummaryStorage(db_path, self.spec, parquet_out=parquet_out)

    def summarize(
        self,
        close_series: Sequence[float] | np.ndarray,
        *,
        symbol: str = "BTCUSDT",
        timestamps: Sequence[Any] | None = None,
        legitimacy: Sequence[float] | None = None,
    ) -> dict[str, object]:
        prices = np.asarray(close_series, dtype=np.float32)
        if prices.size == 0:
            raise ValueError("close_series must contain at least one value")
        time_idx = self._normalize_sequence(timestamps, prices.shape[0])
        legit_idx = self._normalize_sequence(legitimacy, prices.shape[0]) if legitimacy is not None else None

        window_size = max(1, self.spec.window_seconds)
        segments: list[np.ndarray] = []
        bounds: list[tuple[Any | None, Any | None]] = []
        legit_windows: list[float] = []
        position = 0
        while position < prices.shape[0]:
            end = min(prices.shape[0], position + window_size)
            window = prices[position:end]
            if window.size >= 2:
                segments.append(self.qmap.compute(window))
                bounds.append(
                    (self._timestamp_at(time_idx, position), self._timestamp_at(time_idx, end - 1))
                )
                if legit_idx is not None:
                    legit_windows.append(float(np.nanmean(legit_idx[position:end])))
            position = end

        if not segments:
            segments.append(self.qmap.compute(prices))
            bounds.append(
                (self._timestamp_at(time_idx, 0), self._timestamp_at(time_idx, prices.shape[0] - 1))
            )
            if legit_idx is not None:
                legit_windows.append(float(np.nanmean(legit_idx)))

        q_matrix = np.vstack(segments).astype(np.float32)
        summary = self._build_summary(
            symbol=symbol,
            q_matrix=q_matrix,
            bounds=bounds,
            legit_windows=legit_windows,
        )
        self.storage.append(summary)
        return summary

    def write_summary(self, summary: dict[str, object], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, default=str)

    def _build_summary(
        self,
        *,
        symbol: str,
        q_matrix: np.ndarray,
        bounds: list[tuple[Any | None, Any | None]],
        legit_windows: list[float],
    ) -> dict[str, object]:
        window_start, window_end = self._choose_bounds(bounds)
        legitimacy_density, unknown_frac = self._summarize_legitimacy(legit_windows)
        event_marker = self._detect_event(q_matrix)
        summary: dict[str, object] = {
            "symbol": symbol,
            "window_start": window_start,
            "window_end": window_end,
            "window_seconds": self.spec.window_seconds,
            "count": int(q_matrix.shape[0]),
            "legitimacy_density": legitimacy_density,
            "unknown_frac": unknown_frac,
            "event_marker": event_marker,
            "artifact_ts": pd.Timestamp.utcnow(tz="UTC"),
        }
        for idx, name in enumerate(DIMENSION_NAMES):
            summary[f"mean_{name}"] = self._safe_float(np.nanmean(q_matrix[:, idx]))
            summary[f"median_{name}"] = self._safe_float(np.nanmedian(q_matrix[:, idx]))
            summary[f"mad_{name}"] = self._mad(q_matrix[:, idx])
            summary[f"run_length_pos_{name}"] = self._longest_run(q_matrix[:, idx], self.spec.persistence_threshold, positive=True)
            summary[f"run_length_neg_{name}"] = self._longest_run(q_matrix[:, idx], self.spec.persistence_threshold, positive=False)
        for dim in self.spec.quantile_dims:
            if 0 <= dim < q_matrix.shape[1]:
                name = DIMENSION_NAMES[dim]
                for point in self.spec.quantile_points:
                    value = self._safe_quantile(q_matrix[:, dim], point)
                    summary[f"quant_{name}_{quantile_label(point)}"] = value
        return summary

    def _choose_bounds(self, bounds: list[tuple[Any | None, Any | None]]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if not bounds:
            return None, None
        start, _ = bounds[0]
        _, end = bounds[-1]
        return self._ensure_timestamp(start), self._ensure_timestamp(end)

    def _summarize_legitimacy(self, windows: list[float]) -> tuple[float, float]:
        if not windows:
            return float("nan"), 1.0
        arr = np.asarray(windows, dtype=np.float32)
        valid_mask = np.isfinite(arr) & (arr >= 0.0) & (arr <= 1.0)
        density = float(np.nanmean(arr[valid_mask])) if valid_mask.any() else float("nan")
        unknown = 1.0 - float(np.count_nonzero(valid_mask)) / arr.size
        return density, unknown

    def _detect_event(self, q_matrix: np.ndarray) -> bool:
        if q_matrix.size == 0:
            return False
        for dim in self.spec.event_dims:
            if 0 <= dim < q_matrix.shape[1]:
                values = np.abs(q_matrix[:, dim])
                if np.nanmax(values) >= self.spec.event_threshold:
                    return True
        return False

    def _longest_run(self, values: np.ndarray, threshold: float, *, positive: bool) -> int:
        mask = np.isfinite(values) & (
            values >= threshold if positive else values <= -threshold
        )
        max_run = 0
        current = 0
        for flag in mask:
            if flag:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return int(max_run)

    def _mad(self, values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        med = np.nanmedian(values)
        mad_val = np.nanmedian(np.abs(values - med))
        return float(mad_val)

    def _safe_float(self, value: float) -> float:
        if not np.isfinite(value):
            return float("nan")
        return float(value)

    def _safe_quantile(self, values: np.ndarray, point: float) -> float:
        if values.size == 0:
            return float("nan")
        try:
            return float(np.nanquantile(values, point))
        except (ValueError, IndexError):
            return float("nan")

    def _normalize_sequence(self, sequence: Sequence[Any] | None, length: int) -> np.ndarray | None:
        if sequence is None:
            return None
        if len(sequence) != length:
            raise ValueError("Sequence length must match close_series length")
        return np.asarray(sequence)

    def _timestamp_at(self, timestamps: np.ndarray | None, index: int) -> Any | None:
        if timestamps is None:
            return None
        try:
            return timestamps[index]
        except (IndexError, TypeError):
            return None

    def _ensure_timestamp(self, value: Any | None) -> pd.Timestamp | None:
        if value is None:
            return None
        return pd.Timestamp(value)

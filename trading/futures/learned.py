from __future__ import annotations

import csv
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

from .beam import HeuristicTransitionModel, TransitionCandidate, _clip
from .state import CoarseState, CoarseStateEstimator, ObservationSnapshot


def _set_csv_field_limit_to_max() -> None:
    """
    Lift CSV parser field limits so large diagnostic columns do not break
    learned-kernel fitting.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _sign_bucket(value: float, threshold: float = 0.1) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _unit_bucket(value: float, lo: float = 0.33, hi: float = 0.66) -> int:
    if value < lo:
        return 0
    if value < hi:
        return 1
    return 2


def _float(row: dict[str, str], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            return value
    return default


def _int(row: dict[str, str], *keys: str, default: int = 0) -> int:
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return int(float(raw))
        except ValueError:
            continue
    return default


def _trend_return(rows: Sequence[dict[str, str]], idx: int, window: int = 8) -> float:
    start = max(0, idx - window + 1)
    vals = [_float(rows[j], "price_ret") for j in range(start, idx + 1)]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _drawdown_ratio(equity: float, start_equity: float) -> float:
    if start_equity <= 0:
        return 0.0
    return max(0.0, (start_equity - equity) / start_equity)


@dataclass(frozen=True)
class TransitionObservation:
    current_state: CoarseState
    action: int
    next_exposure: float
    next_state: CoarseState
    step_return: float
    branch_risk: float
    diffusion_risk: float
    realized_vol: float
    label: str


@dataclass(frozen=True)
class _BucketStats:
    probability: float
    next_state: CoarseState
    step_return: float
    branch_risk: float
    diffusion_risk: float
    label: str


class LearnedTransitionModel:
    """
    Transition generator estimated from historical per-step trader logs.
    """

    def __init__(
        self,
        buckets: dict[tuple, list[_BucketStats]],
        by_action: dict[int, list[_BucketStats]],
        heuristic_fallback: HeuristicTransitionModel | None = None,
        metadata: dict[str, float | int | str] | None = None,
    ):
        self.buckets = buckets
        self.by_action = by_action
        self.heuristic_fallback = heuristic_fallback or HeuristicTransitionModel()
        self.metadata = metadata or {}

    @classmethod
    def from_observations(
        cls,
        observations: Iterable[TransitionObservation],
        *,
        min_bucket_samples: int = 12,
        label_mode: str = "fixed",
        label_threshold: float = 0.01,
        label_vol_mult: float = 0.5,
    ) -> LearnedTransitionModel:
        grouped: dict[tuple, list[TransitionObservation]] = {}
        action_grouped: dict[int, list[TransitionObservation]] = {}
        label_counts = {"long": 0, "short": 0, "flat": 0, "stress": 0}
        observation_list = list(observations)
        for obs in observation_list:
            key = cls._bucket_key(obs.current_state, obs.action)
            grouped.setdefault(key, []).append(obs)
            action_grouped.setdefault(obs.action, []).append(obs)
            if obs.label in label_counts:
                label_counts[obs.label] += 1
        buckets = {
            key: cls._summarize_group(
                items,
                label_mode=label_mode,
                label_threshold=label_threshold,
                label_vol_mult=label_vol_mult,
            )
            for key, items in grouped.items()
            if len(items) >= min_bucket_samples
        }
        by_action = {
            action: cls._summarize_group(
                items,
                label_mode=label_mode,
                label_threshold=label_threshold,
                label_vol_mult=label_vol_mult,
            )
            for action, items in action_grouped.items()
            if items
        }
        return cls(
            buckets=buckets,
            by_action=by_action,
            metadata={
                "observation_count": len(observation_list),
                "bucket_count": len(buckets),
                "min_bucket_samples": min_bucket_samples,
                "label_mode": label_mode,
                "label_threshold": label_threshold,
                "label_vol_mult": label_vol_mult,
                "label_count_long": label_counts["long"],
                "label_count_short": label_counts["short"],
                "label_count_flat": label_counts["flat"],
                "label_count_stress": label_counts["stress"],
            },
        )

    @classmethod
    def from_csv_paths(
        cls,
        paths: Sequence[pathlib.Path],
        *,
        estimator: CoarseStateEstimator | None = None,
        min_bucket_samples: int = 12,
        max_rows_per_file: int = 12000,
        label_mode: str = "fixed",
        label_threshold: float = 0.01,
        label_vol_mult: float = 0.5,
    ) -> LearnedTransitionModel | None:
        estimator = estimator or CoarseStateEstimator()
        observations: list[TransitionObservation] = []
        used_paths: list[str] = []
        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            file_observations = list(
                _iter_observations_from_csv(
                    path=path,
                    estimator=estimator,
                    max_rows=max_rows_per_file,
                    label_mode=label_mode,
                    label_threshold=label_threshold,
                    label_vol_mult=label_vol_mult,
                )
            )
            if not file_observations:
                continue
            used_paths.append(str(path))
            observations.extend(file_observations)
        if not observations:
            return None
        model = cls.from_observations(
            observations,
            min_bucket_samples=min_bucket_samples,
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
        model.metadata.update(
            {
                "source_count": len(used_paths),
                "sources": ",".join(used_paths[:8]),
            }
        )
        return model

    @staticmethod
    def _bucket_key(state: CoarseState, action: int) -> tuple[int, int, int, int, int, int]:
        return (
            int(action),
            _sign_bucket(state.drift, threshold=0.2),
            _sign_bucket(state.triadic_bias, threshold=0.2),
            _unit_bucket(state.actionability),
            _unit_bucket(state.stress),
            _sign_bucket(state.edge, threshold=0.15),
        )

    @classmethod
    def _label_for_observation(
        cls,
        obs: TransitionObservation,
        *,
        label_mode: str,
        label_threshold: float,
        label_vol_mult: float,
    ) -> str:
        if label_mode == "fixed":
            threshold = label_threshold
        elif label_mode == "vol":
            threshold = max(label_threshold, label_vol_mult * max(obs.realized_vol, 0.0))
        else:
            raise ValueError(f"unknown label_mode: {label_mode}")
        if obs.next_state.stress >= 0.75 or obs.branch_risk >= 0.75:
            return "stress"
        if obs.step_return > threshold:
            return "long"
        if obs.step_return < -threshold:
            return "short"
        return "flat"

    @classmethod
    def _summarize_group(
        cls,
        observations: Sequence[TransitionObservation],
        *,
        label_mode: str,
        label_threshold: float,
        label_vol_mult: float,
    ) -> list[_BucketStats]:
        label_groups: dict[str, list[TransitionObservation]] = {}
        for obs in observations:
            label_groups.setdefault(obs.label, []).append(obs)
        total = float(len(observations))
        stats: list[_BucketStats] = []
        for label, items in sorted(label_groups.items(), key=lambda item: len(item[1]), reverse=True):
            count = float(len(items))
            inv = 1.0 / count
            next_state = CoarseState(
                drift=sum(item.next_state.drift for item in items) * inv,
                triadic_bias=sum(item.next_state.triadic_bias for item in items) * inv,
                actionability=sum(item.next_state.actionability for item in items) * inv,
                stress=sum(item.next_state.stress for item in items) * inv,
                edge=sum(item.next_state.edge for item in items) * inv,
                contraction=sum(item.next_state.contraction for item in items) * inv,
                diffusion=sum(item.next_state.diffusion for item in items) * inv,
                drawdown=sum(item.next_state.drawdown for item in items) * inv,
                current_exposure=sum(item.next_state.current_exposure for item in items) * inv,
            )
            step_return = sum(item.step_return for item in items) * inv
            if label == "flat":
                step_return = 0.0
            stats.append(
                _BucketStats(
                    probability=count / total,
                    next_state=next_state,
                    step_return=step_return,
                    branch_risk=sum(item.branch_risk for item in items) * inv,
                    diffusion_risk=sum(item.diffusion_risk for item in items) * inv,
                    label=label,
                )
            )
        return stats

    def expand(self, state: CoarseState, action: int, next_exposure: float):
        key = self._bucket_key(state, action)
        candidates = self.buckets.get(key) or self.by_action.get(action)
        if not candidates:
            yield from self.heuristic_fallback.expand(state, action, next_exposure)
            return
        total_prob = sum(candidate.probability for candidate in candidates)
        if total_prob <= 0:
            yield from self.heuristic_fallback.expand(state, action, next_exposure)
            return
        inv_total = 1.0 / total_prob
        for candidate in candidates:
            next_state = candidate.next_state
            yield TransitionCandidate(
                label=f"learned_{candidate.label}",
                probability=max(1e-6, candidate.probability * inv_total),
                next_state=CoarseState(
                    drift=_clip(next_state.drift, -1.0, 1.0),
                    triadic_bias=_clip(next_state.triadic_bias, -1.0, 1.0),
                    actionability=_clip(next_state.actionability, 0.0, 1.0),
                    stress=_clip(next_state.stress, 0.0, 1.0),
                    edge=_clip(next_state.edge, -1.0, 1.0),
                    contraction=_clip(next_state.contraction, 0.0, 1.0),
                    diffusion=_clip(next_state.diffusion, 0.0, 1.0),
                    drawdown=_clip(next_state.drawdown, 0.0, 1.0),
                    current_exposure=_clip(next_exposure, -1.0, 1.0),
                ),
                step_return=candidate.step_return,
                step_branch_risk=max(0.0, candidate.branch_risk),
                step_diffusion_risk=max(0.0, candidate.diffusion_risk),
            )


class ResidualTransitionModel:
    def __init__(
        self,
        global_model: LearnedTransitionModel | HeuristicTransitionModel,
        asset_model: LearnedTransitionModel | HeuristicTransitionModel,
        residual_weight: float = 1.0,
    ):
        self.global_model = global_model
        self.asset_model = asset_model
        self.residual_weight = float(residual_weight)
        global_meta = getattr(global_model, "metadata", {}) or {}
        asset_meta = getattr(asset_model, "metadata", {}) or {}
        label_meta = asset_meta if asset_meta.get("observation_count", 0) else global_meta
        self.metadata = {
            "kernel_mode": "residual",
            "fallback_used": int(bool(global_meta.get("fallback_used", False) and asset_meta.get("fallback_used", False))),
            "source_count": int(global_meta.get("source_count", 0)) + int(asset_meta.get("source_count", 0)),
            "bucket_count": max(int(global_meta.get("bucket_count", 0)), int(asset_meta.get("bucket_count", 0))),
            "label_count_long": int(label_meta.get("label_count_long", 0)),
            "label_count_short": int(label_meta.get("label_count_short", 0)),
            "label_count_flat": int(label_meta.get("label_count_flat", 0)),
            "label_count_stress": int(label_meta.get("label_count_stress", 0)),
            "label_mode": label_meta.get("label_mode", "fixed"),
            "label_threshold": float(label_meta.get("label_threshold", 0.0)),
            "label_vol_mult": float(label_meta.get("label_vol_mult", 0.0)),
        }

    def expand(self, state: CoarseState, action: int, next_exposure: float):
        global_candidates = list(self.global_model.expand(state, action, next_exposure))
        asset_candidates = list(self.asset_model.expand(state, action, next_exposure))
        if not global_candidates:
            yield from asset_candidates
            return
        if not asset_candidates:
            yield from global_candidates
            return
        combined: dict[str, TransitionCandidate] = {}
        for candidate in global_candidates:
            combined[candidate.label] = candidate
        for candidate in asset_candidates:
            if candidate.label not in combined:
                combined[candidate.label] = candidate
                continue
            base = combined[candidate.label]
            weight = self.residual_weight
            probability = max(1e-6, base.probability + weight * (candidate.probability - base.probability))
            combined[candidate.label] = TransitionCandidate(
                label=candidate.label,
                probability=probability,
                next_state=CoarseState(
                    drift=_clip(base.next_state.drift + weight * (candidate.next_state.drift - base.next_state.drift), -1.0, 1.0),
                    triadic_bias=_clip(base.next_state.triadic_bias + weight * (candidate.next_state.triadic_bias - base.next_state.triadic_bias), -1.0, 1.0),
                    actionability=_clip(base.next_state.actionability + weight * (candidate.next_state.actionability - base.next_state.actionability), 0.0, 1.0),
                    stress=_clip(base.next_state.stress + weight * (candidate.next_state.stress - base.next_state.stress), 0.0, 1.0),
                    edge=_clip(base.next_state.edge + weight * (candidate.next_state.edge - base.next_state.edge), -1.0, 1.0),
                    contraction=_clip(base.next_state.contraction + weight * (candidate.next_state.contraction - base.next_state.contraction), 0.0, 1.0),
                    diffusion=_clip(base.next_state.diffusion + weight * (candidate.next_state.diffusion - base.next_state.diffusion), 0.0, 1.0),
                    drawdown=_clip(base.next_state.drawdown + weight * (candidate.next_state.drawdown - base.next_state.drawdown), 0.0, 1.0),
                    current_exposure=_clip(next_exposure, -1.0, 1.0),
                ),
                step_return=base.step_return + weight * (candidate.step_return - base.step_return),
                step_branch_risk=max(0.0, base.step_branch_risk + weight * (candidate.step_branch_risk - base.step_branch_risk)),
                step_diffusion_risk=max(0.0, base.step_diffusion_risk + weight * (candidate.step_diffusion_risk - base.step_diffusion_risk)),
            )
        total_prob = sum(candidate.probability for candidate in combined.values())
        if total_prob <= 0:
            yield from asset_candidates
            return
        inv_total = 1.0 / total_prob
        for candidate in combined.values():
            yield TransitionCandidate(
                label=candidate.label,
                probability=max(1e-6, candidate.probability * inv_total),
                next_state=candidate.next_state,
                step_return=candidate.step_return,
                step_branch_risk=candidate.step_branch_risk,
                step_diffusion_risk=candidate.step_diffusion_risk,
            )


class ShrinkageTransitionModel:
    def __init__(
        self,
        global_model: LearnedTransitionModel | HeuristicTransitionModel,
        asset_model: LearnedTransitionModel | HeuristicTransitionModel,
    ):
        self.global_model = global_model
        self.asset_model = asset_model
        global_meta = getattr(global_model, "metadata", {}) or {}
        asset_meta = getattr(asset_model, "metadata", {}) or {}
        label_meta = asset_meta if asset_meta.get("observation_count", 0) else global_meta
        global_count = int(global_meta.get("observation_count", 0))
        asset_count = int(asset_meta.get("observation_count", 0))
        total = global_count + asset_count
        kernel_lambda = asset_count / total if total > 0 else 0.5
        self.metadata = {
            "kernel_mode": "shrinkage",
            "kernel_lambda": float(kernel_lambda),
            "kernel_asset_count": int(asset_count),
            "kernel_global_count": int(global_count),
            "fallback_used": int(bool(global_meta.get("fallback_used", False) and asset_meta.get("fallback_used", False))),
            "source_count": int(global_meta.get("source_count", 0)) + int(asset_meta.get("source_count", 0)),
            "bucket_count": max(int(global_meta.get("bucket_count", 0)), int(asset_meta.get("bucket_count", 0))),
            "label_count_long": int(label_meta.get("label_count_long", 0)),
            "label_count_short": int(label_meta.get("label_count_short", 0)),
            "label_count_flat": int(label_meta.get("label_count_flat", 0)),
            "label_count_stress": int(label_meta.get("label_count_stress", 0)),
            "label_mode": label_meta.get("label_mode", "fixed"),
            "label_threshold": float(label_meta.get("label_threshold", 0.0)),
            "label_vol_mult": float(label_meta.get("label_vol_mult", 0.0)),
        }

    def expand(self, state: CoarseState, action: int, next_exposure: float):
        global_candidates = list(self.global_model.expand(state, action, next_exposure))
        asset_candidates = list(self.asset_model.expand(state, action, next_exposure))
        if not global_candidates:
            yield from asset_candidates
            return
        if not asset_candidates:
            yield from global_candidates
            return
        weight = float(self.metadata.get("kernel_lambda", 0.5))
        combined: dict[str, TransitionCandidate] = {}
        for candidate in global_candidates:
            combined[candidate.label] = candidate
        for candidate in asset_candidates:
            if candidate.label not in combined:
                combined[candidate.label] = candidate
                continue
            base = combined[candidate.label]
            probability = max(1e-6, (1.0 - weight) * base.probability + weight * candidate.probability)
            combined[candidate.label] = TransitionCandidate(
                label=candidate.label,
                probability=probability,
                next_state=CoarseState(
                    drift=_clip(base.next_state.drift + weight * (candidate.next_state.drift - base.next_state.drift), -1.0, 1.0),
                    triadic_bias=_clip(base.next_state.triadic_bias + weight * (candidate.next_state.triadic_bias - base.next_state.triadic_bias), -1.0, 1.0),
                    actionability=_clip(base.next_state.actionability + weight * (candidate.next_state.actionability - base.next_state.actionability), 0.0, 1.0),
                    stress=_clip(base.next_state.stress + weight * (candidate.next_state.stress - base.next_state.stress), 0.0, 1.0),
                    edge=_clip(base.next_state.edge + weight * (candidate.next_state.edge - base.next_state.edge), -1.0, 1.0),
                    contraction=_clip(base.next_state.contraction + weight * (candidate.next_state.contraction - base.next_state.contraction), 0.0, 1.0),
                    diffusion=_clip(base.next_state.diffusion + weight * (candidate.next_state.diffusion - base.next_state.diffusion), 0.0, 1.0),
                    drawdown=_clip(base.next_state.drawdown + weight * (candidate.next_state.drawdown - base.next_state.drawdown), 0.0, 1.0),
                    current_exposure=_clip(next_exposure, -1.0, 1.0),
                ),
                step_return=base.step_return + weight * (candidate.step_return - base.step_return),
                step_branch_risk=max(0.0, base.step_branch_risk + weight * (candidate.step_branch_risk - base.step_branch_risk)),
                step_diffusion_risk=max(0.0, base.step_diffusion_risk + weight * (candidate.step_diffusion_risk - base.step_diffusion_risk)),
            )
        total_prob = sum(candidate.probability for candidate in combined.values())
        if total_prob <= 0:
            yield from global_candidates
            return
        inv_total = 1.0 / total_prob
        for candidate in combined.values():
            yield TransitionCandidate(
                label=candidate.label,
                probability=max(1e-6, candidate.probability * inv_total),
                next_state=candidate.next_state,
                step_return=candidate.step_return,
                step_branch_risk=candidate.step_branch_risk,
                step_diffusion_risk=candidate.step_diffusion_risk,
            )


def _eligible_global_log_paths(log_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in log_dir.glob("trading_log*.csv")
        if "trades" not in path.name and "vulkan" not in path.name and path.name != "trading_log.csv"
    )


def candidate_log_paths(log_dir: pathlib.Path, symbol_name: str) -> list[pathlib.Path]:
    safe = symbol_name.replace(":", "_").replace("/", "_")
    preferred = [
        log_dir / f"trading_log_{safe}.csv",
    ]
    matched = [path for path in preferred if path.exists()]
    if matched:
        return matched
    return _eligible_global_log_paths(log_dir)


def _heuristic_with_metadata(*, kernel_mode: str, fallback_used: bool = True) -> HeuristicTransitionModel:
    model = HeuristicTransitionModel()
    model.metadata = {
        "kernel_mode": kernel_mode,
        "fallback_used": int(bool(fallback_used)),
        "source_count": 0,
        "bucket_count": 0,
        "observation_count": 0,
    }
    return model


def _fit_learned_model(
    *,
    paths: Sequence[pathlib.Path],
    estimator: CoarseStateEstimator,
    min_bucket_samples: int,
    max_rows_per_file: int,
    kernel_mode: str,
    fallback_used: bool,
    label_mode: str,
    label_threshold: float,
    label_vol_mult: float,
) -> LearnedTransitionModel | HeuristicTransitionModel:
    model = LearnedTransitionModel.from_csv_paths(
        paths,
        estimator=estimator,
        min_bucket_samples=min_bucket_samples,
        max_rows_per_file=max_rows_per_file,
        label_mode=label_mode,
        label_threshold=label_threshold,
        label_vol_mult=label_vol_mult,
    )
    if model is None or not model.metadata.get("observation_count", 0):
        return _heuristic_with_metadata(kernel_mode=kernel_mode, fallback_used=True)
    model.metadata.update(
        {
            "kernel_mode": kernel_mode,
            "fallback_used": int(bool(fallback_used)),
            "label_mode": label_mode,
            "label_threshold": label_threshold,
            "label_vol_mult": label_vol_mult,
        }
    )
    return model


def build_transition_model_from_logs(
    *,
    symbol_name: str,
    log_dir: pathlib.Path | str = pathlib.Path("logs"),
    estimator: CoarseStateEstimator | None = None,
    min_bucket_samples: int = 12,
    max_rows_per_file: int = 12000,
    mode: str = "per_asset",
    residual_weight: float = 1.0,
    label_mode: str = "fixed",
    label_threshold: float = 0.01,
    label_vol_mult: float = 0.5,
) -> LearnedTransitionModel | HeuristicTransitionModel | ResidualTransitionModel | ShrinkageTransitionModel:
    log_dir = pathlib.Path(log_dir)
    estimator = estimator or CoarseStateEstimator()
    if mode == "global":
        return _fit_learned_model(
            paths=_eligible_global_log_paths(log_dir),
            estimator=estimator,
            min_bucket_samples=min_bucket_samples,
            max_rows_per_file=max_rows_per_file,
            kernel_mode="global",
            fallback_used=False,
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
    asset_paths = candidate_log_paths(log_dir, symbol_name)
    if mode == "per_asset":
        asset_model = LearnedTransitionModel.from_csv_paths(
            asset_paths,
            estimator=estimator,
            min_bucket_samples=min_bucket_samples,
            max_rows_per_file=max_rows_per_file,
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
        if asset_model is not None and asset_model.metadata.get("observation_count", 0):
            asset_model.metadata.update({"kernel_mode": "per_asset", "fallback_used": 0})
            return asset_model
        global_model = _fit_learned_model(
            paths=_eligible_global_log_paths(log_dir),
            estimator=estimator,
            min_bucket_samples=min_bucket_samples,
            max_rows_per_file=max_rows_per_file,
            kernel_mode="per_asset",
            fallback_used=True,
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
        meta = getattr(global_model, "metadata", {}) or {}
        meta["kernel_mode"] = "per_asset"
        meta["fallback_used"] = 1
        global_model.metadata = meta
        return global_model
    if mode == "residual":
        global_model = _fit_learned_model(
            paths=_eligible_global_log_paths(log_dir),
            estimator=estimator,
            min_bucket_samples=min_bucket_samples,
            max_rows_per_file=max_rows_per_file,
            kernel_mode="global",
            fallback_used=False,
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
        asset_model = _fit_learned_model(
            paths=asset_paths,
            estimator=estimator,
            min_bucket_samples=min_bucket_samples,
            max_rows_per_file=max_rows_per_file,
            kernel_mode="per_asset",
            fallback_used=False,
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
        residual_model = ResidualTransitionModel(global_model=global_model, asset_model=asset_model, residual_weight=residual_weight)
        residual_model.metadata["fallback_used"] = int(
            bool(
                getattr(global_model, "metadata", {}).get("fallback_used", False)
                and getattr(asset_model, "metadata", {}).get("fallback_used", False)
            )
        )
        return residual_model
    if mode == "shrinkage":
        global_model = _fit_learned_model(
            paths=_eligible_global_log_paths(log_dir),
            estimator=estimator,
            min_bucket_samples=min_bucket_samples,
            max_rows_per_file=max_rows_per_file,
            kernel_mode="global",
            fallback_used=False,
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
        asset_model = _fit_learned_model(
            paths=asset_paths,
            estimator=estimator,
            min_bucket_samples=min_bucket_samples,
            max_rows_per_file=max_rows_per_file,
            kernel_mode="per_asset",
            fallback_used=False,
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
        return ShrinkageTransitionModel(global_model=global_model, asset_model=asset_model)
    raise ValueError(f"unknown shadow kernel mode: {mode}")


def _iter_observations_from_csv(
    *,
    path: pathlib.Path,
    estimator: CoarseStateEstimator,
    max_rows: int,
    label_mode: str,
    label_threshold: float,
    label_vol_mult: float,
) -> Iterable[TransitionObservation]:
    _set_csv_field_limit_to_max()
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < 2:
        return
    if max_rows > 0 and len(rows) > max_rows:
        rows = rows[-max_rows:]
    start_equity = _float(rows[0], "equity", "cap", "pnl", default=0.0)
    for idx in range(len(rows) - 1):
        row = rows[idx]
        nxt = rows[idx + 1]
        current_exposure = _clip(_float(row, "pos", default=0.0), -1.0, 1.0)
        next_exposure = _clip(_float(nxt, "pos", default=current_exposure), -1.0, 1.0)
        action = _sign_bucket(next_exposure - current_exposure, threshold=1e-6)
        current_snapshot = ObservationSnapshot(
            price_return=_float(row, "price_ret"),
            trend_return=_trend_return(rows, idx),
            realized_vol=max(abs(_float(row, "sigma_slow", default=0.0)), 1e-9),
            triadic_state=_int(row, "direction", default=0),
            actionability=1.0 if _int(row, "permission", default=0) == 1 else 0.5 if _int(row, "permission", default=0) == 0 else 0.0,
            stress=_float(row, "stress"),
            edge=_float(row, "pred_edge", "edge_ema", "edge_raw"),
            drawdown=_drawdown_ratio(_float(row, "equity", "cap", default=start_equity), start_equity),
            current_exposure=current_exposure,
        )
        next_snapshot = ObservationSnapshot(
            price_return=_float(nxt, "price_ret"),
            trend_return=_trend_return(rows, idx + 1),
            realized_vol=max(abs(_float(nxt, "sigma_slow", default=0.0)), 1e-9),
            triadic_state=_int(nxt, "direction", default=0),
            actionability=1.0 if _int(nxt, "permission", default=0) == 1 else 0.5 if _int(nxt, "permission", default=0) == 0 else 0.0,
            stress=_float(nxt, "stress"),
            edge=_float(nxt, "pred_edge", "edge_ema", "edge_raw"),
            drawdown=_drawdown_ratio(_float(nxt, "equity", "cap", default=start_equity), start_equity),
            current_exposure=next_exposure,
        )
        current_state = estimator.estimate(current_snapshot)
        next_state = estimator.estimate(next_snapshot)
        step_return = next_state.drift
        branch_risk = max(0.0, next_state.stress + max(0.0, next_state.drawdown - current_state.drawdown))
        diffusion_risk = max(next_state.diffusion, abs(next_state.drift - current_state.drift))
        label = LearnedTransitionModel._label_for_observation(
            TransitionObservation(
                current_state=current_state,
                action=action,
                next_exposure=next_exposure,
                next_state=next_state,
                step_return=step_return,
                branch_risk=branch_risk,
                diffusion_risk=diffusion_risk,
                realized_vol=current_snapshot.realized_vol,
                label="flat",
            ),
            label_mode=label_mode,
            label_threshold=label_threshold,
            label_vol_mult=label_vol_mult,
        )
        yield TransitionObservation(
            current_state=current_state,
            action=action,
            next_exposure=next_exposure,
            next_state=next_state,
            step_return=step_return,
            branch_risk=branch_risk,
            diffusion_risk=diffusion_risk,
            realized_vol=current_snapshot.realized_vol,
            label=label,
        )

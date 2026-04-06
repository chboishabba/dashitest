from __future__ import annotations

import numpy as np

from .bridge_agda import (
    canonical_arrow,
    canonical_mask,
    canonical_projection,
    default_source_repo_root,
    load_dasl_source_model,
    source_eigen_fn,
    source_support_basin_pred,
)
from .contracts import NashiContract
from .schema import CANONICAL_FEATURE_COLUMNS


def l1_mdl(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(x, dtype=float))))


def spoke_eigen(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    thirds = np.array_split(np.abs(x), 3)
    padded = list(thirds)
    while len(padded) < 3:
        padded.append(np.zeros(1, dtype=float))
    return {
        "Earth": float(np.sum(padded[0])),
        "Spoke": float(np.sum(padded[1])),
        "Hub": float(np.sum(padded[2])),
    }


def radius_basin(center: np.ndarray, radius: float):
    center = np.asarray(center, dtype=float)

    def pred(x: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(x, dtype=float) - center)) <= radius

    return pred


def make_default_contract(feature_dim: int) -> NashiContract:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    mask = np.ones(feature_dim, dtype=float)
    mask[-1] = -1.0

    def projection(v: np.ndarray) -> np.ndarray:
        return np.asarray(v[:feature_dim], dtype=float)

    def arrow_fn(v: np.ndarray) -> float:
        return float(np.asarray(v, dtype=float)[-1])

    return NashiContract(
        mask=mask,
        projection=projection,
        mdl_fn=lambda v: l1_mdl(projection(v)),
        arrow_fn=arrow_fn,
        basin_pred=radius_basin(np.zeros(feature_dim, dtype=float), radius=10.0),
        eigen_fn=lambda v: spoke_eigen(projection(v)),
    )


def make_canonical_contract(repo_root=None) -> NashiContract:
    source_model = load_dasl_source_model(repo_root or default_source_repo_root())
    return NashiContract(
        mask=canonical_mask(),
        projection=canonical_projection,
        mdl_fn=lambda v: l1_mdl(canonical_projection(v)),
        arrow_fn=canonical_arrow,
        basin_pred=source_support_basin_pred(source_model),
        eigen_fn=source_eigen_fn(source_model),
    )


def canonical_feature_dim() -> int:
    return len(CANONICAL_FEATURE_COLUMNS)

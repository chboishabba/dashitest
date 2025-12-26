from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

from . import mdl_sideinfo, motion_search


def translation_mdl_from_motion(
    mv_grid: jnp.ndarray,
    alpha_u: float,
    alpha_v: float,
    radius: int,
    lag_j: int = 1,
    gamma: float = 0.5,
    max_lag: int = 1,
) -> float:
    bits = mdl_sideinfo.motion_translation_side_bits(
        mv_grid,
        alpha_u=alpha_u,
        alpha_v=alpha_v,
        radius=radius,
        lag_j=lag_j,
        gamma=gamma,
        max_lag=max_lag,
    )
    return float(bits)


def motion_compensated_residual_with_mdl(
    frames: np.ndarray,
    block: int = 8,
    search: int = 4,
    alpha_u: float = 0.6,
    alpha_v: float = 0.6,
    gamma: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, object]]:
    residuals, stats = motion_search.motion_compensated_residual(frames, block=block, search=search)
    mv_counts = stats.get("mv_counts", {})
    mv_list = []
    for (u, v), count in mv_counts.items():
        mv_list.extend([(u, v)] * count)
    if mv_list:
        mv_grid = jnp.asarray(mv_list, dtype=jnp.int32).reshape(-1, 1, 2)
        side_bits = translation_mdl_from_motion(
            mv_grid, alpha_u=alpha_u, alpha_v=alpha_v, radius=search, gamma=gamma
        )
    else:
        side_bits = 0.0
    stats["mv_side_bits"] = side_bits
    return residuals, stats

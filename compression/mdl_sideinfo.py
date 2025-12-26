from __future__ import annotations

from math import log2
from typing import Iterable


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0,1), got {value}.")


def truncated_two_sided_geom_norm(alpha: float, radius: int) -> float:
    _check_unit_interval("alpha", alpha)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    if radius == 0:
        return 1.0
    return 1.0 + 2.0 * alpha * (1.0 - alpha**radius) / (1.0 - alpha)


def log2_two_sided_geom(n: int, alpha: float) -> float:
    _check_unit_interval("alpha", alpha)
    return -log2((1.0 - alpha) / (1.0 + alpha)) - abs(n) * log2(alpha)


def log2_truncated_two_sided_geom(n: int, alpha: float, radius: int) -> float:
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    if abs(n) > radius:
        raise ValueError(f"n must be within [-{radius},{radius}], got {n}.")
    z_r = truncated_two_sided_geom_norm(alpha, radius)
    return log2(z_r) - abs(n) * log2(alpha)


def log2_lag_prior(j: int, gamma: float, max_lag: int) -> float:
    _check_unit_interval("gamma", gamma)
    if max_lag <= 0:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}.")
    if not 1 <= j <= max_lag:
        raise ValueError(f"j must be in [1,{max_lag}], got {j}.")
    trunc = 1.0 - gamma**max_lag
    return -log2(1.0 - gamma) - (j - 1) * log2(gamma) + log2(trunc)


def log2_warp_tag(pi_k: float) -> float:
    _check_unit_interval("pi_k", pi_k)
    return -log2(pi_k)


def log2_quadtree_prior(internal_count: int, leaf_nonmax_count: int, p_split: float) -> float:
    _check_unit_interval("p_split", p_split)
    if internal_count < 0 or leaf_nonmax_count < 0:
        raise ValueError("internal_count and leaf_nonmax_count must be >= 0.")
    return -internal_count * log2(p_split) - leaf_nonmax_count * log2(1.0 - p_split)


def log2_quadtree_prior_depth(
    internal_depths: Iterable[int],
    leaf_depths: Iterable[int],
    p_split_by_depth: dict[int, float],
) -> float:
    total = 0.0
    for depth in internal_depths:
        p_split = p_split_by_depth[depth]
        total -= log2(p_split)
    for depth in leaf_depths:
        p_split = p_split_by_depth[depth]
        total -= log2(1.0 - p_split)
    return total


def log2_translation_params(u: int, v: int, alpha_u: float, alpha_v: float, radius: int) -> float:
    return log2_truncated_two_sided_geom(u, alpha_u, radius) + log2_truncated_two_sided_geom(
        v, alpha_v, radius
    )


def log2_similarity_params(
    u: int,
    v: int,
    theta_idx: int,
    scale_idx: int,
    alpha_u: float,
    alpha_v: float,
    alpha_theta: float,
    alpha_scale: float,
    radius_uv: int,
    radius_theta: int,
    radius_scale: int,
) -> float:
    return (
        log2_truncated_two_sided_geom(u, alpha_u, radius_uv)
        + log2_truncated_two_sided_geom(v, alpha_v, radius_uv)
        + log2_truncated_two_sided_geom(theta_idx, alpha_theta, radius_theta)
        + log2_truncated_two_sided_geom(scale_idx, alpha_scale, radius_scale)
    )


def log2_affine_params(
    u: int,
    v: int,
    a11: int,
    a12: int,
    a21: int,
    a22: int,
    alpha_u: float,
    alpha_v: float,
    alpha_a11: float,
    alpha_a12: float,
    alpha_a21: float,
    alpha_a22: float,
    radius_uv: int,
    radius_a: int,
) -> float:
    return (
        log2_truncated_two_sided_geom(u, alpha_u, radius_uv)
        + log2_truncated_two_sided_geom(v, alpha_v, radius_uv)
        + log2_truncated_two_sided_geom(a11, alpha_a11, radius_a)
        + log2_truncated_two_sided_geom(a12, alpha_a12, radius_a)
        + log2_truncated_two_sided_geom(a21, alpha_a21, radius_a)
        + log2_truncated_two_sided_geom(a22, alpha_a22, radius_a)
    )


def motion_translation_side_bits(
    mv_counts: dict[tuple[int, int], int],
    alpha_u: float,
    alpha_v: float,
    radius: int,
    lag_j: int = 1,
    gamma: float = 0.5,
    max_lag: int = 1,
    include_warp_tag: bool = False,
    pi_trans: float = 1.0,
) -> float:
    l_rho = log2_lag_prior(lag_j, gamma, max_lag)
    l_tau = log2_warp_tag(pi_trans) if include_warp_tag else 0.0
    total = 0.0
    for (u, v), count in mv_counts.items():
        per_block = l_rho + l_tau + log2_translation_params(u, v, alpha_u, alpha_v, radius)
        total += per_block * count
    return total

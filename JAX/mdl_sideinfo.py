from __future__ import annotations

import jax.numpy as jnp


def truncated_two_sided_geom_norm(alpha: float, radius: int) -> jnp.ndarray:
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    if radius == 0:
        return jnp.array(1.0, dtype=jnp.float32)
    return 1.0 + 2.0 * alpha * (1.0 - alpha**radius) / (1.0 - alpha)


def log2_two_sided_geom(n: jnp.ndarray, alpha: float) -> jnp.ndarray:
    return -jnp.log2((1.0 - alpha) / (1.0 + alpha)) - jnp.abs(n) * jnp.log2(alpha)


def log2_truncated_two_sided_geom(n: jnp.ndarray, alpha: float, radius: int) -> jnp.ndarray:
    z_r = truncated_two_sided_geom_norm(alpha, radius)
    return jnp.log2(z_r) - jnp.abs(n) * jnp.log2(alpha)


def log2_lag_prior(j: jnp.ndarray, gamma: float, max_lag: int) -> jnp.ndarray:
    trunc = 1.0 - gamma**max_lag
    return -jnp.log2(1.0 - gamma) - (j - 1) * jnp.log2(gamma) + jnp.log2(trunc)


def log2_warp_tag(pi_k: float) -> jnp.ndarray:
    return -jnp.log2(pi_k)


def log2_quadtree_prior(internal_count: int, leaf_nonmax_count: int, p_split: float) -> float:
    if internal_count < 0 or leaf_nonmax_count < 0:
        raise ValueError("internal_count and leaf_nonmax_count must be >= 0.")
    return float(-internal_count * jnp.log2(p_split) - leaf_nonmax_count * jnp.log2(1.0 - p_split))


def log2_translation_params(
    u: jnp.ndarray, v: jnp.ndarray, alpha_u: float, alpha_v: float, radius: int
) -> jnp.ndarray:
    return log2_truncated_two_sided_geom(u, alpha_u, radius) + log2_truncated_two_sided_geom(
        v, alpha_v, radius
    )


def log2_similarity_params(
    u: jnp.ndarray,
    v: jnp.ndarray,
    theta_idx: jnp.ndarray,
    scale_idx: jnp.ndarray,
    alpha_u: float,
    alpha_v: float,
    alpha_theta: float,
    alpha_scale: float,
    radius_uv: int,
    radius_theta: int,
    radius_scale: int,
) -> jnp.ndarray:
    return (
        log2_truncated_two_sided_geom(u, alpha_u, radius_uv)
        + log2_truncated_two_sided_geom(v, alpha_v, radius_uv)
        + log2_truncated_two_sided_geom(theta_idx, alpha_theta, radius_theta)
        + log2_truncated_two_sided_geom(scale_idx, alpha_scale, radius_scale)
    )


def log2_affine_params(
    u: jnp.ndarray,
    v: jnp.ndarray,
    a11: jnp.ndarray,
    a12: jnp.ndarray,
    a21: jnp.ndarray,
    a22: jnp.ndarray,
    alpha_u: float,
    alpha_v: float,
    alpha_a11: float,
    alpha_a12: float,
    alpha_a21: float,
    alpha_a22: float,
    radius_uv: int,
    radius_a: int,
) -> jnp.ndarray:
    return (
        log2_truncated_two_sided_geom(u, alpha_u, radius_uv)
        + log2_truncated_two_sided_geom(v, alpha_v, radius_uv)
        + log2_truncated_two_sided_geom(a11, alpha_a11, radius_a)
        + log2_truncated_two_sided_geom(a12, alpha_a12, radius_a)
        + log2_truncated_two_sided_geom(a21, alpha_a21, radius_a)
        + log2_truncated_two_sided_geom(a22, alpha_a22, radius_a)
    )


def motion_translation_side_bits(
    mv_grid: jnp.ndarray,
    alpha_u: float,
    alpha_v: float,
    radius: int,
    lag_j: int = 1,
    gamma: float = 0.5,
    max_lag: int = 1,
    include_warp_tag: bool = False,
    pi_trans: float = 1.0,
) -> jnp.ndarray:
    mv_grid = jnp.asarray(mv_grid)
    u = mv_grid[..., 0]
    v = mv_grid[..., 1]
    l_rho = log2_lag_prior(jnp.array(lag_j, dtype=jnp.float32), gamma, max_lag)
    l_tau = log2_warp_tag(pi_trans) if include_warp_tag else 0.0
    per_block = l_rho + l_tau + log2_translation_params(u, v, alpha_u, alpha_v, radius)
    return jnp.sum(per_block)

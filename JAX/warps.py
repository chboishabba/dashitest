from __future__ import annotations

import jax.numpy as jnp


def _sample_nearest(frame: jnp.ndarray, yy: jnp.ndarray, xx: jnp.ndarray, padding: int = 0) -> jnp.ndarray:
    h, w = frame.shape
    y_int = jnp.rint(yy).astype(jnp.int32)
    x_int = jnp.rint(xx).astype(jnp.int32)
    in_bounds = (y_int >= 0) & (y_int < h) & (x_int >= 0) & (x_int < w)
    y_clip = jnp.clip(y_int, 0, h - 1)
    x_clip = jnp.clip(x_int, 0, w - 1)
    sampled = frame[y_clip, x_clip]
    return jnp.where(in_bounds, sampled, padding)


def translation_block(
    frame: jnp.ndarray, y: int, x: int, block: int, dy: float, dx: float, padding: int = 0
) -> jnp.ndarray:
    yy, xx = jnp.meshgrid(jnp.arange(block), jnp.arange(block), indexing="ij")
    yy = yy + y + dy
    xx = xx + x + dx
    return _sample_nearest(frame, yy, xx, padding=padding)


def affine_block(
    frame: jnp.ndarray,
    y: int,
    x: int,
    block: int,
    matrix: jnp.ndarray,
    offset: jnp.ndarray,
    padding: int = 0,
) -> jnp.ndarray:
    yy, xx = jnp.meshgrid(jnp.arange(block), jnp.arange(block), indexing="ij")
    coords = jnp.stack([yy + y, xx + x], axis=-1).reshape(-1, 2)
    warped = (coords @ matrix.T) + offset
    yy_w = warped[:, 0].reshape(block, block)
    xx_w = warped[:, 1].reshape(block, block)
    return _sample_nearest(frame, yy_w, xx_w, padding=padding)


def similarity_block(
    frame: jnp.ndarray,
    y: int,
    x: int,
    block: int,
    scale: float,
    theta: float,
    offset: jnp.ndarray,
    padding: int = 0,
) -> jnp.ndarray:
    c, s = jnp.cos(theta), jnp.sin(theta)
    matrix = jnp.array([[scale * c, -scale * s], [scale * s, scale * c]], dtype=jnp.float32)
    return affine_block(frame, y, x, block, matrix, offset, padding=padding)

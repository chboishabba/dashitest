from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _block_sad_for_offsets(
    prev: jnp.ndarray,
    cur: jnp.ndarray,
    y: jnp.ndarray,
    x: jnp.ndarray,
    offsets: jnp.ndarray,
    block: int,
) -> jnp.ndarray:
    h, w = prev.shape
    block_cur = jax.lax.dynamic_slice(cur, (y, x), (block, block))

    def sad_for_offset(offset: jnp.ndarray) -> jnp.ndarray:
        dy, dx = offset[0], offset[1]
        y1 = jnp.clip(y + dy, 0, h - block)
        x1 = jnp.clip(x + dx, 0, w - block)
        block_prev = jax.lax.dynamic_slice(prev, (y1, x1), (block, block))
        return jnp.sum(jnp.abs(block_cur - block_prev))

    return jax.vmap(sad_for_offset)(offsets)


def block_match(prev: jnp.ndarray, cur: jnp.ndarray, block: int, search: int) -> jnp.ndarray:
    h, w = prev.shape
    y_positions = jnp.arange(0, h, block)
    x_positions = jnp.arange(0, w, block)
    y_positions = jnp.minimum(y_positions, h - block)
    x_positions = jnp.minimum(x_positions, w - block)
    ys, xs = jnp.meshgrid(y_positions, x_positions, indexing="ij")
    positions = jnp.stack([ys.ravel(), xs.ravel()], axis=-1)

    offsets = jnp.array(
        [(dy, dx) for dy in range(-search, search + 1) for dx in range(-search, search + 1)],
        dtype=jnp.int32,
    )

    def sad_for_pos(pos: jnp.ndarray) -> jnp.ndarray:
        return _block_sad_for_offsets(prev, cur, pos[0], pos[1], offsets, block)

    sads = jax.vmap(sad_for_pos)(positions)
    best_idx = jnp.argmin(sads, axis=1)
    best_offsets = offsets[best_idx]
    mv_grid = best_offsets.reshape(ys.shape[0], ys.shape[1], 2)
    return mv_grid


def motion_compensated_residual(
    frames: np.ndarray, block: int = 8, search: int = 4
) -> Tuple[np.ndarray, Dict[str, object]]:
    t, h, w = frames.shape
    if t <= 1:
        base = frames[0].astype(np.int16) - 128
        mv_counts = {(0, 0): (h // block) * (w // block)}
        return base.ravel(), {"mv_counts": mv_counts, "max_abs": int(np.max(np.abs(base)))}

    residuals = []
    mv_counts: Dict[Tuple[int, int], int] = {}
    max_abs = 0
    for ti in range(t):
        if ti == 0:
            base = frames[0].astype(np.int16) - 128
            residuals.append(base)
            max_abs = max(max_abs, int(np.max(np.abs(base))))
            continue
        prev = jnp.asarray(frames[ti - 1], dtype=jnp.int16)
        cur = jnp.asarray(frames[ti], dtype=jnp.int16)
        mv_grid = block_match(prev, cur, block, search)
        mv_np = np.asarray(mv_grid)
        pred = np.zeros_like(frames[ti], dtype=np.int16)
        for by in range(mv_np.shape[0]):
            for bx in range(mv_np.shape[1]):
                dy, dx = mv_np[by, bx]
                y0 = min(by * block, h - block)
                x0 = min(bx * block, w - block)
                y1 = int(np.clip(y0 + dy, 0, h - block))
                x1 = int(np.clip(x0 + dx, 0, w - block))
                pred[y0 : y0 + block, x0 : x0 + block] = frames[ti - 1][
                    y1 : y1 + block, x1 : x1 + block
                ]
                key = (int(dy), int(dx))
                mv_counts[key] = mv_counts.get(key, 0) + 1
        res = frames[ti].astype(np.int16) - pred
        residuals.append(res)
        max_abs = max(max_abs, int(np.max(np.abs(res))))

    stacked = np.stack(residuals, axis=0)
    return stacked.ravel(), {"mv_counts": mv_counts, "max_abs": max_abs}

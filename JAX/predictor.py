from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import jax.numpy as jnp

from . import warps


@dataclass(frozen=True)
class BlockParam:
    lag: int
    warp_tag: str
    params: Tuple[float, ...]


def predict_frame_from_blocks(
    frames: jnp.ndarray,
    t: int,
    blocks: Iterable[Tuple[int, int, int]],
    params: Iterable[BlockParam],
    padding: int = 0,
) -> jnp.ndarray:
    frame = frames[t]
    pred = jnp.zeros_like(frame)
    for (y, x, block), param in zip(blocks, params):
        ref = frames[t - param.lag]
        if param.warp_tag == "trans":
            dy, dx = param.params
            patch = warps.translation_block(ref, y, x, block, dy, dx, padding=padding)
        elif param.warp_tag == "sim":
            dy, dx, theta, scale = param.params
            offset = jnp.array([dy, dx], dtype=jnp.float32)
            patch = warps.similarity_block(ref, y, x, block, scale, theta, offset, padding=padding)
        elif param.warp_tag == "aff":
            dy, dx, a11, a12, a21, a22 = param.params
            matrix = jnp.array([[a11, a12], [a21, a22]], dtype=jnp.float32)
            offset = jnp.array([dy, dx], dtype=jnp.float32)
            patch = warps.affine_block(ref, y, x, block, matrix, offset, padding=padding)
        else:
            raise ValueError(f"Unknown warp_tag {param.warp_tag}")
        pred = pred.at[y : y + block, x : x + block].set(patch)
    return pred


def predict_video_from_blocks(
    frames: jnp.ndarray,
    blocks: Iterable[Tuple[int, int, int]],
    params_per_frame: Iterable[Iterable[BlockParam]],
    padding: int = 0,
) -> List[jnp.ndarray]:
    preds = []
    for t, params in enumerate(params_per_frame):
        if t == 0:
            preds.append(jnp.zeros_like(frames[0]))
            continue
        preds.append(predict_frame_from_blocks(frames, t, blocks, params, padding=padding))
    return preds

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np


def balanced_digits_needed(values: jnp.ndarray) -> int:
    max_abs = jnp.max(jnp.abs(values)).astype(jnp.int32)
    total = 2 * max_abs + 1
    digits = 1
    while 3**digits < int(total):
        digits += 1
    return digits


def balanced_ternary_digits(values: jnp.ndarray, digits: int) -> jnp.ndarray:
    x = values.astype(jnp.int64).ravel()
    out = []
    for _ in range(digits):
        r = (x % 3).astype(jnp.int8)
        adjust = r == 2
        r = jnp.where(adjust, -1, r)
        x = (x - r) // 3
        out.append(r)
    return jnp.stack(out, axis=0)


def compute_streams(frames: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    frames_u8 = jnp.asarray(frames, dtype=jnp.uint8)
    t, h, w = frames_u8.shape
    raw_stream = frames_u8.reshape(-1)

    if t > 1:
        diffs = (frames_u8[1:].astype(jnp.int16) - frames_u8[:-1].astype(jnp.int16)) & 0xFF
        residual = jnp.concatenate([frames_u8[0].reshape(-1), diffs.reshape(-1).astype(jnp.uint8)])
    else:
        residual = frames_u8[0].reshape(-1)

    base = frames_u8[0].astype(jnp.int16) - 128
    if t > 1:
        diffs_signed = frames_u8[1:].astype(jnp.int16) - frames_u8[:-1].astype(jnp.int16)
        signed_resid = jnp.concatenate([base.reshape(-1), diffs_signed.reshape(-1)])
    else:
        signed_resid = base.reshape(-1)

    coarse = jnp.minimum(raw_stream, 255 - raw_stream).astype(jnp.uint8)
    sign = (raw_stream > 127).astype(jnp.uint8)

    if t > 1:
        coarse_frames = coarse.reshape(t, h, w)
        sign_frames = sign.reshape(t, h, w)
        coarse_resid = jnp.concatenate(
            [coarse_frames[0].reshape(-1), ((coarse_frames[1:] - coarse_frames[:-1]) & 0xFF).reshape(-1)]
        ).astype(jnp.uint8)
        sign_resid = jnp.concatenate(
            [sign_frames[0].reshape(-1), (sign_frames[1:] ^ sign_frames[:-1]).reshape(-1)]
        ).astype(jnp.uint8)
    else:
        coarse_resid = coarse
        sign_resid = sign

    streams = {
        "raw": np.asarray(raw_stream),
        "residual": np.asarray(residual),
        "coarse": np.asarray(coarse),
        "sign": np.asarray(sign),
        "coarse_resid": np.asarray(coarse_resid),
        "sign_resid": np.asarray(sign_resid),
    }
    aux = {"signed_resid": np.asarray(signed_resid)}
    return streams, np.asarray(signed_resid), aux


def compute_bt_planes(
    signed_resid: np.ndarray, frames_shape: Tuple[int, int, int]
) -> Dict[str, np.ndarray]:
    signed_j = jnp.asarray(signed_resid, dtype=jnp.int16)
    bt_digits = balanced_digits_needed(signed_j)
    bt_planes = balanced_ternary_digits(signed_j, bt_digits)
    bt_planes_u8 = (bt_planes + 1).astype(jnp.uint8)
    t, h, w = frames_shape
    bt_planes_i8 = bt_planes.astype(jnp.int8).reshape(bt_digits, t, h, w)
    bt_mag = jnp.abs(bt_planes_i8).astype(jnp.uint8)
    bt_sign = (bt_planes_i8 > 0).astype(jnp.uint8)
    return {
        "bt_digits": np.asarray(bt_digits),
        "bt_planes_u8": np.asarray(bt_planes_u8),
        "bt_planes_i8": np.asarray(bt_planes_i8),
        "bt_mag": np.asarray(bt_mag),
        "bt_sign": np.asarray(bt_sign),
    }

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from . import mdl_sideinfo, rans  # type: ignore
except ImportError:  # direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from compression import mdl_sideinfo, rans  # type: ignore


def ffprobe_video(path: Path) -> Tuple[int, int, int]:
    """Return (width, height, nb_frames|0)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    nb_frames = int(stream["nb_frames"]) if stream.get("nb_frames", "0").isdigit() else 0
    return width, height, nb_frames


def decode_gray(path: Path, width: int, height: int, max_frames: int) -> np.ndarray:
    """Decode video to grayscale raw frames via ffmpeg; returns array [T,H,W] uint8."""
    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-vframes",
        str(max_frames),
        "-loglevel",
        "error",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    frame_size = width * height
    total_pixels = len(raw)
    if total_pixels % frame_size != 0:
        raise ValueError("Decoded byte count not divisible by frame size; ffmpeg decode mismatch.")
    frames = total_pixels // frame_size
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape(frames, height, width)


def decode_rgb(path: Path, width: int, height: int, max_frames: int) -> np.ndarray:
    """Decode video to RGB frames via ffmpeg; returns array [T,H,W,3] uint8."""
    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vframes",
        str(max_frames),
        "-loglevel",
        "error",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    frame_size = width * height * 3
    total_pixels = len(raw)
    if total_pixels % frame_size != 0:
        raise ValueError("Decoded byte count not divisible by frame size; ffmpeg decode mismatch.")
    frames = total_pixels // frame_size
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape(frames, height, width, 3)


def rgb_to_ycocg_r(frames_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reversible YCoCg-R transform with sign+mag for Co/Cg."""
    r = frames_rgb[..., 0].astype(np.int16)
    g = frames_rgb[..., 1].astype(np.int16)
    b = frames_rgb[..., 2].astype(np.int16)
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    y_u8 = np.clip(y, 0, 255).astype(np.uint8)
    co_sign = (co < 0).astype(np.uint8)
    cg_sign = (cg < 0).astype(np.uint8)
    co_mag = np.abs(co).astype(np.uint8)
    cg_mag = np.abs(cg).astype(np.uint8)
    return y_u8, co_mag, co_sign, cg_mag, cg_sign


def _write_pgm(path: Path, img: np.ndarray) -> None:
    h, w = img.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(img.tobytes())


def _write_png(path: Path, img: np.ndarray) -> bool:
    try:
        import imageio.v2 as imageio  # type: ignore

        imageio.imwrite(path, img)
        return True
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.imsave(path, img, cmap="gray", vmin=0, vmax=255)
        return True
    except Exception:
        return False


def _dump_bt_planes(
    out_dir: Path,
    label_prefix: str,
    planes_i8: np.ndarray,
    mag: np.ndarray,
    sign: np.ndarray,
    max_frames: int,
) -> None:
    tag = label_prefix.rstrip("_") or "gray"
    plane_dir = out_dir / f"{tag}_bt_planes"
    plane_dir.mkdir(parents=True, exist_ok=True)
    npz_path = plane_dir / "bt_planes.npz"
    np.savez_compressed(npz_path, planes_i8=planes_i8, mag=mag, sign=sign)

    png_dir = plane_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    lut = np.array([0, 127, 255], dtype=np.uint8)
    frame_limit = planes_i8.shape[1] if max_frames <= 0 else min(max_frames, planes_i8.shape[1])
    png_ok = None
    for pi in range(planes_i8.shape[0]):
        for ti in range(frame_limit):
            img_u8 = lut[planes_i8[pi, ti] + 1]
            out_path = png_dir / f"plane{pi:02d}_t{ti:04d}.png"
            if png_ok is None:
                png_ok = _write_png(out_path, img_u8)
                if not png_ok:
                    _write_pgm(out_path.with_suffix(".pgm"), img_u8)
                    print(f"PNG writer unavailable; wrote PGM to {out_path.with_suffix('.pgm')}")
            elif png_ok:
                _write_png(out_path, img_u8)
            else:
                _write_pgm(out_path.with_suffix(".pgm"), img_u8)


def stream_entropy(symbols: np.ndarray) -> float:
    counts = np.bincount(symbols, minlength=int(symbols.max()) + 1)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0


def _balanced_ternary_digits(values: np.ndarray, digits: int) -> np.ndarray:
    """Return balanced ternary digits in {-1,0,1} with shape (digits, N)."""
    x = values.astype(np.int64).ravel()
    out = np.empty((digits, x.size), dtype=np.int8)
    for k in range(digits):
        r = (x % 3).astype(np.int8)
        adjust = r == 2
        r = r.copy()
        r[adjust] = -1
        x = (x - r) // 3
        out[k] = r
    return out


def _balanced_digits_needed(values: np.ndarray) -> int:
    max_abs = int(np.max(np.abs(values)))
    total = 2 * max_abs + 1
    digits = 1
    while 3**digits < total:
        digits += 1
    return digits


def _downsample_3x(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape
    h3 = (h // 3) * 3
    w3 = (w // 3) * 3
    cropped = frame[:h3, :w3]
    return cropped.reshape(h3 // 3, 3, w3 // 3, 3).mean(axis=(1, 3)).astype(np.int16)


def _motion_compensated_residual(
    frames: np.ndarray, block: int = 8, search: int = 4, pyramid_levels: int = 0
) -> tuple[np.ndarray, dict]:
    """Return signed residuals after blockwise translational motion compensation."""
    t, h, w = frames.shape
    if t <= 1:
        base = frames[0].astype(np.int16) - 128
        return base.ravel(), {"mv_counts": {(0, 0): (h // block) * (w // block)}, "max_abs": int(np.max(np.abs(base)))}

    residuals = []
    mv_counts: dict[tuple[int, int], int] = {}
    max_abs = 0
    for ti in range(t):
        if ti == 0:
            base = frames[0].astype(np.int16) - 128
            residuals.append(base)
            max_abs = max(max_abs, int(np.max(np.abs(base))))
            continue
        prev = frames[ti - 1].astype(np.int16)
        cur = frames[ti].astype(np.int16)
        if pyramid_levels > 0:
            prev_pyr = [prev]
            cur_pyr = [cur]
            for _ in range(pyramid_levels):
                prev_pyr.append(_downsample_3x(prev_pyr[-1]))
                cur_pyr.append(_downsample_3x(cur_pyr[-1]))
        else:
            prev_pyr = [prev]
            cur_pyr = [cur]
        pred = np.zeros_like(cur)
        for y in range(0, h, block):
            for x in range(0, w, block):
                y0 = min(y, h - block)
                x0 = min(x, w - block)
                block_cur = cur[y0 : y0 + block, x0 : x0 + block]
                best = (0, 0)
                best_sad = None
                if pyramid_levels > 0:
                    coarse_h, coarse_w = prev_pyr[-1].shape
                    cy = y0 // (3**pyramid_levels)
                    cx = x0 // (3**pyramid_levels)
                    coarse_block = cur_pyr[-1][cy : cy + max(1, block // (3**pyramid_levels)),
                                                cx : cx + max(1, block // (3**pyramid_levels))]
                    csearch = max(1, search // (3**pyramid_levels))
                    for dy in range(-csearch, csearch + 1):
                        for dx in range(-csearch, csearch + 1):
                            y1 = np.clip(cy + dy, 0, coarse_h - coarse_block.shape[0])
                            x1 = np.clip(cx + dx, 0, coarse_w - coarse_block.shape[1])
                            cand = prev_pyr[-1][y1 : y1 + coarse_block.shape[0], x1 : x1 + coarse_block.shape[1]]
                            sad = int(np.abs(coarse_block - cand).sum())
                            if best_sad is None or sad < best_sad:
                                best_sad = sad
                                best = (dy * (3**pyramid_levels), dx * (3**pyramid_levels))
                    for level in range(pyramid_levels - 1, -1, -1):
                        scale = 3**level
                        yb = y0 // scale
                        xb = x0 // scale
                        block_h = max(1, block // scale)
                        block_w = max(1, block // scale)
                        cur_block = cur_pyr[level][yb : yb + block_h, xb : xb + block_w]
                        search_r = max(1, search // scale)
                        dy0, dx0 = best
                        dy0 //= 3
                        dx0 //= 3
                        best_ref = (dy0, dx0)
                        best_sad = None
                        for dy in range(-search_r, search_r + 1):
                            for dx in range(-search_r, search_r + 1):
                                y1 = np.clip(yb + dy0 + dy, 0, cur_pyr[level].shape[0] - block_h)
                                x1 = np.clip(xb + dx0 + dx, 0, cur_pyr[level].shape[1] - block_w)
                                cand = prev_pyr[level][y1 : y1 + block_h, x1 : x1 + block_w]
                                sad = int(np.abs(cur_block - cand).sum())
                                if best_sad is None or sad < best_sad:
                                    best_sad = sad
                                    best_ref = (dy0 + dy, dx0 + dx)
                        best = (best_ref[0] * scale, best_ref[1] * scale)
                else:
                    for dy in range(-search, search + 1):
                        for dx in range(-search, search + 1):
                            y1 = np.clip(y0 + dy, 0, h - block)
                            x1 = np.clip(x0 + dx, 0, w - block)
                            cand = prev[y1 : y1 + block, x1 : x1 + block]
                            sad = int(np.abs(block_cur - cand).sum())
                            if best_sad is None or sad < best_sad:
                                best_sad = sad
                                best = (dy, dx)
                mv_counts[best] = mv_counts.get(best, 0) + 1
                y1 = np.clip(y0 + best[0], 0, h - block)
                x1 = np.clip(x0 + best[1], 0, w - block)
                pred[y0 : y0 + block, x0 : x0 + block] = prev[y1 : y1 + block, x1 : x1 + block]
        res = cur - pred
        residuals.append(res)
        max_abs = max(max_abs, int(np.max(np.abs(res))))

    stacked = np.stack(residuals, axis=0)
    return stacked.ravel(), {"mv_counts": mv_counts, "max_abs": max_abs}


def _block_reuse_actions_and_mask(
    planes: np.ndarray, block: int = 16, dict_size: int = 256, hash_planes: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return action stream, encode mask, flip witness bits, and reuse refs."""
    if planes.shape[0] < 1:
        return (
            np.array([], dtype=np.int8),
            np.zeros((planes.shape[1], planes.shape[2], planes.shape[3]), dtype=bool),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
        )
    planes_used = max(1, min(hash_planes, planes.shape[0]))
    _, t, h, w = planes.shape
    actions = []
    refs = []
    mask = np.zeros((t, h, w), dtype=bool)
    flips = []
    dict_list: list[int] = []
    dict_map: dict[int, int] = {}
    blocks_y = (h + block - 1) // block
    blocks_x = (w + block - 1) // block
    prev_sig_grid = np.full((blocks_y, blocks_x), -1, dtype=np.int64)

    for ti in range(t):
        for iy, y in enumerate(range(0, h, block)):
            for ix, x in enumerate(range(0, w, block)):
                y0 = min(y, h - block)
                x0 = min(x, w - block)
                sig = 0
                block_sum = 0
                for pi in range(planes_used):
                    blockp = planes[pi, ti, y0 : y0 + block, x0 : x0 + block]
                    block_sum += int(blockp.sum())
                flip = 1 if block_sum < 0 else 0
                for pi in range(planes_used):
                    blockp = planes[pi, ti, y0 : y0 + block, x0 : x0 + block]
                    if flip:
                        blockp = -blockp
                    sig = zlib.crc32(blockp.tobytes(), sig)

                prev_sig = int(prev_sig_grid[iy, ix])
                if prev_sig != -1 and sig == prev_sig:
                    act = 0  # same as prev frame at same position
                    ref = 0
                elif sig in dict_map:
                    act = 1  # reuse from dictionary
                    ref = dict_map[sig]
                else:
                    act = -1  # new
                    ref = 0
                actions.append(act)
                flips.append(flip)
                if act == -1:
                    mask[ti, y0 : y0 + block, x0 : x0 + block] = True
                    dict_list.append(sig)
                    if len(dict_list) > dict_size:
                        dict_list.pop(0)
                        dict_map = {v: i for i, v in enumerate(dict_list)}
                    else:
                        dict_map[sig] = len(dict_list) - 1
                refs.append(ref)
                prev_sig_grid[iy, ix] = sig

    return (
        np.array(actions, dtype=np.int8),
        mask,
        np.array(flips, dtype=np.uint8),
        np.array(refs, dtype=np.uint16),
    )


def _build_context_tables(
    plane: np.ndarray,
    prev_plane: np.ndarray | None,
    alpha: float = 0.5,
    min_count: int = 16,
) -> list[rans.FreqTable]:
    """Build context tables for trits using left/up/prev-frame and optional prev-plane."""
    t, h, w = plane.shape
    contexts = 81  # 3^4
    counts = np.zeros((contexts, 3), dtype=np.int64)

    def idx_from_trits(a: int, b: int, c: int, d: int) -> int:
        return (((a * 3 + b) * 3 + c) * 3 + d)

    for ti in range(t):
        for yi in range(h):
            row = plane[ti, yi]
            up = plane[ti, yi - 1] if yi > 0 else None
            prev_frame = plane[ti - 1, yi] if ti > 0 else None
            prev_p = prev_plane[ti, yi] if prev_plane is not None else None
            for xi in range(w):
                left = row[xi - 1] if xi > 0 else 0
                up_val = up[xi] if up is not None else 0
                prev_val = prev_frame[xi] if prev_frame is not None else 0
                prevp_val = prev_p[xi] if prev_p is not None else 0
                # map {-1,0,1} -> {0,1,2}
                ctx = idx_from_trits(prevp_val + 1, prev_val + 1, up_val + 1, left + 1)
                counts[ctx, row[xi] + 1] += 1

    tables: list[rans.FreqTable] = []
    for ctx in range(contexts):
        freq = counts[ctx].astype(np.float64)
        if freq.sum() < min_count:
            freq = counts.sum(axis=0).astype(np.float64)
        freq += alpha
        total = freq.sum()
        scaled = np.maximum(1, np.floor(freq / total * rans.MAX_TOTAL)).astype(np.int64)
        diff = rans.MAX_TOTAL - int(scaled.sum())
        if diff > 0:
            scaled[:diff] += 1
        elif diff < 0:
            idx = np.where(scaled > 1)[0]
            take = min(-diff, len(idx))
            scaled[idx[:take]] -= 1
        tables.append(rans.FreqTable.from_freqs(scaled.astype(int).tolist()))
    return tables


def _context_entropy_trit(plane: np.ndarray, prev_plane: np.ndarray | None) -> float:
    """Empirical conditional entropy (bits/symbol) for ternary plane contexts."""
    t, h, w = plane.shape
    contexts = 81
    counts = np.zeros((contexts, 3), dtype=np.int64)

    def idx_from_trits(a: int, b: int, c: int, d: int) -> int:
        return (((a * 3 + b) * 3 + c) * 3 + d)

    for ti in range(t):
        for yi in range(h):
            row = plane[ti, yi]
            up = plane[ti, yi - 1] if yi > 0 else None
            prev_frame = plane[ti - 1, yi] if ti > 0 else None
            prev_p = prev_plane[ti, yi] if prev_plane is not None else None
            for xi in range(w):
                left = row[xi - 1] if xi > 0 else 0
                up_val = up[xi] if up is not None else 0
                prev_val = prev_frame[xi] if prev_frame is not None else 0
                prevp_val = prev_p[xi] if prev_p is not None else 0
                ctx = idx_from_trits(prevp_val + 1, prev_val + 1, up_val + 1, left + 1)
                counts[ctx, row[xi] + 1] += 1

    total = counts.sum()
    if total == 0:
        return 0.0
    ent = 0.0
    for ctx in range(contexts):
        cnt = counts[ctx]
        csum = cnt.sum()
        if csum == 0:
            continue
        probs = cnt[cnt > 0] / csum
        ent -= (csum / total) * float((probs * np.log2(probs)).sum())
    return ent


def _encode_plane_contexted(
    plane: np.ndarray,
    prev_plane: np.ndarray | None,
    tables: list[rans.FreqTable],
) -> bytes:
    t, h, w = plane.shape
    enc = rans.RangeEncoder()

    def idx_from_trits(a: int, b: int, c: int, d: int) -> int:
        return (((a * 3 + b) * 3 + c) * 3 + d)

    for ti in range(t):
        for yi in range(h):
            row = plane[ti, yi]
            up = plane[ti, yi - 1] if yi > 0 else None
            prev_frame = plane[ti - 1, yi] if ti > 0 else None
            prev_p = prev_plane[ti, yi] if prev_plane is not None else None
            for xi in range(w):
                left = row[xi - 1] if xi > 0 else 0
                up_val = up[xi] if up is not None else 0
                prev_val = prev_frame[xi] if prev_frame is not None else 0
                prevp_val = prev_p[xi] if prev_p is not None else 0
                ctx = idx_from_trits(prevp_val + 1, prev_val + 1, up_val + 1, left + 1)
                enc.encode(int(row[xi] + 1), tables[ctx])
    return enc.finish()


def _build_binary_context_tables(
    plane: np.ndarray,
    prev_plane: np.ndarray | None,
    alpha: float = 0.5,
    min_count: int = 16,
) -> list[rans.FreqTable]:
    """Build context tables for binary symbols using left/up/prev-frame and optional prev-plane."""
    t, h, w = plane.shape
    contexts = 16  # 2^4
    counts = np.zeros((contexts, 2), dtype=np.int64)

    def idx_from_bits(a: int, b: int, c: int, d: int) -> int:
        return (((a << 1) | b) << 2) | (c << 1) | d

    for ti in range(t):
        for yi in range(h):
            row = plane[ti, yi]
            up = plane[ti, yi - 1] if yi > 0 else None
            prev_frame = plane[ti - 1, yi] if ti > 0 else None
            prev_p = prev_plane[ti, yi] if prev_plane is not None else None
            for xi in range(w):
                left = int(row[xi - 1]) if xi > 0 else 0
                up_val = int(up[xi]) if up is not None else 0
                prev_val = int(prev_frame[xi]) if prev_frame is not None else 0
                prevp_val = int(prev_p[xi]) if prev_p is not None else 0
                ctx = idx_from_bits(prevp_val, prev_val, up_val, left)
                counts[ctx, int(row[xi])] += 1

    tables: list[rans.FreqTable] = []
    for ctx in range(contexts):
        freq = counts[ctx].astype(np.float64)
        if freq.sum() < min_count:
            freq = counts.sum(axis=0).astype(np.float64)
        freq += alpha
        total = freq.sum()
        scaled = np.maximum(1, np.floor(freq / total * rans.MAX_TOTAL)).astype(np.int64)
        diff = rans.MAX_TOTAL - int(scaled.sum())
        if diff > 0:
            scaled[:diff] += 1
        elif diff < 0:
            idx = np.where(scaled > 1)[0]
            take = min(-diff, len(idx))
            scaled[idx[:take]] -= 1
        tables.append(rans.FreqTable.from_freqs(scaled.astype(int).tolist()))
    return tables


def _context_entropy_bin(plane: np.ndarray, prev_plane: np.ndarray | None) -> float:
    """Empirical conditional entropy (bits/symbol) for binary plane contexts."""
    t, h, w = plane.shape
    contexts = 16
    counts = np.zeros((contexts, 2), dtype=np.int64)

    def idx_from_bits(a: int, b: int, c: int, d: int) -> int:
        return (((a << 1) | b) << 2) | (c << 1) | d

    for ti in range(t):
        for yi in range(h):
            row = plane[ti, yi]
            up = plane[ti, yi - 1] if yi > 0 else None
            prev_frame = plane[ti - 1, yi] if ti > 0 else None
            prev_p = prev_plane[ti, yi] if prev_plane is not None else None
            for xi in range(w):
                left = int(row[xi - 1]) if xi > 0 else 0
                up_val = int(up[xi]) if up is not None else 0
                prev_val = int(prev_frame[xi]) if prev_frame is not None else 0
                prevp_val = int(prev_p[xi]) if prev_p is not None else 0
                ctx = idx_from_bits(prevp_val, prev_val, up_val, left)
                counts[ctx, int(row[xi])] += 1

    total = counts.sum()
    if total == 0:
        return 0.0
    ent = 0.0
    for ctx in range(contexts):
        cnt = counts[ctx]
        csum = cnt.sum()
        if csum == 0:
            continue
        probs = cnt[cnt > 0] / csum
        ent -= (csum / total) * float((probs * np.log2(probs)).sum())
    return ent


def _encode_binary_plane_contexted(
    plane: np.ndarray,
    prev_plane: np.ndarray | None,
    tables: list[rans.FreqTable],
) -> bytes:
    t, h, w = plane.shape
    enc = rans.RangeEncoder()

    def idx_from_bits(a: int, b: int, c: int, d: int) -> int:
        return (((a << 1) | b) << 2) | (c << 1) | d

    for ti in range(t):
        for yi in range(h):
            row = plane[ti, yi]
            up = plane[ti, yi - 1] if yi > 0 else None
            prev_frame = plane[ti - 1, yi] if ti > 0 else None
            prev_p = prev_plane[ti, yi] if prev_plane is not None else None
            for xi in range(w):
                left = int(row[xi - 1]) if xi > 0 else 0
                up_val = int(up[xi]) if up is not None else 0
                prev_val = int(prev_frame[xi]) if prev_frame is not None else 0
                prevp_val = int(prev_p[xi]) if prev_p is not None else 0
                ctx = idx_from_bits(prevp_val, prev_val, up_val, left)
                enc.encode(int(row[xi]), tables[ctx])
    return enc.finish()


def _build_gated_sign_tables(
    mag: np.ndarray,
    sign: np.ndarray,
    prev_sign: np.ndarray | None,
    alpha: float = 0.5,
    min_count: int = 16,
) -> list[rans.FreqTable]:
    """Build context tables for gated sign bits using sign + mag neighbors and prev-plane sign."""
    t, h, w = sign.shape
    contexts = 128  # 2^7
    counts = np.zeros((contexts, 2), dtype=np.int64)

    def idx_from_bits(a: int, b: int, c: int, d: int, e: int, f: int, g: int) -> int:
        return (((((((a << 1) | b) << 1) | c) << 1 | d) << 1 | e) << 1 | f) << 1 | g

    for ti in range(t):
        for yi in range(h):
            row = sign[ti, yi]
            mag_row = mag[ti, yi]
            up = sign[ti, yi - 1] if yi > 0 else None
            up_mag = mag[ti, yi - 1] if yi > 0 else None
            prev_frame = sign[ti - 1, yi] if ti > 0 else None
            prev_mag = mag[ti - 1, yi] if ti > 0 else None
            prev_p = prev_sign[ti, yi] if prev_sign is not None else None
            for xi in range(w):
                if mag_row[xi] == 0:
                    continue
                left = int(row[xi - 1]) if xi > 0 else 0
                up_val = int(up[xi]) if up is not None else 0
                prev_val = int(prev_frame[xi]) if prev_frame is not None else 0
                prevp_val = int(prev_p[xi]) if prev_p is not None else 0
                left_mag = int(mag_row[xi - 1]) if xi > 0 else 0
                up_mag_val = int(up_mag[xi]) if up_mag is not None else 0
                prev_mag_val = int(prev_mag[xi]) if prev_mag is not None else 0
                ctx = idx_from_bits(prevp_val, prev_val, up_val, left, prev_mag_val, up_mag_val, left_mag)
                counts[ctx, int(row[xi])] += 1

    tables: list[rans.FreqTable] = []
    for ctx in range(contexts):
        freq = counts[ctx].astype(np.float64)
        if freq.sum() < min_count:
            freq = counts.sum(axis=0).astype(np.float64)
        freq += alpha
        total = freq.sum()
        scaled = np.maximum(1, np.floor(freq / total * rans.MAX_TOTAL)).astype(np.int64)
        diff = rans.MAX_TOTAL - int(scaled.sum())
        if diff > 0:
            scaled[:diff] += 1
        elif diff < 0:
            idx = np.where(scaled > 1)[0]
            take = min(-diff, len(idx))
            scaled[idx[:take]] -= 1
        tables.append(rans.FreqTable.from_freqs(scaled.astype(int).tolist()))
    return tables


def _context_entropy_sign(
    mag: np.ndarray,
    sign: np.ndarray,
    prev_sign: np.ndarray | None,
) -> float:
    """Empirical conditional entropy (bits/symbol) for gated sign contexts."""
    t, h, w = sign.shape
    contexts = 128
    counts = np.zeros((contexts, 2), dtype=np.int64)

    def idx_from_bits(a: int, b: int, c: int, d: int, e: int, f: int, g: int) -> int:
        return (((((((a << 1) | b) << 1) | c) << 1 | d) << 1 | e) << 1 | f) << 1 | g

    for ti in range(t):
        for yi in range(h):
            row = sign[ti, yi]
            mag_row = mag[ti, yi]
            up = sign[ti, yi - 1] if yi > 0 else None
            up_mag = mag[ti, yi - 1] if yi > 0 else None
            prev_frame = sign[ti - 1, yi] if ti > 0 else None
            prev_mag = mag[ti - 1, yi] if ti > 0 else None
            prev_p = prev_sign[ti, yi] if prev_sign is not None else None
            for xi in range(w):
                if mag_row[xi] == 0:
                    continue
                left = int(row[xi - 1]) if xi > 0 else 0
                up_val = int(up[xi]) if up is not None else 0
                prev_val = int(prev_frame[xi]) if prev_frame is not None else 0
                prevp_val = int(prev_p[xi]) if prev_p is not None else 0
                left_mag = int(mag_row[xi - 1]) if xi > 0 else 0
                up_mag_val = int(up_mag[xi]) if up_mag is not None else 0
                prev_mag_val = int(prev_mag[xi]) if prev_mag is not None else 0
                ctx = idx_from_bits(prevp_val, prev_val, up_val, left, prev_mag_val, up_mag_val, left_mag)
                counts[ctx, int(row[xi])] += 1

    total = counts.sum()
    if total == 0:
        return 0.0
    ent = 0.0
    for ctx in range(contexts):
        cnt = counts[ctx]
        csum = cnt.sum()
        if csum == 0:
            continue
        probs = cnt[cnt > 0] / csum
        ent -= (csum / total) * float((probs * np.log2(probs)).sum())
    return ent


def _encode_gated_sign_contexted(
    mag: np.ndarray,
    sign: np.ndarray,
    prev_sign: np.ndarray | None,
    tables: list[rans.FreqTable],
) -> bytes:
    t, h, w = sign.shape
    enc = rans.RangeEncoder()

    def idx_from_bits(a: int, b: int, c: int, d: int, e: int, f: int, g: int) -> int:
        return (((((((a << 1) | b) << 1) | c) << 1 | d) << 1 | e) << 1 | f) << 1 | g

    for ti in range(t):
        for yi in range(h):
            row = sign[ti, yi]
            mag_row = mag[ti, yi]
            up = sign[ti, yi - 1] if yi > 0 else None
            up_mag = mag[ti, yi - 1] if yi > 0 else None
            prev_frame = sign[ti - 1, yi] if ti > 0 else None
            prev_mag = mag[ti - 1, yi] if ti > 0 else None
            prev_p = prev_sign[ti, yi] if prev_sign is not None else None
            for xi in range(w):
                if mag_row[xi] == 0:
                    continue
                left = int(row[xi - 1]) if xi > 0 else 0
                up_val = int(up[xi]) if up is not None else 0
                prev_val = int(prev_frame[xi]) if prev_frame is not None else 0
                prevp_val = int(prev_p[xi]) if prev_p is not None else 0
                left_mag = int(mag_row[xi - 1]) if xi > 0 else 0
                up_mag_val = int(up_mag[xi]) if up_mag is not None else 0
                prev_mag_val = int(prev_mag[xi]) if prev_mag is not None else 0
                ctx = idx_from_bits(prevp_val, prev_val, up_val, left, prev_mag_val, up_mag_val, left_mag)
                enc.encode(int(row[xi]), tables[ctx])
    return enc.finish()


def _encode_plane_contexted_range(
    plane: np.ndarray,
    prev_plane: np.ndarray | None,
    tables: list[rans.FreqTable],
    start_t: int,
) -> bytes:
    t, h, w = plane.shape
    enc = rans.RangeEncoder()

    def idx_from_trits(a: int, b: int, c: int, d: int) -> int:
        return (((a * 3 + b) * 3 + c) * 3 + d)

    for ti in range(start_t, t):
        for yi in range(h):
            row = plane[ti, yi]
            up = plane[ti, yi - 1] if yi > 0 else None
            prev_frame = plane[ti - 1, yi] if ti > 0 else None
            prev_p = prev_plane[ti, yi] if prev_plane is not None else None
            for xi in range(w):
                left = row[xi - 1] if xi > 0 else 0
                up_val = up[xi] if up is not None else 0
                prev_val = prev_frame[xi] if prev_frame is not None else 0
                prevp_val = prev_p[xi] if prev_p is not None else 0
                ctx = idx_from_trits(prevp_val + 1, prev_val + 1, up_val + 1, left + 1)
                enc.encode(int(row[xi] + 1), tables[ctx])
    return enc.finish()


def _encode_binary_plane_contexted_range(
    plane: np.ndarray,
    prev_plane: np.ndarray | None,
    tables: list[rans.FreqTable],
    start_t: int,
) -> bytes:
    t, h, w = plane.shape
    enc = rans.RangeEncoder()

    def idx_from_bits(a: int, b: int, c: int, d: int) -> int:
        return (((a << 1) | b) << 2) | (c << 1) | d

    for ti in range(start_t, t):
        for yi in range(h):
            row = plane[ti, yi]
            up = plane[ti, yi - 1] if yi > 0 else None
            prev_frame = plane[ti - 1, yi] if ti > 0 else None
            prev_p = prev_plane[ti, yi] if prev_plane is not None else None
            for xi in range(w):
                left = int(row[xi - 1]) if xi > 0 else 0
                up_val = int(up[xi]) if up is not None else 0
                prev_val = int(prev_frame[xi]) if prev_frame is not None else 0
                prevp_val = int(prev_p[xi]) if prev_p is not None else 0
                ctx = idx_from_bits(prevp_val, prev_val, up_val, left)
                enc.encode(int(row[xi]), tables[ctx])
    return enc.finish()


def _encode_gated_sign_contexted_range(
    mag: np.ndarray,
    sign: np.ndarray,
    prev_sign: np.ndarray | None,
    tables: list[rans.FreqTable],
    start_t: int,
) -> bytes:
    t, h, w = sign.shape
    enc = rans.RangeEncoder()

    def idx_from_bits(a: int, b: int, c: int, d: int, e: int, f: int, g: int) -> int:
        return (((((((a << 1) | b) << 1) | c) << 1 | d) << 1 | e) << 1 | f) << 1 | g

    for ti in range(start_t, t):
        for yi in range(h):
            row = sign[ti, yi]
            mag_row = mag[ti, yi]
            up = sign[ti, yi - 1] if yi > 0 else None
            up_mag = mag[ti, yi - 1] if yi > 0 else None
            prev_frame = sign[ti - 1, yi] if ti > 0 else None
            prev_mag = mag[ti - 1, yi] if ti > 0 else None
            prev_p = prev_sign[ti, yi] if prev_sign is not None else None
            for xi in range(w):
                if mag_row[xi] == 0:
                    continue
                left = int(row[xi - 1]) if xi > 0 else 0
                up_val = int(up[xi]) if up is not None else 0
                prev_val = int(prev_frame[xi]) if prev_frame is not None else 0
                prevp_val = int(prev_p[xi]) if prev_p is not None else 0
                left_mag = int(mag_row[xi - 1]) if xi > 0 else 0
                up_mag_val = int(up_mag[xi]) if up_mag is not None else 0
                prev_mag_val = int(prev_mag[xi]) if prev_mag is not None else 0
                ctx = idx_from_bits(prevp_val, prev_val, up_val, left, prev_mag_val, up_mag_val, left_mag)
                enc.encode(int(row[xi]), tables[ctx])
    return enc.finish()

def compress_stats(name: str, payload: bytes) -> Dict[str, float]:
    import gzip
    import lzma
    import zlib

    stats = {}
    start = time.perf_counter()
    stats["lzma"] = len(lzma.compress(payload, preset=6))
    stats["lzma_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    stats["gzip"] = len(gzip.compress(payload, compresslevel=6))
    stats["gzip_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    stats["zlib"] = len(zlib.compress(payload, level=6))
    stats["zlib_ms"] = (time.perf_counter() - start) * 1000.0
    return stats


def _report_triadic(
    label: str,
    signed_resid: np.ndarray,
    frames: np.ndarray,
    height: int,
    width: int,
    pixels: int,
    train_split: float,
    block_reuse: bool,
    reuse_block: int,
    reuse_dict: int,
    reuse_planes: int,
    bt_planes_u8: np.ndarray | None = None,
    bt_planes_i8: np.ndarray | None = None,
    bt_mag: np.ndarray | None = None,
    bt_sign: np.ndarray | None = None,
    bt_digits_override: int | None = None,
) -> dict:
    if bt_planes_u8 is None or bt_planes_i8 is None or bt_mag is None or bt_sign is None:
        bt_digits = _balanced_digits_needed(signed_resid)
        bt_planes = _balanced_ternary_digits(signed_resid, bt_digits)
        bt_planes_u8 = (bt_planes + 1).astype(np.uint8)  # map {-1,0,1} -> {0,1,2}
        bt_planes_i8 = bt_planes.astype(np.int8).reshape(bt_digits, frames.shape[0], height, width)
        bt_mag = np.abs(bt_planes_i8).astype(np.uint8)
        bt_sign = (bt_planes_i8 > 0).astype(np.uint8)
    else:
        bt_digits = int(bt_digits_override if bt_digits_override is not None else bt_planes_u8.shape[0])

    train_frames = max(1, int(frames.shape[0] * train_split))

    print(f"\n{label} balanced ternary digits: {bt_digits} planes")
    plane_bytes = []
    ctx_plane_bytes = []
    ctx_test_bytes = []
    ctx_entropies = []
    for idx in range(bt_digits):
        plane = bt_planes_u8[idx]
        ent = stream_entropy(plane)
        ctx_ent = _context_entropy_trit(bt_planes_i8[idx], bt_planes_i8[idx - 1] if idx > 0 else None)
        ctx_entropies.append(ctx_ent)
        start = time.perf_counter()
        enc = rans.encode(plane, alphabet=3)
        ms = (time.perf_counter() - start) * 1000.0
        plane_bytes.append(len(enc))
        bpc = (len(enc) * 8) / pixels
        print(
            f"bt_plane{idx:<2} entropy={ent:6.3f}  "
            f"rANS {len(enc):8d} ({bpc:5.3f} bpc, {ms:5.1f} ms)"
        )
        ctx_start = time.perf_counter()
        prev_plane = bt_planes_i8[idx - 1] if idx > 0 else None
        tables = _build_context_tables(bt_planes_i8[idx], prev_plane)
        ctx_enc = _encode_plane_contexted(bt_planes_i8[idx], prev_plane, tables)
        ctx_ms = (time.perf_counter() - ctx_start) * 1000.0
        ctx_plane_bytes.append(len(ctx_enc))
        ctx_bpc = (len(ctx_enc) * 8) / pixels
        print(
            f"bt_plane{idx:<2} ctx_rANS {len(ctx_enc):8d} ({ctx_bpc:5.3f} bpc, {ctx_ms:5.1f} ms)"
        )
        if train_frames < frames.shape[0]:
            train_tables = _build_context_tables(bt_planes_i8[idx][:train_frames], prev_plane[:train_frames] if prev_plane is not None else None)
            test_enc = _encode_plane_contexted_range(bt_planes_i8[idx], prev_plane, train_tables, train_frames)
            ctx_test_bytes.append(len(test_enc))

    total_bt = sum(plane_bytes)
    bpc_bt = (total_bt * 8) / pixels
    print(f"\n{label} multistream (balanced ternary planes via rANS): {total_bt} bytes ({bpc_bt:5.3f} bpc)")
    total_ctx = sum(ctx_plane_bytes)
    bpc_ctx = (total_ctx * 8) / pixels
    print(f"{label} multistream (balanced ternary planes ctx_rANS): {total_ctx} bytes ({bpc_ctx:5.3f} bpc)")
    if ctx_test_bytes:
        total_test = sum(ctx_test_bytes)
        test_pixels = pixels * (frames.shape[0] - train_frames) / frames.shape[0]
        bpc_test = (total_test * 8) / max(1.0, test_pixels)
        print(f"{label} multistream (balanced ternary planes ctx_rANS test-only): {total_test} bytes ({bpc_test:5.3f} bpc)")

    # Per-plane Z2 quotient: magnitude plane + gated sign stream
    print(f"\n{label} balanced ternary plane quotient (mag + gated sign)")
    mag_bytes = []
    mag_ctx_bytes = []
    sign_bytes = []
    sign_ctx_bytes = []
    sign_ctx_test_bytes = []
    mag_ctx_entropies = []
    sign_ctx_entropies = []
    for idx in range(bt_digits):
        mag = bt_mag[idx]
        sign = bt_sign[idx]
        sign_mask = mag.astype(bool)
        sign_stream = sign[sign_mask]

        mag_ent = stream_entropy(mag.ravel())
        start = time.perf_counter()
        mag_enc = rans.encode(mag.ravel(), alphabet=2)
        mag_ms = (time.perf_counter() - start) * 1000.0
        mag_bytes.append(len(mag_enc))
        mag_bpc = (len(mag_enc) * 8) / pixels

        ctx_start = time.perf_counter()
        prev_mag = bt_mag[idx - 1] if idx > 0 else None
        mag_tables = _build_binary_context_tables(mag, prev_mag)
        mag_ctx_enc = _encode_binary_plane_contexted(mag, prev_mag, mag_tables)
        mag_ctx_ms = (time.perf_counter() - ctx_start) * 1000.0
        mag_ctx_bytes.append(len(mag_ctx_enc))
        mag_ctx_bpc = (len(mag_ctx_enc) * 8) / pixels
        mag_ctx_entropies.append(_context_entropy_bin(mag, prev_mag))

        sign_ent = stream_entropy(sign_stream.astype(np.uint8)) if sign_stream.size else 0.0
        start = time.perf_counter()
        sign_enc = rans.encode(sign_stream.astype(np.uint8), alphabet=2)
        sign_ms = (time.perf_counter() - start) * 1000.0
        sign_bytes.append(len(sign_enc))
        sign_bpc = (len(sign_enc) * 8) / pixels

        ctx_sign_ms = 0.0
        if sign_stream.size:
            ctx_start = time.perf_counter()
            prev_sign = bt_sign[idx - 1] if idx > 0 else None
            sign_tables = _build_gated_sign_tables(mag, sign, prev_sign)
            sign_ctx_enc = _encode_gated_sign_contexted(mag, sign, prev_sign, sign_tables)
            ctx_sign_ms = (time.perf_counter() - ctx_start) * 1000.0
            sign_ctx_bytes.append(len(sign_ctx_enc))
            sign_ctx_entropies.append(_context_entropy_sign(mag, sign, prev_sign))
        else:
            sign_ctx_bytes.append(0)
            sign_ctx_entropies.append(0.0)
        sign_ctx_bpc = (sign_ctx_bytes[-1] * 8) / pixels

        if train_frames < frames.shape[0] and sign_stream.size:
            prev_sign = bt_sign[idx - 1] if idx > 0 else None
            train_tables = _build_gated_sign_tables(
                mag[:train_frames],
                sign[:train_frames],
                prev_sign[:train_frames] if prev_sign is not None else None,
            )
            test_enc = _encode_gated_sign_contexted_range(
                mag,
                sign,
                prev_sign,
                train_tables,
                train_frames,
            )
            sign_ctx_test_bytes.append(len(test_enc))

        print(
            f"bt_plane{idx:<2} mag_ent={mag_ent:5.3f} "
            f"rANS {len(mag_enc):7d} ({mag_bpc:5.3f} bpc, {mag_ms:5.1f} ms)  "
            f"ctx {len(mag_ctx_enc):7d} ({mag_ctx_bpc:5.3f} bpc, {mag_ctx_ms:5.1f} ms)  "
            f"sign_ent={sign_ent:5.3f} "
            f"sign_rANS {len(sign_enc):7d} ({sign_bpc:5.3f} bpc, {sign_ms:5.1f} ms)  "
            f"sign_ctx {sign_ctx_bytes[-1]:7d} ({sign_ctx_bpc:5.3f} bpc, {ctx_sign_ms:5.1f} ms)"
        )

    total_mag = sum(mag_bytes)
    total_mag_ctx = sum(mag_ctx_bytes)
    total_sign = sum(sign_bytes)
    total_sign_ctx = sum(sign_ctx_bytes)
    total_q = total_mag + total_sign
    total_q_ctx = total_mag_ctx + total_sign
    total_q_ctx_sign = total_mag_ctx + total_sign_ctx
    print(f"\n{label} multistream (bt mag + sign via rANS): {total_q} bytes ({(total_q * 8) / pixels:5.3f} bpc)")
    print(f"{label} multistream (bt mag ctx + sign via rANS): {total_q_ctx} bytes ({(total_q_ctx * 8) / pixels:5.3f} bpc)")
    bpc_mag_sign = (total_q_ctx_sign * 8) / pixels
    print(f"{label} multistream (bt mag ctx + sign ctx via rANS): {total_q_ctx_sign} bytes ({bpc_mag_sign:5.3f} bpc)")
    if sign_ctx_test_bytes:
        total_sign_ctx_test = sum(sign_ctx_test_bytes)
        total_q_ctx_sign_test = total_mag_ctx + total_sign_ctx_test
        test_pixels = pixels * (frames.shape[0] - train_frames) / frames.shape[0]
        bpc_test = (total_q_ctx_sign_test * 8) / max(1.0, test_pixels)
        print(f"{label} multistream (bt mag ctx + sign ctx test-only): {total_q_ctx_sign_test} bytes ({bpc_test:5.3f} bpc)")

    block_reuse_bpc = None
    if block_reuse:
        actions, mask, flips, refs = _block_reuse_actions_and_mask(
            bt_planes_i8, block=reuse_block, dict_size=reuse_dict, hash_planes=reuse_planes
        )
        if actions.size:
            action_syms = (actions + 1).astype(np.uint8)
            ent = stream_entropy(action_syms)
            start = time.perf_counter()
            enc = rans.encode(action_syms, alphabet=3)
            ms = (time.perf_counter() - start) * 1000.0
            bpb = (len(enc) * 8) / max(1, action_syms.size)
            new_count = int((actions == -1).sum())
            same_count = int((actions == 0).sum())
            reuse_count = int((actions == 1).sum())
            print(
                f"{label} block_action entropy={ent:6.3f}  rANS {len(enc):8d} "
                f"({bpb:5.3f} bpb, {ms:5.1f} ms)  "
                f"new={new_count} same={same_count} reuse={reuse_count}"
            )

            masked_ctx_bytes = []
            for idx in range(bt_digits):
                plane = bt_planes_i8[idx].copy()
                plane[~mask] = 0
                prev_plane = bt_planes_i8[idx - 1].copy() if idx > 0 else None
                if prev_plane is not None:
                    prev_plane[~mask] = 0
                tables = _build_context_tables(plane, prev_plane)
                enc_plane = _encode_plane_contexted(plane, prev_plane, tables)
                masked_ctx_bytes.append(len(enc_plane))
            ref_bytes = 0
            if reuse_count:
                ref_stream = refs[actions == 1]
                if reuse_dict <= 256:
                    ref_bytes = len(rans.encode(ref_stream.astype(np.uint8), alphabet=reuse_dict))
                else:
                    ref_lo = (ref_stream % 256).astype(np.uint8)
                    ref_hi = (ref_stream // 256).astype(np.uint8)
                    hi_alphabet = int(ref_hi.max()) + 1 if ref_hi.size else 1
                    ref_bytes = len(rans.encode(ref_lo, alphabet=256))
                    ref_bytes += len(rans.encode(ref_hi, alphabet=max(2, hi_alphabet)))
            flip_bytes = 0
            if flips.size:
                flip_stream = flips[actions != -1]
                flip_bytes = len(rans.encode(flip_stream.astype(np.uint8), alphabet=2))
            total_masked_ctx = sum(masked_ctx_bytes) + len(enc) + ref_bytes + flip_bytes
            bpc_masked = (total_masked_ctx * 8) / pixels
            print(
                f"{label} block_reuse ctx_rANS (masked planes + actions + refs + flips): "
                f"{total_masked_ctx} bytes ({bpc_masked:5.3f} bpc)  "
                f"action={len(enc)} ref={ref_bytes} flip={flip_bytes} planes={sum(masked_ctx_bytes)}"
            )
            block_reuse_bpc = bpc_masked

    # Shannon efficiency summary (context-conditional entropy vs coded length)
    h_ctx_bits = 0.0
    for idx, h in enumerate(ctx_entropies):
        h_ctx_bits += h * bt_planes_u8[idx].size
    l_ctx_bits = total_ctx * 8
    eta_ctx = (h_ctx_bits / l_ctx_bits) if l_ctx_bits else 0.0
    print(f"{label} eta_ctx_trit={eta_ctx:5.3f} (H={h_ctx_bits:.1f} bits, L={l_ctx_bits:.1f} bits)")

    h_mag_bits = 0.0
    h_sign_bits = 0.0
    for idx, h in enumerate(mag_ctx_entropies):
        h_mag_bits += h * bt_mag[idx].size
    for idx, h in enumerate(sign_ctx_entropies):
        # gated signs only where mag != 0
        h_sign_bits += h * int(bt_mag[idx].sum())
    l_mag_bits = sum(mag_ctx_bytes) * 8
    l_sign_bits = sum(sign_ctx_bytes) * 8
    eta_mag = (h_mag_bits / l_mag_bits) if l_mag_bits else 0.0
    eta_sign = (h_sign_bits / l_sign_bits) if l_sign_bits else 0.0
    eta_mdl = ((h_mag_bits + h_sign_bits) / (l_mag_bits + l_sign_bits)) if (l_mag_bits + l_sign_bits) else 0.0
    print(
        f"{label} eta_mag_ctx={eta_mag:5.3f} eta_sign_ctx={eta_sign:5.3f} "
        f"eta_MDL={eta_mdl:5.3f}"
    )
    return {
        "ctx_trit_bpc": bpc_ctx,
        "mag_sign_bpc": bpc_mag_sign,
        "block_reuse_bpc": block_reuse_bpc,
    }


def run_video_bench(
    path: Path,
    max_frames: int,
    mc: bool,
    jax_mc: bool,
    jax_pipeline: bool,
    mc_block: int,
    mc_search: int,
    mc_pyramid: int,
    train_split: float,
    block_reuse: bool,
    reuse_block: int,
    reuse_dict: int,
    reuse_planes: int,
    color: bool,
    color_transform: str,
    dump_planes_dir: Path | None,
    dump_planes_frames: int,
) -> None:
    width, height, nb_frames = ffprobe_video(path)
    if color:
        frames_rgb = decode_rgb(path, width, height, max_frames)
        actual_frames = frames_rgb.shape[0]
        pixels = frames_rgb.shape[0] * frames_rgb.shape[1] * frames_rgb.shape[2]
        print(f"Video: {path.name} | {width}x{height} | frames decoded: {actual_frames} (probe reported {nb_frames})")
        print(f"Original file bytes: {os.path.getsize(path)}")
        print(f"Total pixels per channel: {pixels}")
        print("color mode: per-channel bpc (sum R+G+B for bpp)")
        print()

        if color_transform == "ycocg":
            y, co_mag, co_sign, cg_mag, cg_sign = rgb_to_ycocg_r(frames_rgb)
            print("--- channel y ---")
            y_summary = _bench_frames(
                y,
                label_prefix="y_",
                width=width,
                height=height,
                pixels=pixels,
                mc=mc,
                jax_mc=jax_mc,
                jax_pipeline=jax_pipeline,
                mc_block=mc_block,
                mc_search=mc_search,
                mc_pyramid=mc_pyramid,
                train_split=train_split,
                block_reuse=block_reuse,
                reuse_block=reuse_block,
                reuse_dict=reuse_dict,
                reuse_planes=reuse_planes,
                dump_planes_dir=dump_planes_dir,
                dump_planes_frames=dump_planes_frames,
            )
            print("--- channel co_mag ---")
            co_summary = _bench_frames(
                co_mag,
                label_prefix="co_",
                width=width,
                height=height,
                pixels=pixels,
                mc=mc,
                jax_mc=jax_mc,
                jax_pipeline=jax_pipeline,
                mc_block=mc_block,
                mc_search=mc_search,
                mc_pyramid=mc_pyramid,
                train_split=train_split,
                block_reuse=block_reuse,
                reuse_block=reuse_block,
                reuse_dict=reuse_dict,
                reuse_planes=reuse_planes,
                dump_planes_dir=dump_planes_dir,
                dump_planes_frames=dump_planes_frames,
            )
            print("--- channel cg_mag ---")
            cg_summary = _bench_frames(
                cg_mag,
                label_prefix="cg_",
                width=width,
                height=height,
                pixels=pixels,
                mc=mc,
                jax_mc=jax_mc,
                jax_pipeline=jax_pipeline,
                mc_block=mc_block,
                mc_search=mc_search,
                mc_pyramid=mc_pyramid,
                train_split=train_split,
                block_reuse=block_reuse,
                reuse_block=reuse_block,
                reuse_dict=reuse_dict,
                reuse_planes=reuse_planes,
                dump_planes_dir=dump_planes_dir,
                dump_planes_frames=dump_planes_frames,
            )
            co_sign_bytes = len(rans.encode(co_sign.ravel(), alphabet=2))
            cg_sign_bytes = len(rans.encode(cg_sign.ravel(), alphabet=2))
            sign_bpp = (co_sign_bytes + cg_sign_bytes) * 8 / pixels
            print(
                f"ycocg sign streams: co={co_sign_bytes} bytes cg={cg_sign_bytes} bytes "
                f"(sign bpp={sign_bpp:5.3f})"
            )
            total_bpp = 0.0
            for summary in (y_summary, co_summary, cg_summary):
                if block_reuse and summary["block_reuse_bpc"] is not None:
                    total_bpp += summary["block_reuse_bpc"]
                else:
                    total_bpp += summary["ctx_trit_bpc"]
            total_bpp += sign_bpp
            print(f"ycocg combined_total_bpp: {total_bpp:5.3f}")
        else:
            for ch, name in enumerate(["r", "g", "b"]):
                print(f"--- channel {name} ---")
                summary = _bench_frames(
                    frames_rgb[..., ch],
                    label_prefix=f"{name}_",
                    width=width,
                    height=height,
                    pixels=pixels,
                    mc=mc,
                    jax_mc=jax_mc,
                    jax_pipeline=jax_pipeline,
                    mc_block=mc_block,
                    mc_search=mc_search,
                    mc_pyramid=mc_pyramid,
                    train_split=train_split,
                    block_reuse=block_reuse,
                    reuse_block=reuse_block,
                    reuse_dict=reuse_dict,
                    reuse_planes=reuse_planes,
                    dump_planes_dir=dump_planes_dir,
                    dump_planes_frames=dump_planes_frames,
                )
                if ch == 0:
                    total_bpp = 0.0
                if block_reuse and summary["block_reuse_bpc"] is not None:
                    total_bpp += summary["block_reuse_bpc"]
                else:
                    total_bpp += summary["ctx_trit_bpc"]
            print(f"rgb combined_total_bpp: {total_bpp:5.3f}")
        return

    frames = decode_gray(path, width, height, max_frames)
    actual_frames = frames.shape[0]
    pixels = frames.size

    _bench_frames(
        frames,
        label_prefix="",
        width=width,
        height=height,
        pixels=pixels,
        mc=mc,
        jax_mc=jax_mc,
        jax_pipeline=jax_pipeline,
        mc_block=mc_block,
        mc_search=mc_search,
        mc_pyramid=mc_pyramid,
        train_split=train_split,
        block_reuse=block_reuse,
        reuse_block=reuse_block,
        reuse_dict=reuse_dict,
        reuse_planes=reuse_planes,
        dump_planes_dir=dump_planes_dir,
        dump_planes_frames=dump_planes_frames,
    )


def _bench_frames(
    frames: np.ndarray,
    label_prefix: str,
    width: int,
    height: int,
    pixels: int,
    mc: bool,
    jax_mc: bool,
    jax_pipeline: bool,
    mc_block: int,
    mc_search: int,
    mc_pyramid: int,
    train_split: float,
    block_reuse: bool,
    reuse_block: int,
    reuse_dict: int,
    reuse_planes: int,
    dump_planes_dir: Path | None,
    dump_planes_frames: int,
) -> None:
    actual_frames = frames.shape[0]
    if jax_pipeline:
        try:
            from JAX import pipeline as jax_pipeline_mod
        except ImportError as exc:
            raise SystemExit(
                "JAX pipeline requested but JAX modules are unavailable. Install JAX or run without --jax-pipeline."
            ) from exc
        streams, signed_resid, aux = jax_pipeline_mod.compute_streams(frames)
        bt_cache = jax_pipeline_mod.compute_bt_planes(signed_resid, frames.shape)
    else:
        # Streams
        raw_stream = frames.ravel()
        residual = np.empty_like(raw_stream)
        residual[: width * height] = raw_stream[: width * height]
        if actual_frames > 1:
            diffs = (frames[1:] - frames[:-1]) & 0xFF
            residual[width * height :] = diffs.ravel()

        # Signed temporal residuals for balanced ternary coding
        base = frames[0].astype(np.int16) - 128
        if actual_frames > 1:
            diffs_signed = frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)
            signed_resid = np.concatenate([base.ravel(), diffs_signed.ravel()])
        else:
            signed_resid = base.ravel()

        # Orbit canonicalization for grayscale: reflect around mid (127.5)
        coarse = np.minimum(raw_stream, 255 - raw_stream).astype(np.uint8)  # orbit ID
        sign = (raw_stream > 127).astype(np.uint8)  # witness/refinement

        # Temporal residuals on canonicalized streams
        coarse_resid = np.empty_like(coarse)
        sign_resid = np.empty_like(sign)
        coarse_resid[: width * height] = coarse[: width * height]
        sign_resid[: width * height] = sign[: width * height]
        if actual_frames > 1:
            coarse_frames = coarse.reshape(actual_frames, height, width)
            sign_frames = sign.reshape(actual_frames, height, width)
            coarse_resid[width * height :] = ((coarse_frames[1:] - coarse_frames[:-1]) & 0xFF).ravel()
            sign_resid[width * height :] = (sign_frames[1:] ^ sign_frames[:-1]).ravel()

        streams = {
            "raw": raw_stream,
            "residual": residual,
            "coarse": coarse,
            "sign": sign,
            "coarse_resid": coarse_resid,
            "sign_resid": sign_resid,
        }
        bt_cache = None

    # Explained energy by temporal residuals (L2)
    centered = frames.astype(np.int32) - 128
    base_energy = float((centered * centered).sum())
    resid_energy = float((signed_resid.astype(np.int64) ** 2).sum())
    explained_l2 = 1.0 - (resid_energy / base_energy) if base_energy else 0.0
    print(f"{label_prefix}explained_L2 (temporal residual): {explained_l2:6.4f}")

    for name, arr in streams.items():
        ent = stream_entropy(arr)
        payload = arr.tobytes()
        stats = compress_stats(name, payload)
        rans_start = time.perf_counter()
        rans_bytes = len(rans.encode(arr))
        rans_ms = (time.perf_counter() - rans_start) * 1000.0
        bpc = lambda n: (n * 8) / pixels
        print(
            f"{label_prefix}{name:<9} entropy={ent:6.3f}  "
            f"lzma {stats['lzma']:8d} ({bpc(stats['lzma']):5.3f} bpc, {stats['lzma_ms']:5.1f} ms)  "
            f"gzip {stats['gzip']:8d} ({bpc(stats['gzip']):5.3f} bpc, {stats['gzip_ms']:5.1f} ms)  "
            f"zlib {stats['zlib']:8d} ({bpc(stats['zlib']):5.3f} bpc, {stats['zlib_ms']:5.1f} ms)  "
            f"rANS {rans_bytes:8d} ({bpc(rans_bytes):5.3f} bpc, {rans_ms:5.1f} ms)"
        )

    # Combined orbit + sign streams (entropy-coded separately)
    combos = [
        ("coarse+sign", ["coarse", "sign"]),
        ("coarse_resid+sign_resid", ["coarse_resid", "sign_resid"]),
    ]
    for label, keys in combos:
        total_bytes = sum(len(rans.encode(streams[k])) for k in keys)
        bpc_combined = (total_bytes * 8) / pixels
        print(f"\n{label_prefix}multistream ({label} via rANS): {total_bytes} bytes ({bpc_combined:5.3f} bpc)")

    if bt_cache is None:
        bt_digits = _balanced_digits_needed(signed_resid)
        bt_planes = _balanced_ternary_digits(signed_resid, bt_digits)
        bt_planes_u8 = (bt_planes + 1).astype(np.uint8)
        bt_planes_i8 = bt_planes.astype(np.int8).reshape(bt_digits, frames.shape[0], height, width)
        bt_mag = np.abs(bt_planes_i8).astype(np.uint8)
        bt_sign = (bt_planes_i8 > 0).astype(np.uint8)
        bt_cache = {
            "bt_planes_u8": bt_planes_u8,
            "bt_planes_i8": bt_planes_i8,
            "bt_mag": bt_mag,
            "bt_sign": bt_sign,
            "bt_digits": bt_digits,
        }

    if dump_planes_dir is not None:
        _dump_bt_planes(
            dump_planes_dir,
            label_prefix,
            bt_cache["bt_planes_i8"],
            bt_cache["bt_mag"],
            bt_cache["bt_sign"],
            dump_planes_frames,
        )

    summary = _report_triadic(
        label=f"{label_prefix}base",
        signed_resid=signed_resid,
        frames=frames,
        height=height,
        width=width,
        pixels=pixels,
        train_split=train_split,
        block_reuse=block_reuse,
        reuse_block=reuse_block,
        reuse_dict=reuse_dict,
        reuse_planes=reuse_planes,
        bt_planes_u8=bt_cache["bt_planes_u8"],
        bt_planes_i8=bt_cache["bt_planes_i8"],
        bt_mag=bt_cache["bt_mag"],
        bt_sign=bt_cache["bt_sign"],
        bt_digits_override=int(bt_cache["bt_digits"]),
    )

    if mc:
        if jax_mc:
            try:
                from JAX import mdl_sideinfo as jax_mdl
                from JAX import motion_search as jax_motion
                import jax.numpy as jnp
            except ImportError as exc:
                raise SystemExit(
                    "JAX modules not available. Install JAX or run without --jax-mc."
                ) from exc

            mc_resid, stats = jax_motion.motion_compensated_residual(
                frames, block=mc_block, search=mc_search
            )
            mv_alpha = 0.6
            mv_blocks = sum(stats["mv_counts"].values())
            mv_list = []
            for (u, v), count in stats["mv_counts"].items():
                mv_list.extend([(u, v)] * count)
            if mv_list:
                mv_grid = jnp.asarray(mv_list, dtype=jnp.int32).reshape(-1, 1, 2)
                mv_side_bits = float(
                    jax_mdl.motion_translation_side_bits(
                        mv_grid, alpha_u=mv_alpha, alpha_v=mv_alpha, radius=mc_search
                    )
                )
            else:
                mv_side_bits = 0.0
            mv_bpb = (mv_side_bits / mv_blocks) if mv_blocks else 0.0
            mv_bpp = mv_side_bits / pixels if pixels else 0.0
            print(
                f"\n{label_prefix}mc stats (jax): block={mc_block} search={mc_search} pyramid=0 "
                f"max_abs={stats['max_abs']} mv_unique={len(stats['mv_counts'])} "
                f"mv_side_bits={mv_side_bits:.1f} mv_bpb={mv_bpb:.3f} mv_bpp={mv_bpp:.6f} "
                f"mv_alpha={mv_alpha:.2f}"
            )
        else:
            mc_resid, stats = _motion_compensated_residual(
                frames, block=mc_block, search=mc_search, pyramid_levels=mc_pyramid
            )
            mv_alpha = 0.6
            mv_blocks = sum(stats["mv_counts"].values())
            mv_side_bits = mdl_sideinfo.motion_translation_side_bits(
                stats["mv_counts"],
                alpha_u=mv_alpha,
                alpha_v=mv_alpha,
                radius=mc_search,
            )
            mv_bpb = (mv_side_bits / mv_blocks) if mv_blocks else 0.0
            mv_bpp = mv_side_bits / pixels if pixels else 0.0
            print(
                f"\n{label_prefix}mc stats: block={mc_block} search={mc_search} pyramid={mc_pyramid} "
                f"max_abs={stats['max_abs']} mv_unique={len(stats['mv_counts'])} "
                f"mv_side_bits={mv_side_bits:.1f} mv_bpb={mv_bpb:.3f} mv_bpp={mv_bpp:.6f} "
                f"mv_alpha={mv_alpha:.2f}"
            )
        _report_triadic(
            label=f"{label_prefix}mc",
            signed_resid=mc_resid,
            frames=frames,
            height=height,
            width=width,
            pixels=pixels,
            train_split=train_split,
            block_reuse=block_reuse,
            reuse_block=reuse_block,
            reuse_dict=reuse_dict,
            reuse_planes=reuse_planes,
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark triadic-style compression vs ffmpeg file.")
    parser.add_argument("video", type=Path, help="Path to input video (e.g., MP4).")
    parser.add_argument("--frames", type=int, default=120, help="Max frames to decode for the benchmark.")
    parser.add_argument("--mc", action="store_true", help="Enable blockwise motion compensation.")
    parser.add_argument("--jax-mc", action="store_true", help="Use JAX for motion compensation search.")
    parser.add_argument("--jax-pipeline", action="store_true", help="Use JAX for stream prep and triadic digits.")
    parser.add_argument("--color", action="store_true", help="Process RGB channels separately.")
    parser.add_argument(
        "--color-transform",
        choices=["rgb", "ycocg"],
        default="rgb",
        help="Color transform for --color (rgb or reversible ycocg).",
    )
    parser.add_argument("--mc-block", type=int, default=8, help="Block size for motion compensation.")
    parser.add_argument("--mc-search", type=int, default=4, help="Search radius for motion compensation.")
    parser.add_argument("--mc-pyramid", type=int, default=0, help="Factor-3 pyramid levels for motion search.")
    parser.add_argument("--train-split", type=float, default=0.5, help="Fraction of frames used for context training.")
    parser.add_argument("--block-reuse", action="store_true", help="Emit block reuse action stream stats.")
    parser.add_argument("--reuse-block", type=int, default=16, help="Block size for reuse actions.")
    parser.add_argument("--reuse-dict", type=int, default=256, help="Block reuse dictionary size.")
    parser.add_argument("--reuse-planes", type=int, default=2, help="Planes used for block hash.")
    parser.add_argument("--dump-planes", type=Path, help="Output dir for balanced-ternary plane PNGs + NPZ.")
    parser.add_argument(
        "--dump-planes-frames",
        type=int,
        default=1,
        help="Number of frames to dump per plane (0 = all).",
    )
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    run_video_bench(
        args.video,
        max_frames=args.frames,
        mc=args.mc or args.jax_mc,
        jax_mc=args.jax_mc,
        jax_pipeline=args.jax_pipeline,
        mc_block=args.mc_block,
        mc_search=args.mc_search,
        mc_pyramid=args.mc_pyramid,
        train_split=args.train_split,
        block_reuse=args.block_reuse,
        reuse_block=args.reuse_block,
        reuse_dict=args.reuse_dict,
        reuse_planes=args.reuse_planes,
        color=args.color,
        color_transform=args.color_transform,
        dump_planes_dir=args.dump_planes,
        dump_planes_frames=args.dump_planes_frames,
    )


if __name__ == "__main__":
    main()

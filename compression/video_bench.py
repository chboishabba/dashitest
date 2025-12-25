import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from . import rans  # type: ignore
except ImportError:  # direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from compression import rans  # type: ignore


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


def run_video_bench(path: Path, max_frames: int) -> None:
    width, height, nb_frames = ffprobe_video(path)
    frames = decode_gray(path, width, height, max_frames)
    actual_frames = frames.shape[0]
    pixels = frames.size

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
    bt_digits = _balanced_digits_needed(signed_resid)
    bt_planes = _balanced_ternary_digits(signed_resid, bt_digits)
    bt_planes_u8 = (bt_planes + 1).astype(np.uint8)  # map {-1,0,1} -> {0,1,2}
    bt_planes_i8 = bt_planes.astype(np.int8).reshape(bt_digits, actual_frames, height, width)
    bt_mag = np.abs(bt_planes_i8).astype(np.uint8)
    bt_sign = (bt_planes_i8 > 0).astype(np.uint8)

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

    print(f"Video: {path.name} | {width}x{height} | frames decoded: {actual_frames} (probe reported {nb_frames})")
    print(f"Original file bytes: {os.path.getsize(path)}")
    print(f"Total pixels: {pixels}")
    print()

    for name, arr in streams.items():
        ent = stream_entropy(arr)
        payload = arr.tobytes()
        stats = compress_stats(name, payload)
        rans_start = time.perf_counter()
        rans_bytes = len(rans.encode(arr))
        rans_ms = (time.perf_counter() - rans_start) * 1000.0
        bpc = lambda n: (n * 8) / pixels
        print(
            f"{name:<9} entropy={ent:6.3f}  "
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
        print(f"\nmultistream ({label} via rANS): {total_bytes} bytes ({bpc_combined:5.3f} bpc)")

    # Balanced ternary planes for signed temporal residuals (lossless triadic)
    print(f"\nbalanced ternary digits for signed residuals: {bt_digits} planes")
    plane_bytes = []
    ctx_plane_bytes = []
    for idx in range(bt_digits):
        plane = bt_planes_u8[idx]
        ent = stream_entropy(plane)
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
    total_bt = sum(plane_bytes)
    bpc_bt = (total_bt * 8) / pixels
    print(f"\nmultistream (balanced ternary planes via rANS): {total_bt} bytes ({bpc_bt:5.3f} bpc)")
    total_ctx = sum(ctx_plane_bytes)
    bpc_ctx = (total_ctx * 8) / pixels
    print(f"multistream (balanced ternary planes ctx_rANS): {total_ctx} bytes ({bpc_ctx:5.3f} bpc)")

    # Per-plane Z2 quotient: magnitude plane + gated sign stream
    print("\nbalanced ternary plane quotient (mag + gated sign)")
    mag_bytes = []
    mag_ctx_bytes = []
    sign_bytes = []
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

        sign_ent = stream_entropy(sign_stream.astype(np.uint8)) if sign_stream.size else 0.0
        start = time.perf_counter()
        sign_enc = rans.encode(sign_stream.astype(np.uint8), alphabet=2)
        sign_ms = (time.perf_counter() - start) * 1000.0
        sign_bytes.append(len(sign_enc))
        sign_bpc = (len(sign_enc) * 8) / pixels

        print(
            f"bt_plane{idx:<2} mag_ent={mag_ent:5.3f} "
            f"rANS {len(mag_enc):7d} ({mag_bpc:5.3f} bpc, {mag_ms:5.1f} ms)  "
            f"ctx {len(mag_ctx_enc):7d} ({mag_ctx_bpc:5.3f} bpc, {mag_ctx_ms:5.1f} ms)  "
            f"sign_ent={sign_ent:5.3f} "
            f"sign_rANS {len(sign_enc):7d} ({sign_bpc:5.3f} bpc, {sign_ms:5.1f} ms)"
        )

    total_mag = sum(mag_bytes)
    total_mag_ctx = sum(mag_ctx_bytes)
    total_sign = sum(sign_bytes)
    total_q = total_mag + total_sign
    total_q_ctx = total_mag_ctx + total_sign
    print(f"\nmultistream (bt mag + sign via rANS): {total_q} bytes ({(total_q * 8) / pixels:5.3f} bpc)")
    print(f"multistream (bt mag ctx + sign via rANS): {total_q_ctx} bytes ({(total_q_ctx * 8) / pixels:5.3f} bpc)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark triadic-style compression vs ffmpeg file.")
    parser.add_argument("video", type=Path, help="Path to input video (e.g., MP4).")
    parser.add_argument("--frames", type=int, default=120, help="Max frames to decode for the benchmark.")
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    run_video_bench(args.video, max_frames=args.frames)


if __name__ == "__main__":
    main()

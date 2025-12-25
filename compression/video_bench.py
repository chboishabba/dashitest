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
    total_bt = sum(plane_bytes)
    bpc_bt = (total_bt * 8) / pixels
    print(f"\nmultistream (balanced ternary planes via rANS): {total_bt} bytes ({bpc_bt:5.3f} bpc)")


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

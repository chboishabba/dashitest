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



def _trit_flip_u8(arr: np.ndarray) -> np.ndarray:
    """Map trit plane encoded as {0,1,2} to its sign-flipped version."""
    # -1 <-> +1 i.e. 0 <-> 2; 1 stays.
    return (2 - arr).astype(np.uint8)

def _canonical_block_signature(planes: list[np.ndarray]) -> bytes:
    """Canonicalize a multi-plane block up to global sign flip (Z2 quotient)."""
    sig = b"".join(p.tobytes() for p in planes)
    flipped = b"".join(_trit_flip_u8(p).tobytes() for p in planes)
    return sig if sig <= flipped else flipped

def block_reuse_codec_size(
    bt_planes_u8: np.ndarray,
    h: int,
    w: int,
    frames: int,
    block: int = 8,
    dict_size: int = 256,
    hash_planes: int = 2,
) -> dict:
    """
    Approximate a *real* block-reuse coding path:
      - build action stream {NEW,SAME,REUSE}
      - build ref-index stream for REUSE
      - mask non-NEW blocks in all planes to the neutral trit (1) and encode planes

    bt_planes_u8: shape (digits, pixels_total) with pixels_total = frames*h*w, stored frame-major.
    Returns sizes in bytes.
    """
    digits, total = bt_planes_u8.shape
    assert total == frames * h * w, "bt_planes_u8 must be frame-major with total = frames*h*w"
    hp = max(1, min(hash_planes, digits))

    # View each plane as (frames, h, w) without copying.
    planes_fhw = [bt_planes_u8[k].reshape(frames, h, w) for k in range(digits)]
    nbx = w // block
    nby = h // block
    nblocks = nbx * nby

    # Dictionary: fixed-size slot table so indices are stable.
    dict_slots = [None] * dict_size  # bytes or None
    sig_to_idx: dict[bytes, int] = {}
    next_slot = 0

    # Previous-frame signatures per position for SAME detection.
    prev_pos_sig = [None] * nblocks

    actions = []  # 0=new, 1=same, 2=reuse
    refs = []     # dict indices for reuse actions

    # Mask of NEW blocks: bool array (frames, nby, nbx)
    new_mask = np.zeros((frames, nby, nbx), dtype=bool)

    for t in range(frames):
        for by in range(nby):
            y0 = by * block
            y1 = y0 + block
            for bx in range(nbx):
                x0 = bx * block
                x1 = x0 + block
                pos = by * nbx + bx

                # Signature from the first hp planes.
                blk_planes = [planes_fhw[k][t, y0:y1, x0:x1] for k in range(hp)]
                sig = _canonical_block_signature(blk_planes)

                if prev_pos_sig[pos] == sig:
                    actions.append(1)  # SAME (free copy from prev frame)
                    # don't insert into dict again; it is effectively already available via SAME
                else:
                    idx = sig_to_idx.get(sig)
                    if idx is not None:
                        actions.append(2)  # REUSE
                        refs.append(idx)
                    else:
                        actions.append(0)  # NEW
                        new_mask[t, by, bx] = True

                        # Insert into dictionary.
                        old = dict_slots[next_slot]
                        if old is not None:
                            sig_to_idx.pop(old, None)
                        dict_slots[next_slot] = sig
                        sig_to_idx[sig] = next_slot
                        next_slot = (next_slot + 1) % dict_size

                prev_pos_sig[pos] = sig

    actions_u8 = np.array(actions, dtype=np.uint8)
    # Encode action stream and ref stream.
    action_bytes = rans.encode(actions_u8, alphabet=3)
    ref_bytes = b"" if len(refs) == 0 else rans.encode(np.array(refs, dtype=np.uint16), alphabet=dict_size)

    # Mask planes: keep pixels of NEW blocks, set others to neutral trit (1).
    # We operate blockwise to avoid huge boolean masks per pixel.
    plane_sizes = []
    for k in range(digits):
        masked = planes_fhw[k].copy()
        for t in range(frames):
            for by in range(nby):
                y0 = by * block
                y1 = y0 + block
                for bx in range(nbx):
                    if not new_mask[t, by, bx]:
                        x0 = bx * block
                        x1 = x0 + block
                        masked[t, y0:y1, x0:x1] = 1  # neutral trit (0 in {-1,0,1})
        plane_sizes.append(len(rans.encode(masked.ravel().astype(np.uint8), alphabet=3)))

    return {
        "block": block,
        "dict_size": dict_size,
        "hash_planes": hp,
        "nblocks": nblocks,
        "actions_bytes": len(action_bytes),
        "refs_bytes": len(ref_bytes),
        "planes_bytes": int(sum(plane_sizes)),
        "total_bytes": int(len(action_bytes) + len(ref_bytes) + sum(plane_sizes)),
        "reuse_rate": float(np.mean(actions_u8 == 2)) if len(actions_u8) else 0.0,
        "same_rate": float(np.mean(actions_u8 == 1)) if len(actions_u8) else 0.0,
        "new_rate": float(np.mean(actions_u8 == 0)) if len(actions_u8) else 0.0,
    }

def run_video_bench(path: Path, max_frames: int, *, block_reuse: bool=False, reuse_block: int=8, reuse_dict: int=256, reuse_planes: int=2) -> None:
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

    if block_reuse:
        reuse = block_reuse_codec_size(
            bt_planes_u8, h=h, w=w, frames=actual_frames,
            block=reuse_block, dict_size=reuse_dict, hash_planes=reuse_planes,
        )
        reuse_bpc = (8.0 * reuse['total_bytes']) / pixels
        print(
            f"\nblock_reuse rANS (actions+refs+masked planes): {reuse['total_bytes']} bytes ({reuse_bpc:.3f} bpc)"
            f" | new={reuse['new_rate']:.3f} same={reuse['same_rate']:.3f} reuse={reuse['reuse_rate']:.3f}"
            f" | action={reuse['actions_bytes']} ref={reuse['refs_bytes']} planes={reuse['planes_bytes']}"
        )



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark triadic-style compression vs ffmpeg file.")
    parser.add_argument("video", type=Path, help="Path to input video (e.g., MP4).")
    parser.add_argument("--frames", type=int, default=120, help="Max frames to decode for the benchmark.")
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    run_video_bench(Path(args.video), args.frames, block_reuse=args.block_reuse, reuse_block=args.reuse_block, reuse_dict=args.reuse_dict, reuse_planes=args.reuse_planes)


if __name__ == "__main__":
    main()
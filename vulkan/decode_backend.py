from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Generator, Iterable, Tuple

import numpy as np


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


def decode_gray_stream(
    path: Path,
    width: int,
    height: int,
    max_frames: int,
    use_vaapi: bool,
    vaapi_device: str,
) -> Tuple[Iterable[np.ndarray], bool]:
    """Yield grayscale frames as 2D uint8 arrays; returns (iterator, used_vaapi)."""
    frame_size = width * height
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
    used_vaapi = False
    if use_vaapi:
        cmd = [
            "ffmpeg",
            "-hwaccel",
            "vaapi",
            "-vaapi_device",
            vaapi_device,
            "-hwaccel_output_format",
            "vaapi",
            "-i",
            str(path),
            "-vf",
            "hwdownload,format=gray",
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
        used_vaapi = True

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _iter_frames() -> Generator[np.ndarray, None, None]:
        assert proc.stdout is not None
        for _ in range(max_frames):
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            arr = np.frombuffer(raw, dtype=np.uint8)
            yield arr.reshape(height, width)

    return _iter_frames(), used_vaapi


def decode_gray_buffer(
    path: Path,
    width: int,
    height: int,
    max_frames: int,
    use_vaapi: bool,
    vaapi_device: str,
) -> Tuple[np.ndarray, bool]:
    """Decode all grayscale frames into memory; returns array [T,H,W]."""
    frames_iter, used_vaapi = decode_gray_stream(
        path, width, height, max_frames, use_vaapi, vaapi_device
    )
    frames = list(frames_iter)
    if not frames:
        return np.zeros((0, height, width), dtype=np.uint8), used_vaapi
    stacked = np.stack(frames, axis=0)
    return stacked, used_vaapi

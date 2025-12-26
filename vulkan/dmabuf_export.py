from __future__ import annotations

import os
import socket
import struct
import subprocess
from pathlib import Path
from typing import List, Tuple

MAX_OBJECTS = 4
MAX_PLANES = 4

HEADER_FMT = "<IIIII"
OBJ_FMT = "<QQ"
PLANE_FMT = "<III"
INFO_SIZE = (
    struct.calcsize(HEADER_FMT)
    + MAX_OBJECTS * struct.calcsize(OBJ_FMT)
    + MAX_PLANES * struct.calcsize(PLANE_FMT)
)


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout.strip()


def _build_helper(source: Path, binary: Path) -> None:
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return
    cflags = _run(
        ["pkg-config", "--cflags", "libavformat", "libavcodec", "libavutil", "libva", "libva-drm"]
    ).split()
    libs = _run(
        ["pkg-config", "--libs", "libavformat", "libavcodec", "libavutil", "libva", "libva-drm"]
    ).split()
    cmd = [
        "cc",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-o",
        str(binary),
        str(source),
    ] + cflags + libs
    subprocess.run(cmd, check=True)


def _recv_dmabuf(sock: socket.socket) -> Tuple[dict, List[int]]:
    data, ancdata, _, _ = sock.recvmsg(INFO_SIZE, socket.CMSG_SPACE(MAX_OBJECTS * 4))
    if len(data) < INFO_SIZE:
        raise RuntimeError("short dmabuf metadata read")
    fds: List[int] = []
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.extend(struct.unpack(f"<{len(cmsg_data) // 4}i", cmsg_data))
    offset = 0
    width, height, drm_format, nb_objects, nb_planes = struct.unpack_from(HEADER_FMT, data, offset)
    offset += struct.calcsize(HEADER_FMT)
    objects = []
    for _ in range(MAX_OBJECTS):
        modifier, size = struct.unpack_from(OBJ_FMT, data, offset)
        objects.append({"modifier": modifier, "size": size})
        offset += struct.calcsize(OBJ_FMT)
    planes = []
    for _ in range(MAX_PLANES):
        obj_index, plane_offset, pitch = struct.unpack_from(PLANE_FMT, data, offset)
        planes.append({"object_index": obj_index, "offset": plane_offset, "pitch": pitch})
        offset += struct.calcsize(PLANE_FMT)
    return (
        {
            "width": width,
            "height": height,
            "drm_format": drm_format,
            "nb_objects": nb_objects,
            "nb_planes": nb_planes,
            "objects": objects,
            "planes": planes,
        },
        fds,
    )


def export_dmabuf(
    video: Path,
    vaapi_device: str,
    *,
    force_linear: bool = False,
    drm_device: str | None = None,
    timeout_s: float = 5.0,
    debug: bool = False,
) -> Tuple[dict, List[int], str]:
    source = Path(__file__).with_name("vaapi_dmabuf_export.c")
    binary = Path(__file__).with_name("vaapi_dmabuf_export")
    _build_helper(source, binary)

    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    parent_sock.settimeout(timeout_s)
    env = os.environ.copy()
    env["DMABUF_STUB_SOCK_FD"] = str(child_sock.fileno())
    cmd = [str(binary), str(video), vaapi_device]
    if force_linear:
        cmd.append("--force-linear")
    if debug:
        cmd.append("--debug")
    if drm_device:
        cmd.extend(["--drm-device", drm_device])
    proc = subprocess.Popen(
        cmd,
        pass_fds=[child_sock.fileno()],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_sock.close()

    try:
        info, fds = _recv_dmabuf(parent_sock)
    except socket.timeout as exc:
        proc.kill()
        out, err = proc.communicate()
        detail = err.strip() or out.strip()
        if detail:
            detail = f" | exporter: {detail}"
        raise RuntimeError(f"timed out waiting for dmabuf export ({timeout_s}s){detail}") from exc
    finally:
        parent_sock.close()

    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        out, err = proc.communicate()
        raise RuntimeError(f"export helper did not exit in {timeout_s}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(out + err)

    fds = fds[: info["nb_objects"]]
    if len(fds) < info["nb_objects"]:
        raise RuntimeError("did not receive enough dmabuf fds")

    log = (err or "") + (out or "")
    return info, fds, log.strip()

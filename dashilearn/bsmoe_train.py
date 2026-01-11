"""
bsmoe_train.py
--------------
Block-sparse MoE training demo:
  - Gate selects active tiles (block-level)
  - Tile masks are derived from per-output gate activity
  - Dense int8 matmul microkernel runs only on active tiles
  - Emit once per block
"""

import argparse
import time
import os
import sys
import ctypes
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path so top-level Vulkan helpers import reliably.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from vulkan_compute.frame_capture import VulkanFrameCapture
except Exception:
    VulkanFrameCapture = None


def gate_prob_for_tile_density(tile_active, tile):
    if tile_active <= 0.0:
        return 0.0
    if tile_active >= 1.0:
        return 1.0
    return 1.0 - (1.0 - tile_active) ** (1.0 / (tile * tile))


def tiles_from_gate_mask(gate_mask, tile=32):
    M, N = gate_mask.shape
    tiles = np.zeros(((M + tile - 1) // tile, (N + tile - 1) // tile), dtype=bool)
    for ti in range(tiles.shape[0]):
        for tj in range(tiles.shape[1]):
            i0, j0 = ti * tile, tj * tile
            i1, j1 = min(i0 + tile, M), min(j0 + tile, N)
            tiles[ti, tj] = np.any(gate_mask[i0:i1, j0:j1])
    return tiles


SHEET_OUT_PATH = Path(__file__).with_name("sheet_energy.npy")
VULKAN_CAPTURE_THRESHOLD = 128


def tile_energy_map(C, plan):
    energy = np.zeros(plan.tile_grid_shape, dtype=np.float32)
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        row = i0 // plan.tile
        col = j0 // plan.tile
        block = C[i0:i1, j0:j1].astype(np.float32)
        energy[row, col] = float(np.sum(block * block))
    return energy


def dump_sheet_energy(energy, path=SHEET_OUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, energy)
    tmp.replace(path)


def _sheet_values_for_capture(
    sheet_energy: np.ndarray,
    *,
    epoch: int,
    gate_density: float,
    sheet_h: int,
    sheet_w: int,
) -> np.ndarray:
    if sheet_h <= 0 or sheet_w <= 0:
        return np.zeros_like(sheet_energy, dtype=np.float32)
    if sheet_energy.size == 0:
        return np.zeros((sheet_h, sheet_w), dtype=np.float32)
    sheet = np.zeros((sheet_h, sheet_w), dtype=np.float32)
    i = epoch % sheet_h
    j = (epoch // sheet_h) % sheet_w
    sheet[i, j] = np.clip(gate_density, 0.0, 1.0)
    return sheet


def make_data(M=256, K=256, N=256, tiles_active=0.5, tile=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(M, K), dtype=np.int8)
    W = rng.integers(0, 3, size=(K, N), dtype=np.int8)
    # gate mask -> active tiles (tile-level any-activation)
    gate_prob = gate_prob_for_tile_density(tiles_active, tile)
    gate_mask = rng.random((M, N)) < gate_prob
    tiles = tiles_from_gate_mask(gate_mask, tile=tile)
    return X, W, tiles


def dense_matmul(X, W):
    return X.astype(np.int32) @ W.astype(np.int32)


@dataclass
class TilePlan:
    tile: int
    tile_grid_shape: tuple
    i0: np.ndarray
    i1: np.ndarray
    j0: np.ndarray
    j1: np.ndarray
    tile_ids: np.ndarray

    @property
    def count(self):
        return int(self.i0.size)


def build_tile_plan(tiles, tile, M, N):
    coords = np.argwhere(tiles)
    if coords.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return TilePlan(
            tile=tile,
            tile_grid_shape=tiles.shape,
            i0=empty,
            i1=empty,
            j0=empty,
            j1=empty,
            tile_ids=empty,
        )
    i0 = (coords[:, 0] * tile).astype(np.int32)
    j0 = (coords[:, 1] * tile).astype(np.int32)
    i1 = np.minimum(i0 + tile, M).astype(np.int32)
    j1 = np.minimum(j0 + tile, N).astype(np.int32)
    tile_ids = (coords[:, 0] * tiles.shape[1] + coords[:, 1]).astype(np.int32)
    return TilePlan(
        tile=tile,
        tile_grid_shape=tiles.shape,
        i0=i0,
        i1=i1,
        j0=j0,
        j1=j1,
        tile_ids=tile_ids,
    )


def jaccard_similarity(a_ids, b_ids):
    if a_ids.size == 0 and b_ids.size == 0:
        return 1.0
    if a_ids.size == 0 or b_ids.size == 0:
        return 0.0
    a = set(a_ids.tolist())
    b = set(b_ids.tolist())
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 1.0


def update_gate_mask(gate_mask, flip_prob, rng):
    if flip_prob <= 0.0:
        return gate_mask
    flips = rng.random(gate_mask.shape) < flip_prob
    return np.where(flips, ~gate_mask, gate_mask)


def _load_vnni_kernel():
    lib_path = os.path.join(os.path.dirname(__file__), "vnni_kernel.so")
    if not os.path.exists(lib_path):
        return None
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        return None
    fn = lib.vnni_tile_i8
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int8),  # A
        ctypes.POINTER(ctypes.c_int8),  # B
        ctypes.POINTER(ctypes.c_int32),  # C
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        ctypes.c_int,  # lda
        ctypes.c_int,  # ldb
        ctypes.c_int,  # ldc
    ]
    return fn


_VNNI_KERNEL = _load_vnni_kernel()


def vnni_microkernel(Ablk, Bblk):
    if _VNNI_KERNEL is None:
        return Ablk.astype(np.int32) @ Bblk.astype(np.int32)
    A = np.ascontiguousarray(Ablk, dtype=np.int8)
    B = np.ascontiguousarray(Bblk, dtype=np.int8)
    C_view = np.zeros((Ablk.shape[0], Bblk.shape[1]), dtype=np.int32)
    _VNNI_KERNEL(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        C_view.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        A.shape[0],
        B.shape[1],
        A.shape[1],
        A.strides[0] // A.itemsize,
        B.strides[0] // B.itemsize,
        C_view.strides[0] // C_view.itemsize,
    )
    return C_view


def block_sparse_matmul(X, W, tiles, tile=32, microkernel=vnni_microkernel):
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.int32)
    for ti in range(tiles.shape[0]):
        for tj in range(tiles.shape[1]):
            if not tiles[ti, tj]:
                continue
            i0, j0 = ti * tile, tj * tile
            i1, j1 = min(i0 + tile, M), min(j0 + tile, N)
            for k0 in range(0, K, tile):
                k1 = min(k0 + tile, K)
                Ablk = X[i0:i1, k0:k1]
                Bblk = W[k0:k1, j0:j1]
                C[i0:i1, j0:j1] += microkernel(Ablk, Bblk)
    return C


def block_sparse_matmul_plan(X, W, plan, microkernel=vnni_microkernel):
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.int32)
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        for k0 in range(0, K, plan.tile):
            k1 = min(k0 + plan.tile, K)
            Ablk = X[i0:i1, k0:k1]
            Bblk = W[k0:k1, j0:j1]
            C[i0:i1, j0:j1] += microkernel(Ablk, Bblk)
    return C


def activation_plan(C, plan, clamp_min=0):
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        block = C[i0:i1, j0:j1]
        C[i0:i1, j0:j1] = np.maximum(block, clamp_min)
    return C


def energy_plan(C, plan):
    energies = np.zeros(plan.count, dtype=np.float32)
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        block = C[i0:i1, j0:j1]
        energies[idx] = float(np.sum(block * block))
    return energies


def fused_sequence(X, W, plan, microkernel=vnni_microkernel):
    C = block_sparse_matmul_plan(X, W, plan, microkernel=microkernel)
    C = activation_plan(C, plan, clamp_min=0)
    energies = energy_plan(C, plan)
    return C, energies


def train_epoch(X, W, plan, lr=1e-3, microkernel=vnni_microkernel):
    # forward block-sparse
    C = block_sparse_matmul_plan(X, W, plan, microkernel=microkernel)
    # fake target: zeros
    err = C
    # backward: simple gradient on W for active tiles
    gradW = np.zeros_like(W, dtype=np.int32)
    for idx in range(plan.count):
        i0 = int(plan.i0[idx])
        j0 = int(plan.j0[idx])
        i1 = int(plan.i1[idx])
        j1 = int(plan.j1[idx])
        # X_blk: (tile, K) -> transpose -> (K, tile)
        X_blk = X[i0:i1, :].astype(np.int32)  # (tile, K)
        err_blk = err[i0:i1, j0:j1]           # (tile, tile)
        gradW[:, j0:j1] += X_blk.T @ err_blk
    W = W - lr * gradW.astype(W.dtype)
    loss = float((err ** 2).mean())
    return W, loss, C


def bench(fn, *args, reps=3, **kwargs):
    fn(*args, **kwargs)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        dur = (t1 - t0) * 1e3
        best = dur if best is None or dur < best else best
    return best


def main():
    parser = argparse.ArgumentParser(description="dashilearn block-sparse demo")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--stay-open",
        type=float,
        default=0.0,
        help="Seconds to keep running after training finishes (writes the last sheet each second)",
    )
    parser.add_argument("--stay-interval", type=float, default=1.0, help="Seconds between sheet refresh while staying open")
    parser.add_argument(
        "--capture-vulkan",
        action="store_true",
        help="Capture the Vulkan sheet frame once per epoch via the Task B wiring helper",
    )
    parser.add_argument("--vulkan-block-px", type=int, default=16, help="Pixels per sheet tile in the captured frame")
    parser.add_argument("--vulkan-alpha", type=float, default=0.97, help="Sheet fade alpha for Vulkan capture")
    parser.add_argument("--vulkan-vmin", type=float, default=0.0, help="Minimum sheet value clamp")
    parser.add_argument("--vulkan-vmax", type=float, default=1.0, help="Maximum sheet value clamp")
    parser.add_argument(
        "--vulkan-clamp",
        action="store_true",
        help="Clamp sheet values before the Vulkan shader runs",
    )
    parser.add_argument(
        "--vk-icd",
        type=Path,
        help="Optional Vulkan ICD JSON used when capturing the frame",
    )
    args = parser.parse_args()

    M = N = K = 256
    tile_active = 0.5
    tile = 32
    epochs = args.epochs
    gate_flip = 0.01
    jaccard_thresh = 0.9
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(M, K), dtype=np.int8)
    W = rng.integers(0, 3, size=(K, N), dtype=np.int8)
    gate_prob = gate_prob_for_tile_density(tile_active, tile)
    gate_mask = rng.random((M, N)) < gate_prob
    tiles = tiles_from_gate_mask(gate_mask, tile=tile)

    # Baseline dense timing
    t_dense = bench(dense_matmul, X, W)

    t_plan0 = time.perf_counter()
    plan = build_tile_plan(tiles, tile=tile, M=M, N=N)
    t_plan = (time.perf_counter() - t_plan0) * 1e3
    t_pack = 0.0

    # Block-sparse timing (plan reused)
    t_bs = bench(block_sparse_matmul_plan, X, W, plan, microkernel=vnni_microkernel)
    C_ref = block_sparse_matmul_plan(X, W, plan, microkernel=vnni_microkernel)
    t_act = bench(lambda: activation_plan(C_ref.copy(), plan, clamp_min=0))
    t_energy = bench(lambda: energy_plan(C_ref, plan))
    t_fused = bench(lambda: fused_sequence(X, W, plan, microkernel=vnni_microkernel))
    sheet_energy = tile_energy_map(C_ref, plan)
    dump_sheet_energy(sheet_energy)
    vulkan_capture = None
    if args.capture_vulkan:
        try:
            if VulkanFrameCapture is None:
                raise RuntimeError("vulkan_compute.frame_capture is unavailable")
            if args.vk_icd:
                os.environ["VK_ICD_FILENAMES"] = str(args.vk_icd)
            sheet_h, sheet_w = plan.tile_grid_shape
            width = sheet_w * args.vulkan_block_px
            height = sheet_h * args.vulkan_block_px
            vulkan_capture = VulkanFrameCapture(
                width=width,
                height=height,
                sheet_w=sheet_w,
                sheet_h=sheet_h,
                block_px=args.vulkan_block_px,
                alpha=args.vulkan_alpha,
                vmin=args.vulkan_vmin,
                vmax=args.vulkan_vmax,
                use_clamp=args.vulkan_clamp,
            )
        except Exception as exc:
            print(f"warning: Vulkan frame capture disabled: {exc}", file=sys.stderr)
            vulkan_capture = None
    active_frac = float(tiles.mean()) if tiles.size else 0.0
    last_capture_sheet = None
    if vulkan_capture:
        try:
            capture_sheet = _sheet_values_for_capture(
                sheet_energy,
                epoch=0,
                gate_density=active_frac,
                sheet_h=sheet_h,
                sheet_w=sheet_w,
            )
            frame = vulkan_capture.capture(capture_sheet)
            thr = VULKAN_CAPTURE_THRESHOLD
            frac_above_thr = (frame > thr).mean()
            max_val = float(frame.max())
            print(
                "Initial Vulkan frame captured:",
                frame.shape,
                f"mean={frame.mean():.2f} std={frame.std():.2f}",
                f"frac>{thr}={frac_above_thr:.3f}",
                f"max={max_val:.2f}",
            )
            last_capture_sheet = capture_sheet
        except Exception as exc:
            print(f"warning: failed to capture initial Vulkan frame: {exc}", file=sys.stderr)

    print("Block-sparse MoE-style matmul")
    print(f"M=N=K={M}, active tiles ~{active_frac*100:.1f}% (target {tile_active*100:.1f}%)")
    print(f"dense matmul      : {t_dense:6.2f} ms/call")
    print(f"block-sparse matmul: {t_bs:6.2f} ms/call   speedup x{t_dense/t_bs:5.2f}")
    if _VNNI_KERNEL is None:
        print("microkernel        : vnni_microkernel (numpy int32 fallback)")
    else:
        print("microkernel        : vnni_kernel.so (ctypes)")

    print(f"plan time         : {t_plan:6.2f} ms")
    print(f"pack time         : {t_pack:6.2f} ms")
    print(f"exec matmul       : {t_bs:6.2f} ms")
    print(f"exec activation   : {t_act:6.2f} ms")
    print(f"exec energy       : {t_energy:6.2f} ms")
    print(f"exec fused total  : {t_fused:6.2f} ms")

    # Tiny training loop (for illustration)
    sheet_energy = np.zeros(plan.tile_grid_shape, dtype=np.float32)
    plan_hits = 0
    for e in range(epochs):
        t_gate0 = time.perf_counter()
        gate_mask = update_gate_mask(gate_mask, gate_flip, rng)
        tiles = tiles_from_gate_mask(gate_mask, tile=tile)
        t_gate = (time.perf_counter() - t_gate0) * 1e3
        next_plan = build_tile_plan(tiles, tile=tile, M=M, N=N)
        jacc = jaccard_similarity(plan.tile_ids, next_plan.tile_ids)
        reuse = jacc >= jaccard_thresh
        if not reuse:
            plan = next_plan
        plan_hits += int(reuse)
        t0 = time.perf_counter()
        W, loss, C = train_epoch(X, W, plan, lr=1e-5, microkernel=vnni_microkernel)
        t1 = time.perf_counter()
        print(
            f"epoch {e+1}: loss={loss:8.2e}  time={(t1-t0)*1e3:6.2f} ms  "
            f"jaccard={jacc:5.2f}  plan_hit={int(reuse)}  gate_time={t_gate:5.2f} ms"
        )
        sheet_energy = tile_energy_map(C, plan)
        dump_sheet_energy(sheet_energy)
        if vulkan_capture:
            try:
                gate_density = float(tiles.mean()) if tiles.size else 0.0
                capture_sheet = _sheet_values_for_capture(
                    sheet_energy,
                    epoch=e,
                    gate_density=gate_density,
                    sheet_h=sheet_h,
                    sheet_w=sheet_w,
                )
                frame = vulkan_capture.capture(capture_sheet)
                thr = VULKAN_CAPTURE_THRESHOLD
                frac_above_thr = (frame > thr).mean()
                max_val = float(frame.max())
                print(
                    f"Vulkan frame [epoch {e+1}] mean={frame.mean():.2f}",
                    f"std={frame.std():.2f}",
                    f"frac>{thr}={frac_above_thr:.3f}",
                    f"max={max_val:.2f}",
                )
                last_capture_sheet = capture_sheet
            except Exception as exc:
                print(f"warning: Vulkan capture failed at epoch {e+1}: {exc}", file=sys.stderr)
    print(f"plan_hit_rate     : {plan_hits}/{epochs}")
    if args.stay_open > 0:
        stay_interval = max(0.01, args.stay_interval)
        end_time = time.time() + args.stay_open
        print(f"staying open for {args.stay_open:.1f}s (refresh every {stay_interval:.2f}s)")
        while time.time() < end_time:
            dump_sheet_energy(sheet_energy)
            if vulkan_capture and last_capture_sheet is not None:
                try:
                    vulkan_capture.capture(last_capture_sheet)
                except Exception:
                    pass
            time.sleep(stay_interval)
    if vulkan_capture:
        vulkan_capture.close()


if __name__ == "__main__":
    main()

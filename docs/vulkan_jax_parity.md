# Vulkan/JAX Parity Map (JAX is Reference-Only)

This note maps existing Vulkan entry points to their JAX reference modules.
JAX is not used at runtime on this machine; it is only a math/behavior
reference when comparing outputs.

Context anchors:
- Learner/executor split with JAX prototypes + Vulkan executor scaffolding
  (`CONTEXT.md#L18786`, `CONTEXT.md#L18795`, `CONTEXT.md#L18809`).
- JAX is optional; Vulkan is the primary executor on RX 580
  (`CONTEXT.md#L18827`, `CONTEXT.md#L18950`).

## Vulkan entry points (executor side)

- `vulkan/video_bench_vk.py`
  - Uses a Vulkan swapchain + optional compute pass (`--mode raw|diff`).
  - Relevant shader: `vulkan/shaders/diff.comp` (per-frame diff).
- `vulkan/symbol_stream_stub.py`
  - SSBO layout stub for symbols + per-plane trits.
- `vulkan_compute/compute_buffer.py`
  - Minimal SSBO compute path (storage buffer + push constants).
- `vulkan_compute/compute_image.py`
  - Storage-image compute path (writes to image).
- `vulkan_compute/compute_image_preview.py`
  - Live preview loop (useful for visual diagnostics).

## JAX reference modules (learner-side math, not runtime here)

- `JAX/motion_search.py`
  - Reference for block matching / motion search.
- `JAX/warps.py`
  - Reference for block warp application (translation/similarity).
- `JAX/predictor.py`
  - Reference for applying warps + producing predicted frames.
- `JAX/mdl_sideinfo.py`
  - MDL side-information (learner-only; stays CPU-side).
- `JAX/codec.py` + `JAX/pipeline.py`
  - Codec/pipeline scaffolding; useful for verifying stream structure.

## First Vulkan kernel to port (recommendation)

**Block-wise residual / diff kernel with block statistics**

Why this first:
- Already matches existing Vulkan scaffolding (`diff.comp`, SSBO paths).
- Maps cleanly to JAX `motion_search.py` cost calculation (SAD/MSE over blocks).
- Produces learner-facing observables (per-block error/energy) without relying on JAX.

Suggested I/O contract:
- Input: two grayscale frames (or predicted vs target), block size.
- Output: per-block error (SAD or sum of squares), plus optional per-block mean.
- Dispatch: one workgroup per block; write one scalar per block into SSBO.

Next kernel after that:
- Warp application (block-wise) guided by per-block params, matching `JAX/warps.py`.

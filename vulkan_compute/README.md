# Vulkan Compute Prototype

This folder contains a minimal Vulkan compute prototype for GPU execution on
gfx803 (RX 580) via RADV. The focus is compute-only kernels (SPIR-V) driven by
thin host code, aligned with the triadic compression pipeline.

## Goals

- Run core grid/CA/residual kernels on Vulkan compute.
- Keep host orchestration simple and explicit.
- Support debug visualization via storage images.

## Status

- Docs scaffolded.
- Minimal compute-buffer prototype added.

## Quick Start

1. Install dependencies:

   - `python-vulkan` (Python bindings)
   - `shaderc` (for `glslc`)

2. Compile the shader:

```bash
glslc vulkan_compute/shaders/add.comp -o vulkan_compute/shaders/add.spv
glslc vulkan_compute/shaders/write_image.comp -o vulkan_compute/shaders/write_image.spv
```

3. Run the sample:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  python vulkan_compute/compute_buffer.py
```

4. Run the storage image sample (optional dump to PPM):

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  python vulkan_compute/compute_image.py --width 256 --height 256 --dump out.ppm
```

5. Compile preview shaders:

```bash
glslc vulkan_compute/shaders/preview.vert -o vulkan_compute/shaders/preview.vert.spv
glslc vulkan_compute/shaders/preview.frag -o vulkan_compute/shaders/preview.frag.spv
```

6. Run the live preview (requires `glfw`):

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  python vulkan_compute/compute_image_preview.py --width 512 --height 512 --frames 240
```

Expected output shows the first 10 elements incremented by 1.

## Sheet Expand + Fade Shader (live sheet visual)

This repo includes a drop-in compute shader for expanding a small "semantic
sheet" into a large visible image with temporal fading, intended for live
"sheet lighting up" visualizations. It follows the learner/executor guidance
in `CONTEXT.md#L21313`.

Shader: `vulkan_compute/shaders/sheet_expand_fade.comp`

Compile:
```bash
glslc vulkan_compute/shaders/sheet_expand_fade.comp -o vulkan_compute/shaders/sheet_expand_fade.spv
```

Descriptor layout (set 0):
- binding 0: SSBO `float sheet[]` (size = `sheet_w*sheet_h`)
- binding 1: `r32f` storage image (accumulator)
- binding 2: `rgba8` storage image (display)

Push constants:
`sheet_w`, `sheet_h`, `out_w`, `out_h`, `block_px`, `alpha`, `vmin`, `vmax`, `use_clamp`.

Note: host wiring for this shader is not yet in `vulkan_compute/`; add a
compute pass that binds the above resources and dispatches over the output
image.

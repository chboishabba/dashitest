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
```

3. Run the sample:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  python vulkan_compute/compute_buffer.py
```

Expected output shows the first 10 elements incremented by 1.

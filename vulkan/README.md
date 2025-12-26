# Vulkan Video Bench (GPU Path)

This folder starts a Vulkan-based path for the video benchmark.

Goals:
- Move hot parts of `compression/video_bench.py` to GPU using Vulkan.
- Provide live graphical output of processing progress (preview window).
- Keep the CPU reference pipeline intact for fair comparisons.

Current status:
- `video_bench_vk.py` decodes grayscale frames on CPU and displays them
  through a Vulkan swapchain, with an optional compute pass for per-frame
  processing (`--mode raw|diff`).
- Future steps will move residuals and triadic digitization to compute shaders.

Requirements:
- `glfw` (Python package)
- Vulkan runtime/driver
- `ffmpeg` and `ffprobe`

Quick start:
```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  python vulkan/video_bench_vk.py path/to/video.mp4 --frames 240
```

Shader compile:
```
glslc vulkan/shaders/diff.comp -o vulkan/shaders/diff.spv
```

Notes:
- The preview uses the shared shaders in `vulkan_compute/shaders/`.
- This is a scaffolding step; compression kernels are not on the GPU yet.

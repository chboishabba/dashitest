# Changelog

## Unreleased

- Initialized project folder and docs.
- Added minimal Vulkan compute buffer sample (Python + GLSL).
- Fixed push-constant handling for python-vulkan.
- Added storage-image compute sample with optional PPM readback.
- Added GLFW preview sample (compute -> sampled image -> swapchain).
- Added `sheet_expand_fade.comp` shader for block-expanded sheet visualization
  with temporal fading.
- Wired `sheet_expand_fade.comp` into `compute_image_preview.py` via `--sheet`
  with SSBO + accumulator bindings and push-constant controls.
- Added `--sheet-data` + reload interval flags so `compute_image_preview.py`
  can read `dashilearn/sheet_energy.npy` (or any .npy) live while running.
- Added `--record-video` rawvideo pipe recording in `compute_image_preview.py`
  (CPU readback + ffmpeg).
- Recording outputs are now auto-timestamped and the output path is printed.

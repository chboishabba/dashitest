# Roadmap

## Phase 0: Bootstrap

- [ ] Minimal Vulkan compute dispatch from Python (python-vulkan).
- [ ] Single storage buffer kernel (sanity check).
- [ ] Validation layers enabled for early errors.

## Phase 1: Visual Debug

- [ ] Storage image output (RGBA8).
- [ ] Simple shader writes visible pattern.
- [ ] Optional CPU readback to PNG for verification.

## Phase 2: Domain Kernels

- [ ] CA step kernel (ternary grid update).
- [ ] Residual computation kernel (frame delta).
- [ ] Motion block SAD kernel (block matching).

## Phase 3: Integration

- [ ] Bridge to `compression/video_bench.py` via a thin CLI.
- [ ] Side-info cost computation on CPU + GPU results.

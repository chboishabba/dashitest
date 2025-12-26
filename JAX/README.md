# JAX prototype modules

This folder contains JAX implementations of the MDL side-information model,
motion search, and predictor warps. It is intended to mirror the math-only
codec specification with GPU-friendly code.

## Quick usage

Run the video benchmark with JAX motion search:

```bash
python compression/video_bench.py path/to/video.mp4 --jax-mc --mc-block 8 --mc-search 4
```

Run the full JAX pre-processing pipeline (streams + balanced ternary digits):

```bash
python compression/video_bench.py path/to/video.mp4 --jax-pipeline
```

If JAX is not installed, the benchmark will fall back to the NumPy motion
search or exit when `--jax-mc` is requested.

# Compression

This directory holds the repo's compression research code. The active work is a
mix of:

- a small synthetic triadic cellular-automaton benchmark,
- a real video benchmark that converts residuals into balanced-ternary planes,
- a local range coder used by those experiments, and
- supporting notes for MDL / side-information accounting.

The code here is experimental. Some files are current benchmark drivers, while
others are retained as design notes or scratch prototypes.

## What is active

- `video_bench.py`: main video compression benchmark. It decodes frames with
  `ffmpeg`, computes signed temporal residuals, expands them into
  balanced-ternary planes, and evaluates several coding layouts including
  contexted plane coding, per-plane magnitude/sign quotienting, optional block
  reuse, optional motion-compensated preprocessing, and optional color/YCoCg-R
  handling.
- `rans.py`: internal entropy coder. Despite the filename, this is currently a
  simple range coder with a stable `encode`/`decode` API used by the
  benchmarks.
- `mdl_sideinfo.py`: helper formulas for MDL-style side-information costs
  (translation priors, quadtree priors, lag priors, similarity parameters).
- `compression_bench.py`: synthetic benchmark over a deterministic triadic CA
  trace. Useful as a fast smoke test and for checking whether residualization
  or symbol transforms are helping on small-alphabet data.
- `comp_ca.py`: plotting-heavy CA generator/visualizer used to inspect the
  triadic gate/flow motifs behind the synthetic benchmark.

## Notes and design docs

- `triadic_pipeline.md`: the best explanation of the current video pipeline,
  especially the balanced-ternary plane representation and the per-plane Z2
  quotient into magnitude plus sign witness.
- `compression_context.txt`: long-form research notebook with compression
  hypotheses, benchmark framing, and next-step ideas.

## Prototype / exploratory files

- `gpt2.py`: packing-efficiency calculations for fitting ternary states into
  binary containers.
- `gpt3.py`: CA sweep and plotting experiment for stability/change-rate
  analysis.
- `gpt4.py`: richer CA exploration with moving anchors, fatigue, and motif
  statistics.
- `naieve_i_think.py`: quick scratch script comparing random bytes versus a
  naive trit expansion under zlib. Kept as a rough intuition check, not as a
  production benchmark.

## How the pieces fit together

1. `compression_bench.py` is the small, synthetic entry point.
2. `video_bench.py` is the main real-data benchmark path.
3. Both rely on `rans.py` for local entropy coding.
4. `mdl_sideinfo.py` supports accounting for model or motion side information.
5. `triadic_pipeline.md` and `docs/compression_bench.md` explain the intended
   interpretation of the reported metrics.

## Typical commands

Synthetic CA smoke benchmark:

```bash
python compression/compression_bench.py --height 48 --width 48 --steps 96
```

Video benchmark on a grayscale path:

```bash
python compression/video_bench.py path/to/video.mp4 --frames 60
```

Video benchmark with block reuse:

```bash
python compression/video_bench.py path/to/video.mp4 \
  --frames 60 \
  --block-reuse \
  --reuse-block 16 \
  --reuse-dict 256 \
  --reuse-planes 2
```

## Tests

- `tests/test_compression_bench.py`: smoke test for the synthetic benchmark.
- `tests/test_rans.py`: roundtrip tests for the range coder API.

## External dependencies and assumptions

- `video_bench.py` shells out to `ffmpeg` and `ffprobe`.
- The JAX-backed path described in `JAX/README.md` is reference-only on this
  machine; the plain NumPy / subprocess path remains the baseline.
- Many files in this directory are research artifacts, so names and interfaces
  are not yet normalized into a polished package API.

# Compression Bench Snapshot (2025-12-10 video + CA synthetic)

## Current codecs compared
- lzma (xz preset 6)
- gzip/zlib (level 6)
- `compression/rans.py` (real range coder with static frequency table)

## Bench 1: Synthetic triadic CA trace (48x48, 96 steps, seed=0)
Command:
```
python compression/compression_bench.py --height 48 --width 48 --steps 96
```
Result (bits per cell, lower is better):
- `raw_gate`: entropy 0.987; lzma 0.03 bpc, gzip/zlib 1.47 bpc (zlib-like via rANS)
- `residual_mod3`: entropy 0.102; lzma 0.03 bpc, gzip/zlib 0.03 bpc

Takeaway: simple temporal residuals crush the stream; general compressors already hit ~0.03 bpc. CA structure is highly predictable.

## Bench 2: Video sample `/home/c/2025-12-10 12-25-05.mp4` (grayscale, 1280x720, 60 frames)
Command:
```
python compression/video_bench.py "/home/c/2025-12-10 12-25-05.mp4" --frames 60
```
Results (bits per pixel, lower is better):
- `raw`: entropy 3.654; lzma 0.060 bpc (slow), gzip/zlib/rANS 1.473 bpc
- `residual`: entropy 0.245; lzma 0.058 bpc, gzip/zlib/rANS 0.076 bpc
- `coarse` (mirror around mid-gray): entropy 3.515; lzma 0.058 bpc, gzip/zlib/rANS 1.399 bpc
- `sign` (bright vs dark): entropy 0.352; lzma 0.004 bpc, gzip/zlib/rANS 0.268 bpc
- Multistream (coarse + sign, rANS/zlib): 1.667 bpc

Takeaway: Plain residuals already beat raw by ~20x; lzma wins on this small sample but is slower. Coarse/sign split alone is not enough; needs a true ANS coder plus better orbit/predictive modeling to approach lzma density without its latency.

## Bench 3: Same video, 600 frames, with canonicalized residual streams
Command:
```
python compression/video_bench.py "/home/c/2025-12-10 12-25-05.mp4" --frames 600
```
Results (bpc):
- `raw`: gzip/zlib/rANS ~1.418; lzma 0.017 (slow)
- `residual`: gzip/zlib/rANS 0.032; lzma 0.022
- `coarse` (mirror): gzip/zlib/rANS 1.334; lzma 0.017
- `sign`: gzip/zlib/rANS 0.220; lzma 0.002
- `coarse_resid`: gzip/zlib/rANS 0.032; lzma 0.021
- `sign_resid`: gzip/zlib/rANS 0.010; lzma 0.002
- Multistream (coarse+sign via rANS): 1.555 bpc
- Multistream (coarse_resid+sign_resid via rANS): 0.042 bpc

Takeaway: adding canonicalization and residuals on coarse/sign did not beat the simple temporal residual (0.032 bpc) with the current placeholder rANS; lzma still wins density but is very slow. A real ANS coder and better orbit/predictive modeling are needed to see a triadic win.

## Bench 4: Triadic planes + context + quotient (current)
Command:
```
python compression/video_bench.py "/home/c/2025-12-10 12-25-05.mp4" --frames 500
```
Observed (bpc, order of magnitude):
- Balanced-ternary planes, contexted: ~0.04 bpc
- Per-plane mag+sign quotient, contexted: ~0.04 bpc
- Test-only rates stay low but rise on short clips (insufficient frames).

Notes:
- Entropy collapse is driven by triadic digit planes + local spatio-temporal contexts.
- Train/test split requires enough frames (30+); very short clips overfit contexts.
- Motion compensation did not materially reduce residual support on this clip.

## Bench 5: Block reuse action stream (quotient over spatio-temporal repeats)
Command:
```
python compression/video_bench.py "/home/c/2025-12-10 12-25-05.mp4" --frames 60 --block-reuse --reuse-block 16 --reuse-dict 256 --reuse-planes 2
```
Observed:
- Action stream mix shows substantial reuse on static video.
- Masked plane coding with actions/refs reduces bpc further vs context-only.

Notes:
- Reuse is indexed by canonicalized block signatures (sign-normalized) over low planes.
- This is a first-order approximation of spatio-temporal quotient reuse.

## Gaps / TODO
- Add dictionary verification (e.g., plane-2 match) to reduce false reuse hits.
- Add motion-compensated reuse lookup and record side-information cost.
- Benchmark on real kernel/CA/motif logs and compare against zstd/xz.

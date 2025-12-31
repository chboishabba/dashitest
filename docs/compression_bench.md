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

## Bench 4: Triadic planes + context + quotient + JAX MC (current)
Command:
```
python compression/video_bench.py '/home/c/2025-12-10 12-25-05.mp4' \
  --jax-pipeline --jax-mc --mc-block 8 --mc-search 4 --frames 500
```
Observed (bpc):
- `raw`: entropy 3.967; lzma 0.020; gzip/zlib 1.370; rANS 3.967
- `residual`: entropy 0.108; lzma 0.026; gzip/zlib 0.037; rANS 0.112
- `coarse`: entropy 3.789; lzma 0.019; gzip/zlib 1.295; rANS 3.795
- `sign`: entropy 0.405; lzma 0.002; gzip/zlib 0.240; rANS 0.416
- `coarse_resid`: entropy 0.107; lzma 0.025; gzip/zlib 0.036; rANS 0.112
- `sign_resid`: entropy 0.009; lzma 0.003; gzip/zlib 0.010; rANS 0.020
- Multistream (coarse+sign via rANS): 4.212 bpc
- Multistream (coarse_resid+sign_resid via rANS): 0.132 bpc
- Base bt planes: rANS 0.216; ctx_rANS 0.039; ctx_rANS test-only 0.055
- Base bt mag+sign: rANS 0.216; mag ctx + sign 0.048; mag ctx + sign ctx 0.040; test-only 0.074
- MC bt planes: rANS 0.208; ctx_rANS 0.038; ctx_rANS test-only 0.052
- MC bt mag+sign: rANS 0.208; mag ctx + sign 0.047; mag ctx + sign ctx 0.039; test-only 0.071
- MC side info: mv_bpp 0.108 (block=8, search=4, mv_unique=81)

Notes:
- Entropy collapse is driven by triadic digit planes + local spatio-temporal contexts.
- Train/test split requires enough frames (30+); very short clips overfit contexts.
- Motion compensation did not materially reduce residual support on this clip.
- Source run logged in `trading/TRADER_CONTEXT.md:40017`.

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

## Bench 6: Color comparison (RGB vs YCoCg-R)
RGB baseline:
```
python compression/video_bench.py "/path/video.mp4" --frames 30 --color --color-transform rgb \
  --block-reuse --reuse-block 16 --reuse-dict 512 --reuse-planes 4
```

YCoCg-R:
```
python compression/video_bench.py "/path/video.mp4" --frames 30 --color --color-transform ycocg \
  --block-reuse --reuse-block 16 --reuse-dict 512 --reuse-planes 4
```

Notes:
- Color mode treats RGB channels independently unless YCoCg-R is selected.
- YCoCg-R encodes Y/Co/Cg magnitudes with the triadic pipeline and reports Co/Cg sign bpp separately.
- The output includes `rgb combined_total_bpp` or `ycocg combined_total_bpp` for total bpp comparisons.

## Gaps / TODO
- Add dictionary verification (e.g., plane-2 match) to reduce false reuse hits.
- Add motion-compensated reuse lookup and record side-information cost.
- Benchmark on real kernel/CA/motif logs and compare against zstd/xz.
- Decide whether to enable JAX x64 (or cast explicitly) to avoid dtype truncation warnings.
- Include MC side-info bpp in any combined-total reporting (residual + mv stream).

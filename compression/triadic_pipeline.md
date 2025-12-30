# Triadic Compression Pipeline Notes (Per-Plane Z2 Quotient)

This document records the current triadic video compression experiment in
`compression/video_bench.py`, focusing on the composition of p-adic refinement
with a per-plane Z2 quotient.

## Scope

The pipeline operates on grayscale video frames and targets lossless compression
for the signed temporal residual stream. The benchmark prints both zero-order
and contexted rANS results for each stream.

## Current Pipeline (Lossless)

1. **Decode**
   - Input video is decoded to grayscale frames `[T,H,W]` (`uint8`).

2. **Signed residual**
   - First frame: `base = frame0 - 128`.
   - Subsequent frames: `diff = frame_t - frame_{t-1}` (signed `int16`).
   - Residual stream is `base || diffs`.

3. **Balanced ternary expansion**
   - Expand residuals into balanced ternary digits:
     `r = sum_k a_k 3^k`, `a_k ∈ {-1,0,+1}`.
   - Planes are ordered from least significant digit to most.

4. **Per-plane Z2 quotient (new)**
   - For each digit plane `a_k`:
     - `mag_k = |a_k| ∈ {0,1}`
     - `sign_k = 1{a_k > 0}` gated by `mag_k == 1`
   - This is the per-plane quotient by inversion (`a_k ~ -a_k`) with a sign
     witness only where the magnitude is nonzero.

5. **Context models**
   - **Ternary planes:** contexted by left, up, previous frame, and previous
     plane (81 contexts).
   - **Magnitude planes:** binary contexted by left, up, previous frame, and
     previous plane (16 contexts).
   - Sign streams are currently zero-order coded (gated positions only).

6. **Entropy coding**
   - All streams are coded with the internal range coder (`rans.py`).

## Block reuse (spatio-temporal quotient, current)

This adds an explicit reuse action stream over blocks, treating repeated blocks
as equivalence classes under translation in time (and within-frame reuse).

Defaults:

- `block_size = 16`
- `dict_size = 256`
- `hash_planes = 2` (first two balanced-ternary planes)
- Actions: `NEW` (encode planes), `SAME` (reuse previous frame at same position), `REUSE` (dictionary hit)

Implementation notes:

- Each block is canonicalized by sign-normalization (flip if summed plane values are negative).
- A rolling dictionary maps block hashes to indices; `REUSE` emits the index.
- When `NEW` is false, planes are masked to neutral and only actions/refs/flip bits are coded.

This is a first-order “quotient by translation” step; it captures exact block
repeats without motion-compensated warping.

## Color modes (RGB vs YCoCg-R)

The benchmark supports either independent RGB channels or a reversible YCoCg-R
transform before triadic coding.

Modes:

- `--color --color-transform rgb`: treats R, G, B as independent grayscale streams.
- `--color --color-transform ycocg`: converts RGB into Y/Co/Cg (reversible), encodes
  Y/Co/Cg magnitudes with the triadic pipeline, and emits Co/Cg sign streams.

Output notes:

- `rgb combined_total_bpp`: sum of per-channel bpc (or block-reuse bpc when enabled).
- `ycocg combined_total_bpp`: sum of Y/Co/Cg bpc plus the Co/Cg sign stream bpp.

## GPU symbol-stream contract (Vulkan path)

The Vulkan path emits symbol streams that match the CPU compressor inputs.
This keeps the entropy coding logic unchanged while moving predictors and
digit-plane generation to the GPU.

### Block symbols (action/ref/flip)

One entry per block in scan order:
`block_id = by * blocks_x + bx`.

```
struct BlockSym {
    uint32 action;   // 0=NEW, 1=SAME, 2=REUSE
    uint32 ref;      // dictionary ref or temporal lag
    uint32 flip;     // sign flip bitmask (per-plane)
    uint32 aux;      // reserved
};
```

Total size: `blocks_x * blocks_y * 16` bytes.

### Residual planes (balanced ternary trits)

Planes are stored as `int32` trits in `{-1,0,+1}` with the index:

```
idx = plane * (W*H) + (y*W + x)
```

Total size: `planes * W * H * 4` bytes.

### Validation stub

`python vulkan/symbol_stream_stub.py ...` allocates these SSBOs, runs a
zero-writer compute kernel, and reads back to validate the GPU->CPU contract.

## Why the per-plane Z2 quotient matters

Balanced ternary exposes scale structure, but it does not remove sign symmetry.
Splitting each plane into magnitude and gated sign composes:

- **Scale refinement** (p-adic digits)
- **Symmetry reduction** (Z2 quotient)

This mirrors the earlier `coarse + sign` decomposition, but now applied at
every triadic scale rather than on raw pixel bytes.

## Benchmark outputs to track

In `compression/video_bench.py`, compare:

- `multistream (balanced ternary planes ctx_rANS)`  
  vs  
- `multistream (bt mag ctx + sign via rANS)`

A further drop indicates the quotient is reducing entropy beyond context
modeling alone.

## Next expected improvements

1. Context the gated sign stream using magnitude + neighbors.
2. Add simple backoff/smoothing for rare contexts.
3. Compare to LZMA on the same residual stream for fair baselines.

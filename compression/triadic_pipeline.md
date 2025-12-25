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


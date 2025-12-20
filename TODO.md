# TODO

- **Block-sparse int8/VNNI path**
  - Build `active_tiles` from P/Q/N or gate masks (tile-level).
  - Call a real VNNI/dpwssd microkernel on active tiles; emit once per tile.
  - Reuse tiles for multiple fused ops to amortize packing/mask build.

- **Order-ternary P/Q/N pipeline**
  - Extend P/Q/N culling into a tile selector feeding the block-sparse kernel.
  - Keep it bitset-only; no lane compaction in the hot loop.

- **Packed backward passes**
  - MoE: histogram-based gradients directly from packed tokens (per expert, per lane counts), no unpacked view.
  - CA: histogram-weighted trainer is in place; add packed feature counting and (optionally) symbolic rule extractor.

- **Dense-structure vs algebra experiment**
  - Swap in a real int8 microkernel (blocked/tiling) and test GF(3)/ternary arithmetic inside it to isolate “structure vs algebra”.
  - Record roofline stats (ops/s, GB/s, ops/byte) for each variant.

- **Life-like CA variant**
  - Add a non-monotone ternary rule (birth/death window or tie-noise) to produce sustained dynamics; compare learned vs true on multi-step rollouts.

- **Frontier/cluster benchmarks**
  - Apply hysteresis mode selection to other frontier-style benches (if any).
  - Add tile-level P/Q/N cull benchmark integrated with block-sparse runner.

- **GF(3) popcount path**
  - Swap in hardware popcnt via CFFI/Numba intrinsic for `gf3_check_bench.py`; re-measure.

- **Trading demo (ternary)**
  - Choose horizon/target (e.g., 1h next-bar {-1,0,+1} with vol-scaled dead-zone).
  - Build ternary/p-adic encoding, baseline binary model, ternary model, walk-forward backtest with costs; compare accuracy and trading metrics.
  - Optionally add a regime-gated MoE (trend/mean-revert/chop).


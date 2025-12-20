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

- **Downloader robustness**
  - Stooq: handle DNS/offline detection gracefully; skip on failure instead of retrying forever.
  - Yahoo: optional install; add Parquet output option in downloader.

- **Visualization**
  - Wire training_dashboard to actual log format from `ternary_trading_demo.py`.
  - Add action/HOLD overlays; add Hamming divergence/motif detection for CA visualiser.

- **Triadic exits/controls (trader)**
  - Exposure decay on HOLD (fast exit when field re-pins).
  - Volatility veto on size (shrink size when realized vol/latent velocity spikes).
  - State-stop exit (exit/reduce if latent velocity exceeds threshold mid-position).
  - Persistence ramp on size (slow ramp in new regime; clamp by vol target and hard cap).
  - Explicit thesis/persistence policy: learn/control {reinforce, hold, decay} separately from direction; add hazard/age inputs.
  - Add explicit persistence clocks (thesis_age, state_age, align_age) and feed them into the control decisions.
- **Execution realism**
  - Add execution layer interface: intent → fills/slippage/queue delay.
  - Implement BarExecution (existing) vs LOBReplayExecution (hftbacktest) switch.
  - Prepare Binance BTC/ETH book+trade schema for hftbacktest; integrate adapter.
  - Keep AUD/USD on bar execution (no L2 available).
  - Wire runner: strategy emits intents; route BTC/ETH to LOBReplayExecution, others to BarExecution. Define data schema (parquet/npz) for Binance book+trades.

- **Triadic exits/controls (trader)**
  - Exposure decay on HOLD (fast exit when field re-pins).
  - Volatility veto on size (shrink size when realized vol/latent velocity spikes).
  - State-stop exit (exit/reduce if latent velocity exceeds threshold mid-position).
  - Persistence ramp on size (slow ramp in new regime; clamp by vol target and hard cap).

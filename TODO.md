# TODO

- **Block-sparse int8/VNNI path**
  - Build `active_tiles` from P/Q/N or gate masks (tile-level).
  - Call a real VNNI/dpwssd microkernel on active tiles; emit once per tile.
  - Reuse tiles for multiple fused ops to amortize packing/mask build.

- **Order-ternary P/Q/N pipeline**
  - Extend P/Q/N culling into a tile selector feeding the block-sparse kernel.
  - Keep it bitset-only; no lane compaction in the hot loop.

- **Wave kernel benchmark follow-ups**
  - Decide whether to include a DC bias term in the dashifine spectral kernel.
  - Confirm the wave-domain periodicity handling vs Euclidean baselines.
  - Add a projection stress test (mask or subsample the input domain).
  - Interpret the top-10 eigenvalue printout against the temperature sweep.
  - Compare RBF/periodic RBF eigenspectra against dashifine temperature sweeps.

- **Generalised learner roadmap (kernel + wave task → broader suite)**
  - Phase 0: lock in baselines, spectra, interpretation, domain assumptions,
    reproducibility (seed flags), and documented outputs (text logs or plots).
  - Phase 1: stress geometry with out-of-band frequencies, mixed spectra,
    phase discontinuities, anisotropy, masked points, line sampling, and
    structured noise.
  - Phase 2: move to dynamics (e.g., reaction–diffusion, coupled oscillators),
    one-step prediction, rollout stability, and projected observations.
  - Phase 3: compare KRR vs `wave_kernel.py` on identical tasks, focusing on
    sample efficiency, failure modes, and projection sensitivity.
  - Phase 4: test non-Fourier invariants (CA with hidden parameters, procedural
    noise pipelines, graph signals, modular arithmetic).
  - Phase 5: study identifiability and grokking-style transitions by sweeping
    training size and tracking learned spectra.
  - Phase 6: formalize the hypothesis space, observation operator, spectral
    bias, and minimal conditions for success/failure; capture in a theory note.
  - Priority next: Phase 2 reaction–diffusion (Gray–Scott) one-step prediction,
    with dashifine vs periodic RBF comparison and output logging/plots.
  - Review Gray–Scott rollout curves and snapshot grids to compare U/V stability
    across kernels using per-component MSE logs.
  - Interpret conserved-quantity logs (mean U/V, total mass) to distinguish
    global drift from structural failure.
  - Add a primes/divisibility benchmark series:
    - p-adic valuation (v_p) or divisibility indicator tasks.
    - Sieve-step prediction on a bitmask state.
  - Extend primes benchmark to sieve-step or primality classification once the
    divisibility/v_p baselines are stable.
  - Review valuation-level indicator results to confirm hierarchy-aligned loss
    is behaving as expected.

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

- **Triadic hysteresis precision–recall sweep (Option 1)**
  - Freeze legitimacy: keep `RegimeSpec` at stability-only (`min_run_length=3`; no vol/flip-rate gates).
  - Fix `tau_on=0.5`; sweep `tau_off ∈ {0.30, 0.35, 0.40, 0.45}` to narrow hysteresis only.
  - Log per-sweep metrics: `acceptable%`, `P(acceptable|ACT)` (precision), `P(ACT|acceptable)` (recall), trades/HOLD%.
  - Stop sweep if precision drops sharply; keep PnL completely out of the loop.
  - Add a tiny PR table/sparkline to the dashboard fed by the sweep output (tau_off → precision/recall).

- **Motif CA diagnostics**
  - Wire motif CA (M4/M7/M9) into visualiser (true vs learned) to see corridors/rims/absorbing basins; optionally toggle levin mode.
  - Add CA hysteresis/confidence sweep (softmax margin → tau_on/tau_off, k_on/k_off) mirroring trading PR knee; report precision/recall vs acceptable.
  - Document mapping: ACT/HOLD/RED ↔ regime gate; corridor/tolerance/absorption motifs ↔ trading behaviors.

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

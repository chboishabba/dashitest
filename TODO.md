# TODO

- **Block-sparse int8/VNNI path**
  - Build `active_tiles` from gate masks (tile-level any-activation) in
    `dashilearn/bsmoe_train.py`.
  - Priority rationale: see `CONTEXT.md#L2626`.
  - Call a compiled int8 microkernel on active tiles (via
    `dashilearn/vnni_kernel.so`); emit once per tile.
  - Reuse tiles for multiple fused ops via a `TilePlan` (active tile offsets,
    optional packed buffers) and benchmark a fused sequence with per-op timing
    breakdown.
  - Add plan caching across epochs: reuse prior TilePlan when Jaccard(tile set)
    >= threshold; log hit rate, similarity, and timing breakdown.

- **Vulkan/JAX parity map**
  - Inventory Vulkan entry points (`vulkan/`, `vulkan_compute/`) and map
    corresponding JAX reference modules (`JAX/`) to clarify what can be ported
    without relying on JAX at runtime.
  - First Vulkan kernel: block-wise residual/diff with per-block stats (SAD or
    energy) as a parity target for `JAX/motion_search.py`.
- **Vulkan sheet visual shader**
  - Wire `vulkan_compute/shaders/sheet_expand_fade.comp` into a small host
    driver (e.g., `compute_image_preview.py`) with SSBO + accumulator + output
    image bindings, plus push-constant controls.

- **Order-ternary P/Q/N pipeline**
  - Extend P/Q/N culling into a tile selector feeding the block-sparse kernel.
  - Keep it bitset-only; no lane compaction in the hot loop.

- **Wave kernel benchmark follow-ups**
  - Decide whether to include a DC bias term in the dashifine spectral kernel.
  - Confirm the wave-domain periodicity handling vs Euclidean baselines.
  - Add a projection stress test (mask or subsample the input domain).
  - Interpret the top-10 eigenvalue printout against the temperature sweep.
  - Compare RBF/periodic RBF eigenspectra against dashifine temperature sweeps.
  - Locate `dashifine/newtest1` + `dashifine/newtest2` energy landscape outputs
    and visualize with `plot_energy_landscape.py` (heatmap + slope).
  - Re-run the tree diffusion benchmark and record the tree-band quotient
    metrics/plots (detail-band energies) alongside existing quotient metrics.
  - Sweep `--adv-band`/`--adv-style` (randphase/sparse/mix) to compare band
    kill rates and leakage between RBF and tree kernels.
  - Add a symmetry-breaking tree diffusion variant (depth-varying diffusion or
    non-commuting observation map) to force model separation.
  - Lock design choices for the next phase: local/adjacent band coupling (vs
    global), static adversary (vs adaptive), coarse→fine vs fine→coarse bridge
    direction, and relative pass/fail thresholds.
  - Add an adversarial operator variant with nonlinear band coupling
    (`x_{t+1} = A x_t + g(B(x_t))`) and a flag to toggle it.
  - Add a bridge-task evaluation (infer `x_{T/2}` or band energies from `(x0, xT)`
    and score in band-quotient space with leakage reporting).
  - Implement the adversarial-operator toggle with local band coupling as the
    default path once the design choices are locked.
  - Implement the bridge-task evaluation (coarse↔fine) with relative thresholds
    once the design choices are locked.
  - Decide adversarial-operator specifics: resize method (nearest/bilinear/area),
    kernel support, polynomial degree, and per-band eps_j schedule.
  - Decide leakage measurement details: band norm, ablation replacement
    strategy (zero vs noise), and delta stability constant.
  - Define bridge-task acceptance thresholds relative to baseline, and confirm
    the baseline implementations (upsample+proj vs downsample).
  - Add core dashboard outputs (influence matrix, leakage summaries, bridge
    asymmetry plots, leakage-vs-error scatter) and file naming.
  - Add strongly recommended visuals (kill-rate/collapse-step plot and operator
    strength sweep panel).
  - Add nice-to-have sample visualizations (band images/energies pre/post
    adversary plus bridge reconstructions).
  - Define closure thresholds (baseline margin, NonLocalLeak factor, asymmetry
    gap limit, correlation magnitude) and record them in the benchmark spec.
  - Add a reproducibility checklist (>=2 seeds, stable separation, no new
    failures) for final closure runs.
  - Add discriminator checks (kernel-matrix diff, distance-matrix correlation,
    sparse impulse initializations) for the tree diffusion benchmark.
  - Re-run the tree diffusion benchmark with the quotient-kernel tree model and
    record whether rollout/quotient separation appears.

- **Compression / codec follow-ups**
  - Re-run the tree diffusion benchmark with `--dump-band-planes` and review
    per-band rollout sheets (optionally stitch GIFs).

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
  - Implement Gray-Scott quotient-space rollout metrics for V+radial(U) per
    `docs/grayscott_quotient.md`.
  - Locate `dashifine/newtest/grayscott_krr.py` (not present in this repo) or
    import it before adding quotient metrics.
  - Review Gray–Scott rollout curves and snapshot grids to compare U/V stability
    across kernels using per-component MSE logs.
  - Interpret conserved-quantity logs (mean U/V, total mass) to distinguish
    global drift from structural failure.
  - Track runtime for Gray-Scott KRR (GIF export can take ~25m on CPU); consider
    caching or runtime flags if this becomes a bottleneck.
  - Add a primes/divisibility benchmark series:
    - p-adic valuation (v_p) or divisibility indicator tasks.
    - Sieve-step prediction on a bitmask state.
  - Extend primes benchmark to sieve-step or primality classification once the
    divisibility/v_p baselines are stable.
  - Review valuation-level indicator results to confirm hierarchy-aligned loss
    is behaving as expected.
  - Add valuation-only targets (valuation vector, max prime power, ultrametric
    distance) per `docs/valuation_primes_plan.md`.

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

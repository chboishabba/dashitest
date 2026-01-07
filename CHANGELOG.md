# Changelog

## Unreleased
- Block-sparse MoE training now derives tile masks from gate activity instead
  of random tile sampling, keeping tile density aligned with the configured
  target.
- Added a benchmark closure criteria section to the tree diffusion spec,
  defining design-freeze rules, required outputs, acceptance logic, and
  reproducibility/stopping conditions.
- Reframed the tree diffusion dashboard outputs as learner-centric diagnostics
  and added a minimal learner-dashboard checklist to the benchmark spec.
- Expanded the tree diffusion benchmark docs with recommended/nice-to-have
  visual outputs and an explicit GIF/WebM consolidation note for bulk images.
- Added detailed definitions and acceptance-test specs for the band-coupled
  adversarial operator and bridge task in the tree diffusion benchmark docs,
  plus required output plots and naming guidance.
- Documented the next-phase design commitments for tree diffusion (operator
  coupling scope, adversary type, bridge direction, and relative thresholds).
- Documented the tree diffusion adversarial-operator and bridge-task extensions,
  and queued the corresponding TODOs for band-coupled dynamics and two-sided
  inference evaluation.
- Documented that the triadic video codec benchmark is stats-only and queued
  plane-visualization output as a follow-up.
- Tree diffusion benchmark now trains the tree kernel on quotient features
  (`quotient_vector`) to break permutation equivalence.
- Documented a valuation-only primes benchmark plan aligned with valuation depth.
- Documented the long runtime for Gray-Scott KRR GIF export runs in the README.
- Added a quotient-invariant integration spec for the triadic trader and
  logged `q_*` quotient features (E/C/S + deltas) per step.
- Fixed `run_trader.py` geometry plot invocation to resolve the correct
  `trading/scripts/plot_decision_geometry.py` path.
- Added quotient stability reporting and plotting helpers, plus a generic
  energy landscape visualizer for dashifine outputs.
- Documented timestamped plot output naming to avoid overwriting prior runs.
- Plot scripts now auto-timestamp `--save` outputs (trading plots and
  `plot_energy_landscape.py`) to prevent overwrites.
- Added a tree diffusion benchmark spec and a runnable script to compare
  Euclidean vs tree-geometry KRR under quotient metrics.
- Tree diffusion benchmark now supports optional rollout plots via `--plots`.
- Tree diffusion benchmark now reports a tree-intrinsic depth-energy quotient.
- Tree diffusion benchmark now reports tree-band detail energies and a band
  quotient rollout plot.
- Documented the sheet-identity mapping between codec plane dumps and tree-band
  detail sheets, plus the `--dump-planes` visualization hook.
- Added `--dump-band-planes` to `tree_diffusion_bench.py` to emit per-band,
  per-step sheet PNGs for codec-style visualization.
- Added band-plane visualization modes (symmetric normalization, energy-height
  scaling, and ternary thresholding) to align tree sheets with codec planes.
- Added `--init-band` to seed tree diffusion with energy isolated to a single
  band for depth-killing separation tests.
- Added adversarial band init flags (`--adv-band`, `--adv-style`, `--adv-sparse-m`,
  `--adv-mix-band`, `--adv-mix-eps`, `--adv-seed`) to stress depth-killing.
- Documented a cleanup guideline to consolidate bulk PNG dumps into GIFs and
  remove the individual PNGs.
- Documented the null-separation/control-case interpretation for tree diffusion
  and queued a symmetry-breaking variant.
- Added a brief control-case interpretation note to the tree diffusion spec.
- Documented discriminator checks for the tree diffusion benchmark.
- Documented permutation-invariance caveat and queued a quotient-aware kernel
  for the tree diffusion benchmark.
- Documented Gray-Scott quotient-space evaluation metrics and selected the
  V+radial(U) quotient target for rollout evaluation.
- Benchmark scripts now emit timestamped run subdirectories under `--output_dir`
  to avoid overwriting prior outputs.
- Added valuation-level indicator targets to the primes/divisibility benchmark
  and a summary plot for hierarchy-aligned loss.
- Documented primes/divisibility interpretation guidance and queued
  valuation-indicator targets to align loss with hierarchy.
- Added optional rollout GIF export (with frame cleanup) to the Gray–Scott
  operator-learning benchmark.
- Added a primes/divisibility KRR benchmark with saved plots and summaries in
  `newtest/primes_krr.py`.
- Documented the field-comparison vs rollout distinction and queued a
  primes/divisibility benchmark series in the roadmap.
- Added per-component U/V rollout MSE logging to the Gray–Scott benchmark CSV.
- Added Gray–Scott multi-step rollout diagnostics (error curves, snapshots, and
  conserved-quantity logs) to the operator-learning benchmark.
- Documented the Gray–Scott U/V interpretation and queued multi-step rollout
  diagnostics in the roadmap.
- Added a Gray–Scott operator-learning KRR benchmark with logged summaries and
  saved spectra/field plots in `newtest/grayscott_krr.py`.
- Added a wave-field baseline comparison summary (dashifine vs RBF/pRBF) and
  recorded the Gray–Scott operator-learning priority in documentation/TODOs.
- Documented output-capture expectations for wave benchmarks and added a
  staged roadmap for the generalised learner experiments.
- Added RBF and periodic RBF baselines (with eigenspectrum diagnostics) to the
  wave KRR benchmark for direct spectral alignment comparisons.
- Added kernel eigenspectrum diagnostics to the wave KRR benchmark so the
  temperature sweep is tied to the underlying spectral geometry.
- Documented interpretation/claims for the wave-kernel temperature sweep and
  recorded next-step diagnostics in the dashifine README.
- The placeholder `dashifine/Main_with_rotation.py` entry point now logs the
  output image paths when run as a script for easier verification.
- Added a dashifine spectral kernel module for PSD kernel experiments and a
  wave-field KRR benchmark script to test wave-aligned generalization.
- Compression bench upgrades:
  - Replaced the lzma shim with a real range coder in `compression/rans.py`.
  - Added balanced-ternary digit planes, per-plane Z2 quotient (mag + gated sign), and contexted coding in `compression/video_bench.py`.
  - Added block reuse actions, reference stream, and masked-plane coding to approximate spatio-temporal quotient reuse.
  - Added optional motion-compensation pyramid search and train/test context split reporting.
  - Added color benchmarking with RGB and reversible YCoCg-R transforms, plus combined total bpp reporting.
  - Documented the triadic pipeline and quotient composition in `compression/triadic_pipeline.md`.
- Added severity-ranked bad-window tooling and docs:
  - `scripts/score_bad_windows.py` computes synthetic bad flags (|return| vs σ or drawdown slope) and ranks top-N windows by summed `p_bad` over a sliding window.
  - News fetch summaries now include severity (`sev_sum_p_bad`) and cap windows/days by triggers to avoid spam; 404s/future dates are treated as empty fetches.
  - `docs/bad_day.md` and `README.md` now describe regime windows (hazard view), tri-state `p_bad` posture policy, and “legal” BAN monetisation (structure/vol/fee avoidance, not forced shorting).
- Added `scripts/contextual_news.py` to aggregate contextual signals per date:
  - Reuters markets RSS (no key) filtered by keywords.
  - TradingEconomics macro calendar if `TE_API_KEY` is set; empty otherwise.
  - Stress proxies via yfinance (VIX, USDCNH, Copper) when available.
- Added `trading/training_dashboard_pg.py`: PyQtGraph-based live dashboard with price/actions/bad_flag shading, PnL+HOLD%, p_bad+bad_flag step fill, and volume panes for fast intraday visualization.
- Recorded historical `trading/run_all.py` behavior before the latest changes (single-threaded, no live view) and captured example results for the legacy run.
- Captured pre/post `trading/run_all.py` outputs for reference:
  - Legacy (5 markets): total_pnl=362,634.4097 with per-market PnL: aapl.us 99,142.2711; btc.us 99,906.5616; btc_intraday -29,813.2553; msft.us 97,964.1213; spy.us 95,434.7109.
  - After added BTC sources (7 markets): total_pnl=1,314,977.2228 with per-market PnL: btc_yf 859,973.1025; btc.us 99,906.5616; aapl.us 99,142.2711; msft.us 97,964.1213; spy.us 95,434.7109; btc_intraday_1s 92,369.7107; btc_intraday -29,813.2553.
- Added Binance-backed BTC downloads (extended 1m window, ~10h 1s bars) and wired them into the downloader CLI; `run_trader` now prefers the richer BTC files.
- Extracted `run_trader.run_trading_loop` so the trading sim can be reused across markets without clobbering logs.
- Extended `trading/run_all.py` to run every cached market, print a scoreboard, and optionally stream a live dashboard via `--live`.
- Added a project map to `README.md` describing the key scripts and components.

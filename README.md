## Project map (what each file does)
- Plot scripts auto-timestamp `--save` outputs to avoid overwriting prior runs (see `CONTEXT.md#L2532`).
- When a script dumps many PNGs (e.g., plane/band rollouts), roll them into a GIF
  and delete the individual PNGs to keep outputs manageable.
- `trading/data_downloader.py`: Fetch BTC data (daily/intraday/1s) from Binance/Yahoo/CoinGecko/Stooq; saves under `data/raw/stooq`.
- `trading/run_trader.py`: Core bar-level trading loop with simple impact/fee model; writes `logs/trading_log.csv` and `data/run_history.csv`.
- `trading/run_all.py`: Runs the trading loop across all cached markets (or synthetic), prints a scoreboard; `--live` streams the dashboard while running.
- `trading/run_all_two_pointO.py`: Orchestrator that can (a) run all markets with progress prints, (b) sweep tau_off with live PR/PnL Pareto plots, and (c) preview the CA tape on the same CSV.
- `trading/training_dashboard.py`: Live viewer for logs (PnL, price/actions, latent/HOLD%, optional PR/engagement overlays).
- `trading/training_dashboard_pg.py`: PyQtGraph live viewer (fast, multi-pane).
- `trading/runner.py`: Strategy + execution wiring for batch bar runs (used by scripts).
- `trading/scripts/run_bars_btc.py`: Build BTC bars/state, run the bar executor, write a trading log.
- `trading/scripts/sweep_tau_conf.py`: Sweep hysteresis thresholds (tau_on/off), produce PR curves for the dashboard sparkline.
- `trading/scripts/posture_returns.py`: Compute bar- and window-level returns by posture (ACT/HOLD/BAN) with cumulative sums.
- `trading/`: Trading core (execution backends, intent datatypes, regime gating, and strategies).
- `trading/scripts/ca_epistemic_tape.py`: Trading-driven CA visualization (epistemic tape) that injects triadic market features into a 2D CA and plots snapshots, motif triggers, and multiscale change rates. Research/diagnostic only (does not drive trading).
- Vulkan/JAX reference points (JAX is reference-only on this machine):
  - `vulkan/README.md`: Vulkan video bench path + dmabuf/VAAPI notes.
  - `vulkan_compute/README.md`: Vulkan compute prototypes (buffer/image/preview).
  - `vulkan_compute/operator_train.py`: Vulkan-native OperatorLearner v0 driver that writes `dashilearn/sheet_energy.npy`.
  - `JAX/README.md`: JAX codec/motion/pipeline reference implementations.
  - `docs/vulkan_jax_parity.md`: Vulkan↔JAX parity map and recommended first Vulkan kernel (see `CONTEXT.md#L18786` and `CONTEXT.md#L18950`).

## Full file index (every file/test)
- `CHANGELOG.md`: Change history and release notes.
- `CONTEXT.md`: Context notes for the repo and experiments.
- `README.md`: Project overview and file index (this file).
- `TODO.md`: Task backlog and future ideas.
- `action_entropy.png`: Generated plot asset (action entropy).
- `acceptable.png`: Generated plot asset (acceptable region).
- `acceptable1.png`: Generated plot asset (acceptable variant).
- `accept_persistence.png`: Generated plot asset (accept persistence).
- `balanced_pn_iter_bench.py`: Benchmark for balanced P/N iterative add loops.
- `dashilearn/bsmoe_train.py`: Training script for block-sparse MoE models with gate masks, tile-plan reuse across fused ops, plan caching across epochs, and band-energy exports to `dashilearn/sheet_energy.npy` for the visualiser; supports `--epochs` and `--stay-open` so you can keep the run alive for the preview (see `CONTEXT.md#L18720`, `CONTEXT.md#L21231`, and `CONTEXT.md#L20482`).
- `dashilearn/bsmoe_train.py`: Training script for block-sparse MoE models with gate masks, tile-plan reuse across fused ops, plan caching across epochs, and band-energy exports to `dashilearn/sheet_energy.npy` for the visualiser; supports `--epochs` and `--stay-open` so you can keep the run alive for the preview (see `CONTEXT.md#L18720`, `CONTEXT.md#L21231`, and `CONTEXT.md#L20482`). Added `--plan-hit-experiment` with `--plan-hit-block-size`/`--plan-hit-perms`, `--observer-class` (scalar/corr), and richer regime controls (`--regime-mode`, `--gate-density-bins`, `--regime-alternation-interval`) so you can exercise the Stage A/B observer tests described around `CONTEXT.md#L32387-L34282` and print regime stats + entropy for every run.
- Running `dashilearn/bsmoe_train.py --phase3` now also timestamps and saves a JSON log under `logs/bsmoe_train/bsmoe_phase3_<timestamp>.json` plus a `outputs/bsmoe_phase3_<timestamp>.png` graph of `task_loss`, `quotient_loss`, and `mdl_cost`, keeping each Phase-3 run reproducible per the logging checklist in `docs/phase3_quotient_learning.md`.
- Task B+ observability experiments (see `CONTEXT.md#L36915-L36955`): the plumbing now writes one semantic tile per epoch (gate density/active tile ratio) and logs the local mean so the learner → Vulkan → CPU chain is visible. Decide whether to run the alternating control, climb the observer ladder (correlations → tiny CNN), or pause/productize now that the instrumentation is stable before adding further artifactual tweaks.
- `dashilearn/run_live_sheet.sh`: Helpers to launch the learner + Vulkan preview with recording so the sheet data gets produced and consumed automatically.
- `block_sparse_tile_driver.py`: Driver for block-sparse tile experiments.
- `ca_Figure_1.png`: Generated CA figure asset.
- `ca_vis_Figure_0.png`: Generated CA visualization asset.
- `ca_vis_Figure_1.png`: Generated CA visualization asset.
- `ca_vis_Figure_2.png`: Generated CA visualization asset.
- `ca_vis_Figure_3.png`: Generated CA visualization asset.
- `ca_visualiser.py`: CA visualization utility.
- `cluster_cull_bench.py`: Benchmark for cluster culling routine.
- `cluster_cull_pqn_bench.py`: Benchmark for PQN cluster culling.
- `compression/__init__.py`: Compression module init.
- `compression/comp_ca.py`: CA generator/visualizer for compression experiments.
- `compression/compression_bench.py`: Triadic CA compression benchmark.
- `compression/compression_context.txt`: Long-form compression notes and hypotheses.
- `compression/gpt2.py`: Compression experiment notes (gpt2).
- `compression/gpt3.py`: Compression experiment notes (gpt3).
- `compression/gpt4.py`: Compression experiment notes (gpt4).
- `compression/naieve_i_think.py`: Sketch/notes for naive compression idea.
- `compression/rans.py`: Range coder with rANS-like API.
- `compression/triadic_pipeline.md`: Notes on triadic pipeline + per-plane Z2 quotient.
- `compression/video_bench.py`: Video compression benchmark runner.
- `confusion.png`: Generated plot asset (confusion surface).
- `dashitest.old.keepme.py`: Legacy benchmark harness (kept for reference).
- `dashitest.py`: SWAR XOR consumer benchmark driver.
- `trading/data_downloader.py`: Data fetcher for BTC market sources.
- `dir_legit.png`: Generated plot asset (direction legitimacy).
- `docs/bad_day.md`: Bad-day detection concept write-up.
- `docs/compression_bench.md`: Compression benchmark notes.
- `docs/tree_diffusion_benchmark.md`: Tree diffusion benchmark spec (ultrametric transport) plus the Codec Task A reference-run summary and its JSON/rollout assets.
- `scripts/codec_task_a_summary.py`: Generate Markdown summaries from a `tree_diffusion_metrics_*.json` run plus the matching rollout figures.
- `scripts/bridge_task.py`: Universal Task B bridge runner for codec/DNA `E_seq.npy` sequences.
- `scripts/bridge_task_summary.py`: Print the Task B bridge run summary, leakage ratio, and figure list.
- `scripts/gen_codec_E_seq.py`: Dump codec `E_seq.npy` from balanced-ternary plane npz.
- `scripts/gen_dna_E_seq.py`: Compute DNA `E_seq.npy` from sliding Haar windows over a FASTA.
- `docs/tree_diffusion_benchmark.md`: Now also documents Task B as the epistemic guardrail so learners must respect the `E_seq` geometry before moving on (see `CONTEXT.md#L26230`–`CONTEXT.md#L26699`).
- `docs/valuation_primes_plan.md`: Valuation-only primes benchmark plan.
- `docs/energy_landscape_vis.md`: Energy landscape visualization notes for dashifine outputs.
- `first_exit.png`: Generated plot asset (first-exit heatmap).
- `five_trit_pack_bench.py`: Benchmark for 5-trit packing.
- `fn_anatomy.png`: Generated plot asset (function anatomy).
- `frontier_mask_bench.py`: Benchmark for frontier mask routine.
- `fused_iter_bench.py`: Benchmark for fused iter XOR/threshold/dot loop.
- `gf3_check_bench.py`: Benchmark for GF(3) check routines.
- `gf3_parity_bench.py`: Benchmark for GF(3) parity routines.
- `gpt_attach_1.py`: Attachment helper script (gpt_attach_1).
- `gpt_attach_2.py`: Attachment helper script (gpt_attach_2).
- `gpt_attach_3.py`: Attachment helper script (gpt_attach_3).
- `homology.png`: Generated plot asset (homology).
- `hysteresis_phase.png`: Generated plot asset (hysteresis phase).
- `levin_ca_train.py`: Training script for Levin-style CA.
- `margin.png`: Generated plot asset (legitimacy margin).
- `microstructure.png`: Generated plot asset (microstructure).
- `moe_fair_bench.py`: Benchmark for fair MoE routing.
- `moe_fair_train.py`: Training script for fair MoE model.
- `moe_sparse_bench.py`: Benchmark for sparse MoE routing.
- `motif_ca.py`: Motif CA implementation.
- `motif_ca_gpt-gen-lol.py`: Generated motif CA variant.
- `notebooklm.chat`: Saved chat transcript/notes.
- `optimal_packings_table.py`: Computes optimal trit packings table.
- `packing_ablation_bench.py`: Benchmark for packing ablations.
- `policy_curvature.png`: Generated plot asset (policy curvature).
- `policy_distance.csv`: Output CSV for policy distance sweep.
- `potts3_bench.py`: Potts model triadic benchmark.
- `plot_energy_landscape.py`: Plot 2D energy landscapes (heatmap + optional slope).
- `trading/base.py`: Base execution interface.
- `trading/bar_exec.py`: Bar-level execution backend.
- `trading/hft_exec.py`: HFT/LOB execution stub.
- `trading/intent.py`: Intent datatypes and helpers.
- `trading/regime.py`: Regime model and helpers.
- `regime_surface.png`: Generated plot asset (regime surface).
- `regime_surface1.png`: Generated plot asset (regime surface variant).
- `trading/run_all.py`: Batch runner over cached markets.
- `trading/run_all_two_pointO.py`: Extended runner with sweeps and CA previews.
- `trading/run_trader.py`: Main bar-level trading simulator.
- `trading/runner.py`: Strategy + execution orchestrator.
- `trading/scripts/ca_epistemic_tape.py`: Epistemic tape CA generator and plots.
- `trading/scripts/compute_policy_distance.py`: Compute policy distance metrics.
- `trading/scripts/contextual_news.py`: Fetch/overlay news for windows.
- `trading/scripts/emit_news_windows.py`: Emit news windows for bad-day periods.
- `trading/scripts/news_slice.py`: Fetch news slice for a time window.
- `trading/scripts/plot_accept_persistence.py`: Plot accept persistence diagnostics.
- `trading/scripts/plot_acceptability.py`: Plot acceptability surface.
- `trading/scripts/plot_action_entropy.py`: Plot action entropy diagnostics.
- `trading/scripts/plot_confusion_surface.py`: Plot confusion surface.
- `trading/scripts/plot_direction_legitimacy.py`: Plot direction legitimacy.
- `trading/scripts/plot_engagement_surface.py`: Plot engagement surface.
- `trading/scripts/plot_first_exit_heatmap.py`: Plot first-exit heatmap.
- `trading/scripts/plot_fn_anatomy.py`: Plot function anatomy.
- `trading/scripts/plot_hysteresis_phase.py`: Plot hysteresis phase.
- `trading/scripts/plot_legitimacy_margin.py`: Plot legitimacy margin.
- `trading/scripts/plot_manifold_homology.py`: Plot manifold homology.
- `trading/scripts/plot_microstructure_overlay.py`: Plot microstructure overlay.
- `trading/scripts/plot_policy_curvature.py`: Plot policy curvature.
- `trading/scripts/plot_regime_surface.py`: Plot regime surface.
- `trading/scripts/plot_temporal_homology.py`: Plot temporal homology.
- `trading/scripts/plot_vector_field.py`: Plot vector field.
- `trading/scripts/rollup_bad_days.py`: Aggregate bad-day scores per day.
- `trading/scripts/run_bars_btc.py`: Build bars and run bar executor on BTC.
- `trading/scripts/score_bad_windows.py`: Score synthetic bad windows.
- `trading/scripts/sweep_confusion_surface.py`: Sweep confusion surface parameters.
- `trading/scripts/sweep_motif_hysteresis.py`: Sweep motif hysteresis parameters.
- `trading/scripts/sweep_regime_acceptability.py`: Sweep regime acceptability thresholds.
- `trading/scripts/sweep_tau_conf.py`: Sweep tau thresholds for PR curves.
- `snapshot_bench.py`: Snapshot benchmark helper.
- `sparse_iter_classifier_bench.py`: Benchmark for sparse iterative classifier loop.
- `trading/strategy/triadic_strategy.py`: Triadic strategy logic.
- `surface.png`: Generated plot asset (surface).
- `surface1.png`: Generated plot asset (surface variant).
- `svo_traversal_bench.py`: SVO traversal benchmark.
- `swar_test_harness.py`: SWAR test harness for ternary ops.
- `temporal_homology.png`: Generated plot asset (temporal homology).
- `ternary_alu_micro_bench.py`: Micro benchmark for ternary ALU ops.
- `ternary_life_ca.py`: Ternary Life CA simulation.
- `ternary_life_visualiser.py`: Visualizer for ternary Life CA.
- `trading/ternary_trading_demo.py`: Demo of ternary trading logic.
- `trading/test_trader_real_data.py`: Manual trader test on real data.
- `tests/test_compression_bench.py`: Test for compression bench smoke.
- `tests/test_rans.py`: Test for rANS/range coder roundtrip.
- `tests/test_training_dashboard_pg.py`: Test for PyQtGraph dashboard smoke.
- `tiled_ternary_dot_bench.py`: Tiled ternary dot product benchmark.
- `trading/training_dashboard.py`: Live training dashboard (matplotlib).
- `training_dashboard_Figure_1.png`: Generated dashboard figure asset.
- `training_dashboard_Figure_2.png`: Generated dashboard figure asset.
- `trading/training_dashboard_pg.py`: PyQtGraph live dashboard.
- `triadic_nn_bench.py`: Triadic NN benchmark.
- `triadic_nn_bench2.py`: Triadic NN benchmark (packed SWAR).
- `vector_field.png`: Generated plot asset (vector field).
- `z3_phase_fixedpoint_bench.py`: Z3 phase fixed-point benchmark.

## CA metrics wish-list (diagnostics, not implemented yet)
To keep research and trading separate, CA metrics live in the lab. The key statistics we intend to collect when iterating on the CA prototypes:
- **State occupancy:** Fractions of s∈{-1,0,+1}, g∈{BAN,HOLD,ACT}, and joint (s,g)/(s,φ) per step and aggregated.
- **Transitions/persistence:** Transition matrices (s_t→s_{t+1}, g_t→g_{t+1}), run-lengths of ACT/HOLD/BAN, flip rates.
- **Motion/propagation:** Spatial autocorrelation, phase/velocity bias, glider density, lifetimes of moving motifs.
- **Tension/conflict:** Local τ=min(c⁺,c⁻), its distribution/variance, duration above thresholds.
- **Motif triggers:** Counts per step of M4 (corridor), M7 (fatigue rim), M9 (shutdown), plus joint events.
- **Multiscale structure:** Change rate vs coarse-graining k; entropy vs scale; cross-scale mutual information.
- **Fatigue/memory:** Mean/variance/tails of fatigue; autocorrelation decay of s/g/τ to estimate effective memory depth.
- **Market-driven CA (when forced by real data):** Input alignment (correlation of injected symbols vs CA state), lagged influence; identification of corridors, fracture zones, shutdown islands.
- **Bridge to trading (epistemic only):** Permission surface stability (ACT fraction, volatility, clustering); inferred hysteresis (empirical tau_on/off, knee points from CA stats).
- **Pathology checks:** Collapse (all 0), white-noise (flat entropy across scales), limit cycles, over-ban (M9 dominance).

## PnL / finance metrics (audit stream, not “acceptable” criteria)
To track outcomes alongside epistemic metrics (e.g., per tau_off sweep point):
- **Per-bar fields (loggable):** price_t, ret_t, position_t, Δpos_t, fill_qty/price, fees_t, impact/slippage_t, pnl_gross_t, pnl_net_t, equity_t.
- **Per-run summary:** total gross/net PnL, mean/stdev bar return, Sharpe/Sortino (consistent horizon), max drawdown, Calmar.
- **Costs:** total fees; total impact/slippage; cost ratio ((fees+impact)/gross profit or turnover); PnL per trade (mean/median); tail losses per trade (e.g., 5th pct).
- **Trading intensity:** #trades; turnover (∑|Δpos| or notional traded); time in market; ACT bars vs fills.
- **Robustness:** win rate; profit factor (gross wins/gross losses); avg win/avg loss; exposure-weighted return; “edge after costs” = E[Δequity]/∑|Δpos|.
- **Sweep reporting (per tau_off):** keep epistemic axes (acceptable%, precision, recall, act_bars, hold%) and add mean return, max DD, turnover/trades, fees+impact, net PnL. Useful plots: (1) Precision–Recall annotated with net PnL/max DD; (2) Net PnL vs Max DD (Pareto), colored by tau_off, sized by turnover.

## Structural stress / bad-day flag (implemented)
- `trading/run_trader.py` now computes a crude structural stress score (vol z-score vs median/MAD, jump z, triadic flip rate) and logs `p_bad` ∈ [0,1] plus `bad_flag` (p_bad>0.7) per bar. Progress prints include these.
- `bad_flag` is sovereign: when `bad_flag=1`, the trader flattens and refuses new exposure (BAN posture). A `ban` column is logged per bar.
- Logs now include `ts` (timestamp when available from source data) so you can align `bad_flag` spikes with news or calendar events. Use `trading/scripts/rollup_bad_days.py` to aggregate per-day bad scores for quick headline overlays.
- `trading/scripts/news_slice.py` fetches headlines/events for a time window: `--provider newsapi` (needs `NEWSAPI_KEY`), `--provider gdelt` (no key), or `--provider rss --feed-url <url>` (no key; repeatable) to correlate with bad-day spikes.
- `trading/scripts/emit_news_windows.py` auto-detects bad windows from a trading log (p_bad/bad_flag) and fetches news/events per window (defaults to GDELT, no API key; supports `rss` and `newsapi` too). Files land under `logs/news_events/` with per-window CSVs plus a summary.
- News caveats: GDELT coverage effectively starts ~2015-06; days beyond the current date are skipped; 404 responses are treated as “no events” and logged once. Windows are capped by trigger count/day to avoid excessive requests.
- News severity: windows are ranked by triggers × summed p_bad so the noisiest/stressful periods are fetched first; summaries include `severity_sum_p_bad` for quick sorting.
- Severity + synthetic labels: `trading/scripts/score_bad_windows.py` computes synthetic bad flags (large returns vs σ or drawdown slope) and ranks top-N windows by summed `p_bad` over a sliding window. Use it to validate p_bad/bad_flag without external news.
- Regime windows as hazard: treat bad onsets as window boundaries; analyze window lengths, PnL per window, and whether damage clusters near window starts. Tri-state p_bad semantics: {+1 coherent, 0 undecidable, −1 incoherent}; policy = ACT if p_bad>θ₊, BAN if p_bad<θ₋, HOLD otherwise. “Legal” BAN monetisation is structure-based (vol/fee avoidance/flat), not forced shorting.
- `trading/scripts/sweep_tau_conf.py` conditions forward returns on engagement and bad_flag: `ret_engaged`, `ret_flat`, `ret_bad`, `ret_good`, plus `edge_per_turnover`. Use `--live-plot` to see PR and PnL-vs-DD live with annotations.
- Orchestrator usage example: `PYTHONPATH=. python trading/run_all_two_pointO.py --markets --market-progress-every 500 --csv data/raw/stooq/btc_intraday_1s.csv --live-sweep --run-ca --ca-report-every 1000`.
- Intent: separate “world is a bad game” detection from directional signals; this is an audit/pre-gate signal, not a change in acceptance logic.
- Run summaries (`run_all_two_pointO.py --markets`) now print average `p_bad` and `bad_rate` per market so the clustering around ~100k PnL can be read alongside structural stress.
- Per-run summary in `data/run_history.csv` now includes `ret_all`, `ret_good`, and `ret_bad` (equity return conditioned on bad=0/1) to validate the classifier economically.
- Full concept write-up (what “bad day” means, how to compute/test it): `docs/bad_day.md`.
## Trading stack: what is implemented today
- **Triadic control loop (implemented):** `trading/run_trader.py` computes a triadic latent state (`compute_triadic_state`) and drives exposure in {-1,0,+1}. It uses: HOLD decay, velocity-based exits, persistence ramp, risk targeting (`SIGMA_TARGET`, `DEFAULT_RISK_FRAC`), impact (`IMPACT_COEFF`), and fees (`cost`). This is the same simulator used by `trading/run_all.py`.
- **Epistemic gating & posture separation (implemented):** Strategy vs execution is split (`trading/strategy/triadic_strategy.py` + `trading/bar_exec.py`). Prediction (state) is distinct from permission/posture; logs include `action`, `hold`, `acceptable`, `actionability` for downstream analysis.
- **27-state kernel / persistence logic (implemented):** The triadic state and hysteresis live in the strategy; HOLD is epistemic, not flatten. Persistence and velocity checks are in `trading/run_trader.py` and honored across markets.
- **Market discovery & replay (implemented):** `trading/run_all.py` discovers all `data/raw/stooq/*.csv`, runs the same bar-level sim per market, writes per-market logs (`logs/trading_log_<market>.csv`), and prints a scoreboard. `--live` streams the dashboard while each market runs.
- **Projection quality (observed):** High-resolution BTC (1s) and daily BTC are profitable; coarse intraday bars (`btc_intraday`) are lossy and can hide forced flows. Use richer projections when possible (e.g., `btc_intraday_1s.csv`, `btc_yf.csv`).
- **Data sources (implemented):** `trading/data_downloader.py` pulls BTC from Binance (1m extended window, 1s resampled trades), Yahoo (daily/intraday), CoinGecko, and Stooq. `trading.run_trader.find_btc_csv` prefers 1s, then 1m, then daily.

## Future work: CA “kernel of kernels” (not implemented yet)
- Concept: each asset is a cell hosting a triadic epistemic kernel (permission, state, fatigue) and exchanging triadic messages with neighbors. Neighborhood can be sector-based or learned (e.g., kNN on correlation embeddings).
- State proposal: visible sign `s∈{-1,0,+1}` plus phase/chirality `φ∈{-1,0,+1}` to allow glider-like motion; gate `g∈{-1,0,+1}` (ban/hold/act).
- Update sketch: gate first (M₄ corridor, M₇ fatigue, M₉ shutdown pressures), then state update with a phase-gradient term to create motion. Market data (returns/vol/spread binned to triads) enters as anchors or boundary forcing.
- Status: design only—no code yet. If we build it, we’ll pick a concrete lattice (e.g., kNN on correlation) and add a prototype CA runner alongside the existing bar simulator.

## How `trading/run_all.py` behaved before the latest changes
- Single-threaded batch runner: iterated cached market CSVs under `data/raw/stooq`, executed the same bar-level simulator as `trading/run_trader.py` (includes fees via `cost`, slippage via `IMPACT_COEFF`, HOLD decay, risk targeting), wrote a single `logs/trading_log.csv`, printed per-run summary, then a simple scoreboard.
- No live dashboard streaming; to visualize you had to run `trading/training_dashboard.py` separately pointing at the log.
- Example run (legacy behavior):
  ```
  ❯ python trading/run_all.py

  === Running market: aapl.us ===
  Run complete: source=aapl.us, steps=10403, trades=1693, pnl=99142.2711

  === Running market: btc.us ===
  Run complete: source=btc.us, steps=348, trades=38, pnl=99906.5616

  === Running market: btc_intraday ===
  Run complete: source=btc_intraday, steps=100799, trades=12453, pnl=-29813.2553

  === Running market: msft.us ===
  Run complete: source=msft.us, steps=10020, trades=1132, pnl=97964.1213

  === Running market: spy.us ===
  Run complete: source=spy.us, steps=5238, trades=507, pnl=95434.7109

  === Per-market results ===
  btc.us               steps=   348 trades=   38 pnl=99906.5616 max_dd= -225.0276 hold=89.1%
  aapl.us              steps= 10403 trades= 1693 pnl=99142.2711 max_dd=-1754.1843 hold=83.7%
  msft.us              steps= 10020 trades= 1132 pnl=97964.1213 max_dd=-2604.6199 hold=88.7%
  spy.us               steps=  5238 trades=  507 pnl=95434.7109 max_dd=-4905.0864 hold=90.3%
  btc_intraday         steps=100799 trades=12453 pnl=-29813.2553 max_dd=-361939.6821 hold=87.6%

  === Overall ===
  markets=5, winners=4, losers=1, total_pnl=362634.4097
  ```

```
balanced_pn_iter_bench: 2-bit (P,N) balanced add with carry, iterative loop
Iterative balanced PN add: N=1024, iters=128,  3187.96 µs/word,     2.57 Mtrits/s
Iterative balanced PN add: N=1024, iters=512, 13531.47 µs/word,     2.42 Mtrits/s
[NB] Iterative balanced PN add: N=1024, iters=128,     1.47 µs/word,  5577.39 Mtrits/s
[NB] Iterative balanced PN add: N=1024, iters=512,     5.84 µs/word,  5608.63 Mtrits/s
dashitest.old.keepme: harness-backed C_XOR benchmark
Correctness OK.
N=100000:  1359.57 µs/call     73.55 Mwords/s
dashitest.py: consumer benchmark
Implementation: C_XOR_array_swar from swar_test_harness (UFT-C semantics, specials quieted, per-word flags)
Correctness smoke: OK (matched harness reference on 10k words with specials)

Benchmarking naïve baseline (C_XOR_naive), full semantics.
Precomputed (stored) timings; run manually if you need to refresh:
Gray-Scott KRR (GIF export) can be long on CPU; a recent run took ~25m:
`PYTHONPATH=./dashifine python dashifine/newtest/grayscott_krr.py --rollout_gif_steps 20 --rollout_gif_stride 1 --rollout_gif_fps 8`
N=     1000: 88474.91 µs/call       0.01 Mwords/s  (stored)
N=   100000: 8802841.42 µs/call       0.01 Mwords/s  (stored)

Benchmarking bitplane baseline (C_XOR_bitplane), normal lanes only (p_special=0).
N=     1000:   273.27 µs/call       3.66 Mwords/s
N=   100000: 31043.63 µs/call       3.22 Mwords/s

Benchmarking harness kernel (C_XOR_array_swar), specials enabled; no specials in inputs for throughput.
N=     1000:    29.35 µs/call      34.07 Mwords/s
N=   100000:   752.25 µs/call     132.93 Mwords/s
N=  5000000: 31806.73 µs/call     157.20 Mwords/s

Benchmarking dot product: reference vs SWAR (normal lanes only, p_special=0).
N=     1000: ref   101.72 µs/call   SWAR    16.65 µs/call   speedup x  6.1
N=   100000: ref 11399.21 µs/call   SWAR   528.86 µs/call   speedup x 21.6

Benchmarking threshold > 10: reference vs SWAR (normal lanes only, p_special=0).
N=     1000: ref   122.50 µs/call   SWAR    14.37 µs/call   speedup x  8.5
N=   100000: ref  7746.02 µs/call   SWAR   124.73 µs/call   speedup x 62.1
fused_iter_bench: XOR -> threshold -> dot loop (cache-resident)
Fused iter bench: N=1024, iters=256,    46.56 µs/iter,   791.73 Mop/s
Fused iter bench: N=1024, iters=1024,    53.95 µs/iter,   683.32 Mop/s
Sparse iterative classifier loop (XOR -> threshold -> dot) on cache-resident data.
K= 128: baseline    75.79 ms/epoch   SWAR    12.20 ms/epoch   speedup x 6.21
K= 512: baseline   321.29 ms/epoch   SWAR    77.48 ms/epoch   speedup x 4.15
Compiling Numba kernels (first call)...
OK: no_specials_small (N=10000, p_special=0.0)
OK: rare_specials (N=200000, p_special=0.0001)
OK: some_specials (N=200000, p_special=0.01)
OK: many_specials (N=50000, p_special=0.2)
BENCH (SWAR candidate): N=     1000     19.26 µs/call     51.93 Mwords/s
BENCH (SWAR candidate): N=   100000   1025.50 µs/call     97.51 Mwords/s
BENCH (SWAR candidate): N=  5000000  27135.95 µs/call    184.26 Mwords/s
All tests passed.
Triadic NN bench: baseline (unpacked int8) vs SWAR packed dot_product
Neurons: 8, input lanes: 12, values in {0,1,2}
N=     1000: baseline    74.55 µs/call ( 1287.71 Mop/s)  SWAR   252.16 µs/call (  380.71 Mop/s)  speedup x  0.3
N=   100000: baseline  5623.10 µs/call ( 1707.24 Mop/s)  SWAR  7810.94 µs/call ( 1229.05 Mop/s)  speedup x  0.7
triadic_nn_bench2: baseline NumPy vs packed SWAR dot_product_swar
N=     1000, M=  8: baseline    79.37 µs/call ( 1209.54 Mop/s) SWAR   202.35 µs/call (  474.42 Mop/s) speedup x 0.39
N=   100000, M= 16: baseline 13097.40 µs/call ( 1465.94 Mop/s) SWAR 17307.89 µs/call ( 1109.32 Mop/s) speedup x 0.76

epoch 997: loss=  1.9975  time= 26.89 ms
epoch 998: loss=  1.9957  time= 29.09 ms
epoch 999: loss=  1.9940  time= 25.70 ms
epoch 1000: loss=  1.9922  time= 26.41 ms
MoE-style sparse ternary benchmark (gate + route + dot) on CPU.
N=4096, M=8, iters=128: baseline   581.39 ms/epoch   SWAR     8.70 ms/epoch   speedup x66.83
Motif CA (M4/M7/M9) rule learning via count-based log-reg
Train acc: 79.64%  Test acc: 79.94%  time=8932.4 ms
Engagement (pred==1 vs acceptable): Train precision=0.793 recall=0.802 | Test precision=0.796 recall=0.803
[train] logreg trained in 34015.6 ms | N=819200 D=10
tau_off=0.45  k_off=1  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.40  k_off=1  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.35  k_off=1  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.30  k_off=1  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.25  k_off=1  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.20  k_off=1  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.15  k_off=1  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.45  k_off=2  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.40  k_off=2  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.35  k_off=2  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.30  k_off=2  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.25  k_off=2  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.20  k_off=2  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.15  k_off=2  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.45  k_off=3  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.40  k_off=3  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.35  k_off=3  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.30  k_off=3  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.25  k_off=3  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.20  k_off=3  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
tau_off=0.15  k_off=3  acceptable=0.012  precision=1.000  recall=0.000  act_cells=0  hold%=1.000
Wrote sweep metrics to logs/ca_pr_curve.csv
k  b_min  spares        efficiency
 1     2        1     79.25%
 2     4        7     79.25%
 3     5        5     95.10%
 4     7       47     90.57%
 5     8       13     99.06%
 6    10      295     95.10%
 7    12     1909     92.46%
 8    13     1631     97.54%
 9    15    13085     95.10%
10    16     6487     99.06%
11    18    84997     96.86%
12    20   517135     95.10%
13    21   502829     98.12%
14    23  3605639     96.48%
15    24  2428309     99.06%
16    26 24062143     97.54%
17    27  5077565     99.79%
18    29 149450423     98.38%
19    31 985222181     97.14%
20    32 808182895     99.06%
21    34 6719515981     97.89%
22    35 2978678759     99.63%
23    37 43295774645     98.52%
24    39 267326277407     97.54%
25    40 252223018333     99.06%
26    42 1856180682775     98.12%
27    43 1170495537221     99.52%
28    45 12307579633871     98.62%
29    46 1738366812781     99.92%
30    48 75583844616007     99.06%
31    50 508226510558677     98.27%
32    51 398779624833407     99.45%
33    53 3448138688185469     98.69%
34    54 1337216809815415     99.79%
35    56 22026048938928229     99.06%
36    58 138135740854712623     98.38%
37    59 126176846412426125     99.40%
38    61 954991291540701863     98.74%
39    62 559130865408411637     99.70%
40    64 6289078614652622815     99.06%

Byte-aligned (b multiple of 8):
 5     8       13     99.06%
10    16     6487     99.06%
15    24  2428309     99.06%
20    32 808182895     99.06%
25    40 252223018333     99.06%
30    48 75583844616007     99.06%
35    56 22026048938928229     99.06%
40    64 6289078614652622815     99.06%

Per-byte optimal packing (5 trits/byte):
5 trits in 8 bits: spares=13, efficiency=99.06% (~95% entropy efficiency)
Packing ablation: N=4096, iters=64, threshold=1
Unpacked:     15.85 ms/call (  595.38 Mop/s)
Radix (pack/unpack):    50.55 ms/call (  186.68 Mop/s)
Packed SWAR:    34.50 ms/call (  273.57 Mop/s)
Speedup SWAR vs unpacked: x 0.46, vs radix: x 1.47
Potts/3-state 1D lattice update (center+left+right mod3):
N=4096, iters=256: baseline     0.72 ms/iter   SWAR     0.15 ms/iter   speedup x 4.84
/home/c/Documents/code/dashitest/trading/training_dashboard.py:113: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  grouped = y.groupby(cat).mean()
/home/c/Documents/code/dashitest/trading/training_dashboard.py:113: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  grouped = y.groupby(cat).mean()
Using BTC data: data/raw/stooq/btc_intraday.csv
^CUsing BTC data: data/raw/stooq/btc_intraday.csv
Run complete: source=btc, steps=7698, trades=972, pnl=228710.5639
       t   ts   symbol  acceptable  state  intent_direction  intent_target  urgency  actionability  fill  fill_price  fee  pnl  exposure  slippage      price  action  hold  z_vel
995  995  995  BTCUSDT       False      1                 1       0.000000      0.3            1.0   0.0   95.675962  0.0  0.0       0.0       0.0  95.675962       1     0    0.0
996  996  996  BTCUSDT       False      1                 1       0.002439      0.4            1.0   0.0   95.730597  0.0  0.0       0.0       0.0  95.730597       1     0    0.0
997  997  997  BTCUSDT       False     -1                -1       0.000000      0.3            1.0   0.0   95.656653  0.0  0.0       0.0       0.0  95.656653      -1     0    0.0
998  998  998  BTCUSDT       False      1                 1       0.000000      0.3            1.0   0.0   95.721089  0.0  0.0       0.0       0.0  95.721089       1     0    0.0
999  999  999  BTCUSDT       False     -1                -1       0.000000      0.3            1.0   0.0   95.584756  0.0  0.0       0.0       0.0  95.584756      -1     0    0.0
Snapshot benchmark (hot P/N compute, optional 5-trit/byte snapshots):
N=8192, iters=128, snap_every=16
Hot only:    23.15 ms/call ( 1630.89 Mop/s)
Hot+snap:   166.44 ms/call (  226.80 Mop/s), snapshots=128.00 KiB
Sparse iterative classifier loop (XOR -> threshold -> dot) on cache-resident data.
K= 128: baseline    75.14 ms/epoch   SWAR    13.72 ms/epoch   speedup x 5.48
K= 512: baseline   310.74 ms/epoch   SWAR    48.99 ms/epoch   speedup x 6.34
Ternary SVO traversal (8-ary, depth=4, random states 0/1/2):
Nodes=585: baseline   662.51 µs/traversal   SWAR  1788.96 µs/traversal   speedup x 0.37
Numba SWAR:     0.99 µs/traversal   speedup vs baseline x670.33
Compiling Numba kernels (first call)...
OK: no_specials_small (N=10000, p_special=0.0)
OK: rare_specials (N=200000, p_special=0.0001)
OK: some_specials (N=200000, p_special=0.01)
OK: many_specials (N=50000, p_special=0.2)
BENCH (SWAR candidate): N=     1000     17.95 µs/call     55.70 Mwords/s
BENCH (SWAR candidate): N=   100000    817.64 µs/call    122.30 Mwords/s
BENCH (SWAR candidate): N=  5000000  25563.97 µs/call    195.59 Mwords/s
All tests passed.
Ternary ALU microkernel (iterative XOR with specials):
N=4096, iters=128: SWAR     0.05 ms/iter   Emulator   390.88 ms/iter   speedup x7163.9
Cyclic ternary CA (0→1→2→0, no HOLD)
Grid: 64x64, Steps: 30, k=2, wrap=True
t=00: 0 34.30%  1 32.25%  2 33.45%
t=01: 0 35.94%  1 31.98%  2 32.08%
t=02: 0 35.35%  1 31.84%  2 32.81%
t=03: 0 34.69%  1 32.35%  2 32.96%
t=04: 0 33.62%  1 32.28%  2 34.11%
t=05: 0 33.54%  1 32.35%  2 34.11%
t=06: 0 33.91%  1 32.01%  2 34.08%
t=07: 0 34.06%  1 32.74%  2 33.20%
t=08: 0 33.23%  1 33.57%  2 33.20%
t=09: 0 33.13%  1 33.20%  2 33.67%
t=10: 0 33.30%  1 33.54%  2 33.15%
t=11: 0 32.89%  1 33.64%  2 33.47%
t=12: 0 33.40%  1 32.93%  2 33.67%
t=13: 0 33.67%  1 33.40%  2 32.93%
t=14: 0 32.93%  1 33.67%  2 33.40%
t=15: 0 33.40%  1 32.93%  2 33.67%
t=16: 0 33.67%  1 33.40%  2 32.93%
t=17: 0 32.93%  1 33.67%  2 33.40%
t=18: 0 33.40%  1 32.93%  2 33.67%
t=19: 0 33.67%  1 33.40%  2 32.93%
t=20: 0 32.93%  1 33.67%  2 33.40%
t=21: 0 33.40%  1 32.93%  2 33.67%
t=22: 0 33.67%  1 33.40%  2 32.93%
t=23: 0 32.93%  1 33.67%  2 33.40%
t=24: 0 33.40%  1 32.93%  2 33.67%
t=25: 0 33.67%  1 33.40%  2 32.93%
t=26: 0 32.93%  1 33.67%  2 33.40%
t=27: 0 33.40%  1 32.93%  2 33.67%
t=28: 0 33.67%  1 33.40%  2 32.93%
t=29: 0 32.93%  1 33.67%  2 33.40%
kf.kio.filewidgets.kfilefiltercombo: KFileFilterCombo::setCurrentFilter: Could not find file filter KFileFilter(MIME patterns: QList() File patterns: QList("*.png") label: "Portable Network Graphics")
Ternary trading demo (no API keys, local/synthetic data)
Close series length: 1000
Trades executed    : 677
Ternary cum return : 760.06%   Sharpe: 58.87   Hit rate: 62.4%
Baseline (binary)  : 881.08%   Sharpe: 79.26   Hit rate: 99.6%
Runtime            : 19.72 ms

To use real data, provide prices.csv with columns: date, close.
Using BTC data: data/raw/stooq/btc_intraday.csv
Run complete: source=btc, steps=10, trades=8, pnl=99862.1562
PASS: run_trader used cached real data and produced a non-empty log.
Tiled dot product (same blocking, different algebra)
M=N=K=256, block=32
int8 dot :  1109.67 ms       0.03 Gops/s     0.00 GB/s   ops/byte=85.33
GF3 dot  :  1552.04 ms       0.02 Gops/s     0.00 GB/s   ops/byte=170.67   speedup x 0.71


```
Absolutely. Here is a **clean, comprehensive, end-to-end summary** of the entire conversation, without digressions, and in the language you’ve converged on.

---

# 0. The core problem you are solving

You are **not** trying to:

* predict prices
* optimize a neural network
* maximize Sharpe directly
* or build a “safe” trading bot

You are trying to design an **epistemic control system** for trading in **casino capitalism**:

> A system that knows **when the market is a game**, **who is forced to play**, and **how to act without becoming the bag holder**.

This is fundamentally a **dialectical / triadic control problem**, not a binary ML problem.

---

# 1. Triadic foundations (what everything is built from)

Everything is ternary:

[
{-1,;0,;+1}
]

This applies to:

* logic
* permission
* tension
* market posture
* risk
* action

Binary logic collapses contradictions.
Triadic logic **holds them**.

---

# 2. The base epistemic unit: the 27-state kernel

Each kernel evaluates the world through **three lenses**:

* **Self** – internal signal / belief
* **Mirror** – market response / confirmation
* **Norm** – structure, rules, cost, feasibility

Each lens is triadic →
[
(S,M,N)\in{-1,0,+1}^3 \Rightarrow 27 \text{ states}
]

These 27 states fall into:

1. **Coherent** (all equal) → act, suspend, or retire cleanly
2. **Majority + dissent** → caution / dialectical tension
3. **Fully mixed (-,0,+)** → true suspension (M₅), not prohibition

Mistake to avoid:

> **Collapsing mixed tension into “bad” (M₉)**
> That is how systems self-sabotage.

---

# 3. M₆, M₉, and why collapse is catastrophic

* **M₃** = a stance
* **M₆** = two stances held in unresolved tension
* **M₉** = closure of the space itself

Correct hierarchy:

* M₆ **must not** auto-escalate to M₉
* M₆ is the *productive* zone where insight forms
* M₉ is a **veto / closure operator**, used sparingly

In trading terms:

* M₆ = “market is contradictory, but informative”
* M₉ = “this space is untradeable or fatal”

---

# 4. Why encoding debates happened (and why they don’t matter now)

You explored:

* 27 states
* 14 equivalence classes under sign inversion
* 4-bit and 5-bit encodings
* hyper-sheets like (3^9), (9^{(9^9)})

Conclusion:

* Clever packing is possible
* Compression is near-optimal
* **Encoding does not create profit**

That exploration was valuable because it proved:

> The system is structurally sound — but **representation is not the bottleneck**.

So you correctly pulled back.

---

# 5. The real architecture: the joint ABC machine

You converged on **three distinct kernels**, all ternary, all epistemic, all supervisory.

---

## Kernel A — Local Regime Coherence (micro/meso)

**Question:**

> *Is the tape legible enough to express a position without churn?*

Inputs:

* flip rate
* micro-vol vs range
* spread stability
* order-flow persistence

Output (A):

* **+1** coherent / expressible
* **0** mixed / noisy
* **−1** chaotic / execution trap

This kernel is *fast* and *operational*.

---

## Kernel B — Bag-Holder / Forced-Actor Detection (agentic)

**Question:**

> *Who is structurally forced to act or bleed?*

Inputs:

* negative carry (funding, borrow, roll)
* liquidation cascades
* one-sided aggressive flow
* margin / inventory stress

Output (B):

* **+1** others are the bag holders (we have optionality)
* **0** no forced cohort
* **−1** we would be the bag holder

This kernel is **existential**.
If it says −1, the system **must not trade**.

---

## Kernel C — Structural Health / Pathology (macro)

**Question:**

> *Is the market healthy, distorted, or broken?*

Inputs:

* basis dislocations
* carry distortions
* leverage stress
* vol-of-vol
* reflexive policy effects

Output (C):

* **+1** healthy / equilibrating
* **0** stressed / distorted
* **−1** pathological / casino

This kernel is **slow, supervisory, M₉-like**.

---

# 6. The joint ABC decision machine (final)

The system outputs **posture**, not price.

Postures:

* **UNWIND** – reduce exposure
* **OBSERVE** – do nothing, gather info
* **TRADE_NORMAL** – ordinary strategies
* **TRADE_CONVEX** – asymmetric, patient, crisis trades

### Decision logic (non-negotiable)

1. **If B = −1 → UNWIND**
   (Never be the bag holder.)

2. **If B = 0 → OBSERVE**
   (No asymmetry, no edge.)

3. **If B = +1 (others forced):**

   * If C = +1 and A = +1 → TRADE_NORMAL
   * If C = 0 and A = +1 → TRADE_CONVEX-lite
   * If C = −1 and A = +1 → **TRADE_CONVEX** ✅
   * Else → OBSERVE

---

# 7. The “Burry zone” (now precise)

[
(A,B,C) = (+1,\ +1,\ -1)
]

Meaning:

* The market is **structurally broken**
* **Someone else must transact or bleed**
* The tape is legible enough to express a position

This is **not** a safe market.
This is where **normal traders die** and **structural traders win**.

Trading here requires:

* patience
* convexity
* low turnover
* survival of being early

---

# 8. Why your system felt “too safe”

Because initially you had:

* regime detection
* no explicit bag-holder logic
* no inversion for rotten markets

That produces:

* avoidance instead of exploitation
* liberal / amateur behavior
* low turnover but low profit

Adding **Kernel B** fixes this.

---

# 9. Key philosophical resolution

You correctly rejected:

* “kernel prevents bleeding” (deficiency framing)
* “kernel predicts prices” (naïve framing)

The correct synthesis is:

> **Kernel predicts structural advantage — and refuses to act when advantage implies becoming the bag holder.**

That is neither fearful nor naïve.
It is **adversarial realism**.

---

# 10. Final one-sentence summary

> **The system exists to decide when the market is broken, who is forced by that breakage, and whether we can act without becoming the forced party — profit only emerges from that asymmetry.**

If you want, the *next* concrete step would be:

* defining actual feature proxies for **B (bag holder)** from your data,
* or expressing **TRADE_CONVEX** without options,
* or wiring this ABC logic directly into your existing gate.

Just tell me where you want to go next.

# dashitest — Compactified Context

## Scope
- This file is a compact, durable snapshot for the dashitest repo (root).
- It summarizes current intent, implemented state, and the highest-value next steps.

## Intent (current)
- Keep trading stack epistemic gating PnL-free; evaluation = precision/recall on acceptable vs ACT.
- Keep CA/benchmark work as research-lab outputs, not trading inputs.
- Maintain reproducibility: timestamped outputs and documented run artifacts.

## Implemented (high-signal)
- Trading stack: `state → TriadicStrategy → Intent → Execution → Log → Dashboard`.
- Hysteresis gating (`tau_on > tau_off`) + `RegimeSpec` acceptable gate (PnL-free).
- `run_all.py` runs cached markets + optional live dashboard; logs `p_bad`/`bad_flag` for structural stress.
- Dashifine / tree diffusion / compression benchmarks have docs and scripts with timestamped outputs.
- Phase-3 quotient training in `dashilearn/bsmoe_train.py` logs JSON + plots per run.

## Key Docs
- `README.md` (project map + doc index)
- `docs/tree_diffusion_benchmark.md`
- `docs/phase3_quotient_learning.md`
- `docs/b2_acceptance.md`

## Recent Chat Sync (canonical archive)
- Trading diagnostics: ES/NQ proposals are flat; monitor logic is correct; next sprint = amplitude diagnostics.
- Formalizing kernel: capital kernel + Meta-Witness refusal rules; Phase-9 wiring before actions.
- dashiCORE: create Function Coverage Map + benchmarking harness with efficiency surfaces.

## Futures Shadow Calibration (current)
- Shadow kernel now supports score-mode/gating-mode A/B, shrinkage blending, curvature diagnostics, and kernel log dir overrides.
- Kernel training moved to `logs/shadow` to ensure `price_ret` is present; training labels now tracked (long/short/flat/stress).
- Calibration sweeps show two failure poles:
  - Low thresholds (0.01-0.03): all-act, flat mass ~0, basin margin ~1.
  - Higher thresholds (0.05-0.25): training flat labels appear (up to ~5%), but predicted flat mass remains ~0.
- Label-stratified beam retention + flat return-band classification (with fee floor) were added; beam survival counts show flat survives all beam depths.
- Label-aware basin aggregation + fee-floor rerun at 0.05 (`shadow_signal_report_20260313T135734Z_costband005_rerun.md`) now shows `pred_flat` lift-off on both tapes (BTC ~0.0865, SPY ~0.0737), so the basin is no longer structurally binary.
- 0.10 companion rerun (`shadow_signal_report_20260313T143040Z_costband010_rerun.md`) confirms a steep phase boundary: `pred_flat` rises (~0.175) but action rate collapses (BTC 0.0, SPY 0.0036) under entropy gating.
- Gate calibration landed: default lex entropy threshold raised to 0.96; rerun (`shadow_signal_report_20260313T145412Z_costband010_ent096.md`) now produces non-degenerate action rates (BTC ~0.055, SPY ~0.104) while keeping tri-modal basin geometry.
- Matched 0.05 rerun under `ent096` (`shadow_signal_report_20260313T151102Z_costband005_ent096.md`) remains near-all-act (BTC 1.0, SPY 0.9992), so the 0.05 -> 0.10 action-rate slope is still steep.
- Smooth logistic entropy attenuation is now implemented and configurable (`mode/center/tau`) and a 3-point sweep was run (0.05/0.075/0.10) with `H0=0.955`, `tau=0.01`.
- Result: geometry moves smoothly but action rates are still near-all-act across the sweep under current score thresholds.
- Next step is score-threshold/adaptive-action-rate calibration on top of the smooth entropy multiplier.

## Next Steps (short list)
- Implement proposal amplitude diagnostics (size/dir/score quantiles + correlations).
- Decide whether to wire Phase-9 capital kernel + Meta-Witness into stream daemon.
- Extend function coverage map + benchmark harness for dashiCORE.

## Assumptions
- Python 3.11+, NumPy + PyTest are available.
- No GPU dependency required for core correctness; Vulkan/JAX are reference/optional.

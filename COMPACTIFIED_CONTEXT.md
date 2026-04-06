# dashitest — Compactified Context

## Scope
- This file is a compact, durable snapshot for the dashitest repo (root).
- It summarizes current intent, implemented state, and the highest-value next steps.

## Intent (current)
- Keep trading stack epistemic gating PnL-free; evaluation = precision/recall on acceptable vs ACT.
- Keep CA/benchmark work as research-lab outputs, not trading inputs.
- Maintain reproducibility: timestamped outputs and documented run artifacts.
- Keep the mirrored `dashifine` quantum utilities framed as classical,
  quantum-faithful simulations; the bridge/internalization formalism now belongs
  in sibling repo `dashiQ`, not in the root `dashitest` benchmark backlog.
- This wording is now also backed by the local archive threads
  `P-adic quantum systems` and `Quarter turn in quantum`, which sharpen the
  distinction between formal/simulator work and actual hardware claims.

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
- Per-asset adaptive quantile thresholding is now implemented and tested (target action rate 10% over rolling score history).
- Result: SPY action rates are near target (~10–11%), BTC remains high (~18–29%) on short tapes due cold-start/history effects.
- Added offline/prefit quantile initialization hooks (seed adaptive score history from prior `trading_log*.csv` `shadow_score_adjusted` per asset), plus strict run-family seed scoping (`--shadow-score-threshold-prefit-family`) to avoid cross-regime contamination during adaptive cold-start initialization.
- Prefit and family-scoped reruns changed activation materially, but did not reliably improve the economic selection test; the current blocker is now failure-locus diagnosis across proposal amplitude, ranking quality, and activation quality.
- SPY is the main calibration anchor for the next branch; BTC remains secondary validation / negative-control until short-tape instability is better contained.
- Family-scoped prefit comparison (`logs/shadow/shadow_signal_report_20260315T081325Z_prefit_family_compare.md`) showed effectively no material change vs `seed200` unscoped prefit, confirming prefit scope is no longer the highest-value lever.
- Failure-locus smoke diagnostics now exist in `scripts/analyze_shadow_signals.py` and produce raw-score spread, ranking curve, activation curve, and score-vs-return heatmap artifacts (e.g. `logs/shadow/shadow_signal_report_20260315T082856Z_failure_locus_smoke.md`).
- Implemented shadow-only per-asset score standardization with pooled shrinkage (`per_asset_zscore_shrunk`) and ran SPY diagnostics (`logs/shadow/shadow_signal_report_20260315T094904Z_spy_scorecal_long.md`): score scaling stabilized mechanically, but the economic selection test still failed (`E(|ret| | ACT) <= E(|ret| | HOLD)`), so calibration alone is not the blocker.
- Implemented an uncertainty-penalty ablation (`explicit` vs `merged_uncertainty`) and ran SPY partial A/B (`logs/shadow/shadow_signal_report_20260315T120127Z_spy_uncertainty_ab_partial.md`): merged uncertainty improves tail-activation alignment (top bucket starts activating), but ACT vs HOLD separation is still not reliably positive.
- Verified entropy attenuation is not the primary blocker by A/B with entropy gate disabled (`logs/shadow/shadow_signal_report_20260315T121652Z_spy_uncertainty_ab_entoff.md`): selection quality remains weak.
- Ran an action-functional return-mode ablation (`directional` vs `abs`) under merged uncertainty (`logs/shadow/shadow_signal_report_20260315T121923Z_spy_absmerge.md`): treating the return term as magnitude does not yet flip ACT vs HOLD on its own.
- Current best diagnosis on SPY: nonzero ranking signal exists but is weak; activation alignment depends on penalty geometry; the next branch is score-structure tuning (penalty-block simplification and/or two-stage policy: gate on opportunity magnitude, then pick direction separately).

## Next Steps (short list)
- Use the failure-locus plots (amplitude/ranking/activation) to decide whether to (a) simplify penalty geometry further or (b) implement a two-stage policy (opportunity magnitude gate + separate direction choice). Keep SPY as primary anchor and BTC as secondary.
- Decide whether to wire Phase-9 capital kernel + Meta-Witness into stream daemon.
- Extend function coverage map + benchmark harness for dashiCORE.
- Keep the root docs aligned with the current quantum scope split:
  - `dashitest/dashifine` mirrors classical quantum-faithful experiments
  - `dashiQ` owns the bridge/simulator formalism

## Assumptions
- Python 3.11+, NumPy + PyTest are available.
- No GPU dependency required for core correctness; Vulkan/JAX are reference/optional.

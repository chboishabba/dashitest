# Quant Professional Review Packet

This packet is a short, review-focused technical brief for quant colleagues. It
summarizes the current math framing, the shadow-policy state, the empirical
artifacts, and the exact judgement calls needed to tune the system.

It is not a full system spec.

## Reading order (fastest path)

1. `docs/discrete_action_functional.md`
2. `COMPACTIFIED_CONTEXT.md`
3. `docs/appendix_kernel_vulkan.md`
4. `docs/phase7_status_emitter.md`
5. `docs/phase9_legitimacy.md`

## System summary (math structure)

The shadow policy is a finite-horizon decision functional:

```
coarse state → tree expansion → basin mass → action score → hold/act
```

Core elements:

- **Coarse state**: drift, triadic bias, actionability, stress, edge,
  contraction, diffusion, drawdown, current exposure.
- **Tree expansion**: short horizon discrete action grid {-1, 0, +1}.
- **Action functional**: reward block minus penalty block (explicit terms).
- **Basin aggregation**: long / short / flat terminal mass.
- **Hold/act gating**: entropy, basin separation, flat mass, and score.

This layer is shadow-only. It does not control execution.

## Action functional (current form)

```
score =
  + signed return
  + alignment
  + actionability
  + contraction gain
  - branch risk
  - diffusion risk
  - stress
  - drawdown
  - churn
  - inventory incoherence
```

The score is the current main tuning bottleneck. It has already produced two
degenerate regimes:

- all-hold (score too negative)
- near-all-act (score too positive)

## Learned transition kernels (current)

Shadow mode supports three learned-kernel modes:

- `global`: trained on all eligible per-step trader logs
- `per_asset`: trained on symbol-matched logs, fallback to global
- `residual`: simple blended buckets from global and asset-local models

The modes are now directly comparable via the standard shadow analysis script.

## Basin geometry extensions (current)

Shadow diagnostics now include basin geometry signals:

- `beam_curvature = p_long + p_short - p_flat`
- `beam_flat_distance = ||(p_long, p_short, p_flat) - (0,0,1)||`

Shadow gating can optionally:

- hold when curvature is below a threshold
- weight scores by curvature to suppress noisy regimes
- classify flat basin mass using a return band (small returns → flat)
- enforce a fee/slippage cost floor for flat classification

These are logged but not yet used for control takeover.

## Empirical state (latest artifacts)

### Hold-remediation comparison (pre-weight retune)

Artifacts:

- `logs/shadow/*_20260313T061237Z.*`
- `logs/shadow/shadow_signal_report_20260313T061237Z.md`

Result:

- hold attribution is explicit
- kernel modes are comparable
- all modes still held (shadow_hold = 1.0)
- dominant cause: score pressure (not entropy)

### First ActionWeights retune (overshoot)

Artifacts:

- `logs/shadow/*_20260313T062106Z.*`
- `logs/shadow/shadow_signal_report_20260313T062106Z.md`

Result:

- all-hold regime broken
- near-all-act overshoot
- BTC `global` / `per_asset` act ratio ≈ 0.996
- BTC `residual` act ratio = 1.0
- all SPY modes act ratio = 1.0

Current blocker: moderation/calibration.

### Threshold sweeps (label calibration)

Artifacts:

- `logs/shadow/shadow_signal_report_20260313T115844Z.md` (fixed vs vol label modes at 0.01)
- `logs/shadow/shadow_signal_report_20260313T121008Z_th0010.md` to `..._th0030.md` (fixed 0.010–0.030)
- `logs/shadow/shadow_signal_report_20260313T123649Z_th005.md` to `..._th025.md` (fixed 0.05–0.25)
- `logs/shadow/shadow_signal_report_20260313T125828Z_lsr_th005.md` (label-stratified retention at 0.05)
- `logs/shadow/shadow_signal_report_20260313T130029Z_lsr_th010.md` (label-stratified retention at 0.10)
- `logs/shadow/shadow_signal_report_20260313T135734Z_costband005_rerun.md` (label-aware + fee-floor rerun at 0.05 after parser-limit fix)

Results:

- Low thresholds (0.010–0.030): all-act, flat mass ~0, basin margin ~1.
- Higher thresholds (0.05–0.25): training flat labels appear (up to ~5%) but predicted flat mass stays ~0.
- Label-stratified retention (0.05/0.10) initially yielded `pred_flat ≈ 0`; beam survival counts confirmed flat survives all beam depths and was collapsing at aggregation.
- After label-aware flat persistence + fee-floor band rerun at 0.05, `pred_flat` lifted on both tapes (BTC `0.0865`, SPY `0.0737`), confirming tri-modal basin recovery.

Current next step: run the 0.10 companion check under the same parser-fixed path
to map the hold/act transition, then retune gate/score thresholds from a now
valid tri-modal basin.

## What we need from quant reviewers

### A. Action-functional calibration

1. Are the reward and penalty terms structurally well separated?
2. Should branch risk and diffusion be combined or kept distinct?
3. Are churn and inventory penalties too strong relative to expected edge?
4. Should any penalties be nonlinear or regime-conditional?

### B. Hold/act gating

1. Is the current gating hierarchy appropriate?
2. Should flat mass reduce exposure, force hold, or define a separate regime?
3. Should score thresholds be absolute or scale with entropy/variance?

### C. Kernel family structure

1. Is `global` / `per_asset` / `residual` the right decomposition?
2. If residual stays, should the blend be changed to a shrinkage estimator?
3. If residual is removed, what should replace it?

### D. Diagnostics and acceptance criteria

What is the minimal decision-grade evidence that would justify:

- advisory veto mode (shadow can cancel, not replace)
- control takeover (shadow becomes primary)

## Concrete questions to answer

1. Which score terms are misweighted today?
2. What score normalization would you try first?
3. What action-rate range is acceptable for shadow policy in this context?
4. Should the score distribution be centered, bounded, or skewed by design?
5. What plots or statistics would make this tuning decision complete?

## One-line status summary

We now have an interpretable, shadow-only competing-futures controller with
learned transition kernels, explicit hold diagnostics, and basin-geometry
signals; calibration has escaped all-hold but now overshoots into all-act.
Label-threshold sweeps show flat labels exist in training, yet flat basin mass
initially remained ~0 even after label-stratified retention, indicating basin
aggregation collapse. With label-aware basin classification (cost-floor band)
plus beam survival instrumentation, the 0.05 rerun now shows nonzero
`pred_flat` on both BTC/SPY; immediate work is threshold calibration (starting
with 0.10 companion run) rather than structural flat-survival fixes.

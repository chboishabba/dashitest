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

Current blocker: failure-locus diagnosis rather than another generic
calibration pass.

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

Current next step: distinguish whether the remaining failure is primarily:

- proposal amplitude / score-spread collapse
- ranking failure
- activation/gating failure
- tape-specific mixtures of the above

SPY is now the primary calibration anchor. BTC remains secondary validation
until short-tape instability is better controlled.

Latest branch decision:

- The first failure-locus reports on SPY are not consistent with pure
  amplitude collapse: raw-score spread remains non-trivial and ranking uplift
  is positive.
- The next implementation branch is therefore optional raw-score
  standardization with pooled shrinkage, still shadow-only and still judged
  against `E(|ret| | ACT) > E(|ret| | HOLD)` before any uncertainty-block
  simplification.

## What we need from quant reviewers now

### 1. Failure locus

Which is the dominant failure mode now?

- A. upstream proposal amplitude / score-spread collapse
- B. bad score ranking
- C. over-harsh activation/gating
- D. different failure modes by tape

Recommended default:

`D`, tested in order `A -> B -> C`.

### 2. Calibration vs weight retune

What should change first?

- A. per-asset raw-score calibration
- B. reward/penalty retune
- C. both, in strict order

Recommended default:

`C`, with calibration before retuning.

### 3. Normalization target

What should be normalized?

- A. final gated score
- B. raw pre-gate score
- C. reward and penalty blocks separately
- D. log both B and C, use B as canonical ranking object

Recommended default:

`D`. Use raw pre-gate score as the main ranking object and log block-level
structure separately for diagnostics.

### 4. Cross-asset policy

How should normalization behave across tapes?

- A. per-asset only
- B. global only
- C. per-asset with pooled shrinkage for short tapes

Recommended default:

`C`.

### 5. Acceptance gate

Which metric should block rollout first?

- A. acceptable-vs-ACT precision/recall
- B. `E(|ret| | ACT) > E(|ret| | HOLD)`
- C. ACT Sharpe proxy
- D. ACT sign accuracy

Recommended default:

`A -> B`.

### 6. Penalty structure

What ablation should we run?

- A. keep explicit penalties, retune weights
- B. merge correlated uncertainty terms
- C. compare both

Recommended default:

`C`, with bias toward `B` if performance parity holds.

### 7. BTC status

How should BTC be treated in the next branch?

- A. primary calibration tape
- B. secondary validation tape
- C. explicit negative control
- D. separate controller family candidate

Recommended default:

`B`, with some operational use of `C`.

## One-line status summary

We now have an interpretable, shadow-only competing-futures controller with
learned transition kernels, explicit hold diagnostics, basin-geometry signals,
and functioning adaptive-threshold controls. Those controls can move action
rates materially, but recent prefit/family-scope reruns did not reliably
improve the main economic test. Immediate work is therefore not more generic
gate tuning; it is failure-locus diagnosis on score spread, ranking quality,
and activation quality, with SPY as the main calibration anchor and BTC kept as
secondary validation.

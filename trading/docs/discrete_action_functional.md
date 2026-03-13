# Discrete Action Functional for the Current Trader

## Purpose

Add an optional policy layer that scores short future paths instead of only the
next step. The archived thread `Reconsidering Trading Bot`
(`69b200ec-a590-83a1-8ee2-032a3712f038`,
canonical `18b31603c8832988de325375f56637a44310ebab`) called for a concrete
Python module spec that fits the current trader rather than replacing it.

This document records the repo-local version of that design.

## Constraints

- Keep the existing `Intent` contract as the execution boundary.
- Do not replace the baseline triadic controller by default.
- Reuse current observables where possible: triadic state, actionability,
  stress, edge/cost diagnostics, exposure, drawdown, and quotient-like
  contraction signals.
- Keep horizon short and discrete so the controller remains inspectable.

## Coarse state vector

Version 1 uses eight coarse variables:

1. `drift`: short-horizon signed directional tendency.
2. `triadic_bias`: normalized sign of the current triadic state.
3. `actionability`: legitimacy to act, in `[0, 1]`.
4. `stress`: structural stress / hazard pressure, in `[0, 1]`.
5. `edge`: signed net edge proxy after cost awareness.
6. `contraction`: confidence that the current regime is coherent rather than
   diffusing.
7. `diffusion`: branchiness / uncertainty pressure.
8. `drawdown`: normalized capital stress.

The runtime state also carries current signed exposure so path scoring can
charge churn and inventory risk.

## Transition model

The beam search uses a small branching model over discrete actions:

- `-1`: move short
- `0`: hold / flatten
- `+1`: move long

Each action expands into a few plausible next coarse states:

- trend continuation
- mean reversion
- stress event

Each branch has:

- `probability`
- `next_state`
- `step_return`
- `step_risk`
- `label`

This is intentionally simple. The first implementation is a transparent
heuristic model seeded from the current state; later versions can swap in
empirical estimators without changing the beam-search interface.

## Discrete action functional

For each transition, score:

```text
score =
  + w_return      * signed_return
  + w_alignment   * alignment_with_triadic_bias
  + w_action      * actionability * |exposure|
  + w_contract    * contraction_gain
  - w_diffusion   * diffusion
  - w_stress      * stress
  - w_drawdown    * drawdown
  - w_churn       * |delta_exposure|
  - w_inventory   * |exposure| when regime is incoherent
```

Interpretation:

- reward coherent exposure that aligns with the current trader state
- punish acting when diffusion, stress, or drawdown dominate
- punish unnecessary switching
- preserve a visible decomposition for audit and later logging

## Beam search

Configuration:

- horizon: `4`
- beam width: `12`
- exposure step: `0.25`
- minimum branch probability: `1e-3`

At each depth:

1. expand every live node across the action grid
2. generate transition branches
3. add step score and log-probability
4. prune by probability floor
5. keep the top `beam_width` nodes by cumulative score

The output is a set of terminal nodes plus aggregate masses:

- best score
- long mass
- short mass
- flat mass
- path entropy
- flat mass now uses a return band with a fee floor:
  `|predicted_return| < max(flat_return_band, flat_cost_floor)`

## Intent selection

`BeamIntentPolicy` converts terminal beam statistics back into the current
trader contract:

- choose the sign with the strongest terminal mass
- scale exposure by the best node for that sign
- set `hold=True` when flat mass dominates or the best score is non-positive
- store a compact audit string in `Intent.reason`

This keeps execution unchanged: executors still consume ordinary `Intent`
objects.

## Shipping posture

What is implemented now:

- coarse-state estimator
- discrete action functional
- heuristic transition model
- beam search
- optional beam-backed intent selector
- basin mass classifier
- `BeamSummary` diagnostics for entropy, masses, score, return, risk, contraction, and diffusion
- basin-geometry diagnostics (`beam_curvature`, `beam_flat_distance`)
- shadow runner contract for observe-only validation against the baseline policy
- opt-in `--shadow-futures` engine logging path that records live/shadow decisions in the per-step trader log without changing execution
- corrected pre-execution shadow splice so the shadow policy sees the same state the live policy used before fills mutate cash/position
- decision-grade BTC/SPY shadow report and plots captured at `logs/shadow/*_20260312T052900Z.*`
- learned transition-kernel scaffold fitted from historical per-step trader logs, with heuristic fallback when no usable log history exists
- score-mode and gating-mode A/B (`ratio`, `scaled_diff`, `logistic`; `lex`, `joint`, `score_only`)
- `shrinkage` kernel mode with kernel lambda metadata
- kernel log-dir override (`--shadow-kernel-log-dir`) so the learned model can train from `logs/shadow`
- label-stratified beam retention (quota + overflow) so flat candidates survive pruning
- label-aware basin classification (flat label persists; return band uses fee floor)
- beam label survival logging (per-depth long/short/flat counts)

What is intentionally not implemented yet:

- control takeover in the main loop
- dashboard panes for beam entropy / basin masses
- post-cost-band BTC/SPY recalibration once `pred_flat` lift-off is confirmed

## Decision-grade diagnosis

The corrected BTC/SPY rerun (`logs/shadow/*_20260312T052900Z.*`) shows that the
current futures policy is still an observability scaffold rather than a viable
controller:

- `beam_entropy` remains near-maximal on both BTC and SPY
- the shadow policy remains effectively all-hold
- `beam_flat_mass` is zero throughout the decision-grade baseline
- basin-edge correlation to the next move remains close to zero

The durable diagnosis is recorded in
`logs/shadow/shadow_signal_diagnosis_20260313T020345Z.md`.

That locks the next milestone:

- replace `HeuristicTransitionModel` with a learned transition kernel
- keep the existing coarse-state, action-functional, beam-summary, basin, and
  intent-policy interfaces stable
- rerun the same BTC/SPY shadow analysis after the learned kernel lands

## Learned kernel landing

The shadow path now builds its transition model from historical per-step trader
logs before falling back to the old heuristic generator. The learned model:

- reconstructs the same coarse-state representation from existing CSV logs
- buckets transitions by coarse-state regime and action delta
- emits learned `up` / `down` / `flat` / `stress` transition candidates
- preserves the current beam-search and `TransitionCandidate` contract

Initial short-run sanity after landing the learned kernel:

- `beam_flat_mass` is no longer structurally zero
- entropy dropped from the old near-max baseline in a short BTC shadow run
- the shadow policy is still hold-dominant, so the full BTC/SPY rerun remains
  necessary before any policy-threshold conclusions

## Post-kernel BTC/SPY rerun

The full post-kernel comparison is captured at `logs/shadow/*_20260313T045625Z.*`.

What improved versus the heuristic decision-grade baseline
(`20260312T052900Z`):

- BTC entropy mean dropped from `0.970205` to `0.901038`
- SPY entropy mean dropped from `0.956483` to `0.900277`
- flat-mass support returned:
  - BTC flat-mass nonzero ratio `0.328181`
  - SPY flat-mass nonzero ratio `0.183424`

What did not improve yet:

- `shadow_hold` remained `1.0` on both BTC and SPY
- live/shadow divergence remained very low
- basin-edge correlation to the next move remained weak

That changes the diagnosis:

- the branch generator is no longer the main blocker
- the next blocker is the policy layer above it:
  - entropy hold threshold is still too blunt
  - beam scores still skew negative often enough to keep intent selection flat
  - intent-selection logic needs recalibration now that flat basin mass exists

## Calibration sweeps (label thresholds)

Fixed-threshold sweeps across `0.010–0.030` and `0.05–0.25` show:

- low thresholds: all-act, flat mass ~0, basin margin ~1
- higher thresholds: flat labels appear in training (up to ~5%), but predicted
  flat mass stays ~0, so action rate remains near 100%

Conclusion: the kernel sees flat labels, but beam selection/score still
eliminates flat paths. Label-stratified retention and label-aware return-band
classification (with fee floor) are now implemented, but the 0.05/0.10
retention rechecks still show `pred_flat ≈ 0`, indicating basin aggregation is
still collapsing flat trajectories. The next step is a cost-band sweep with
beam survival counts to confirm `pred_flat` lift-off before gate retuning.

## Hold-bias remediation milestone

The next milestone is still shadow-only and has three goals at once:

1. make every `shadow_hold` decision explicitly attributable
2. reduce all-hold behavior now that learned beam geometry is better
3. compare `global`, `per_asset`, and `residual` learned-kernel modes on the
   same BTC/SPY shadow report

### Structured hold attribution

The futures policy should expose one primary hold cause plus supporting gate
values. Required primary values:

- `entropy`
- `score`
- `basin`
- `flat_basin`
- `empty`
- `act`

Required logged fields:

- `shadow_hold_reason_primary`
- `shadow_hold_entropy`
- `shadow_hold_score`
- `shadow_hold_basin`
- `shadow_hold_flat_basin`
- `shadow_entropy_value`
- `shadow_entropy_threshold`
- `shadow_score_value`
- `shadow_score_threshold`
- `shadow_directional_mass`
- `shadow_directional_margin`

When `shadow_hold=1`, exactly one primary reason should be present.

### Calibrated intent-selection rule

Replace the fixed early entropy gate with a joint decision rule:

- entropy hold only when:
  - `beam_entropy > 0.92`
  - `directional_margin < 0.20`
  - `beam_best_score < 0.05`
- coherent flat-basin hold when:
  - `beam_flat_mass >= 0.50`
  - `beam_best_score < 0.10`
  - `beam_entropy <= 0.92`
- score hold when:
  - `beam_best_score <= -0.05`
- basin hold when:
  - `max(beam_long_mass, beam_short_mass) < 0.55`
  - or `abs(beam_long_mass - beam_short_mass) < 0.15`
- otherwise act using the stronger directional mass

This distinction matters:

- coherent flat basin is evidence
- high-entropy diffuse futures are uncertainty

They should not collapse into the same hold reason.

### Kernel modes

The shadow path should support three explicit learned-kernel modes:

- `global`: train from all eligible per-step trader logs
- `per_asset`: train from symbol-matched logs first, then fall back to global
- `residual`: combine `global` and `per_asset` bucket statistics using
  `global + w * (asset - global)` on aligned labels, with `w=1.0` by default

Required shadow log metadata:

- `shadow_kernel_mode`
- `shadow_kernel_fallback_used`
- `shadow_kernel_source_count`
- `shadow_kernel_bucket_count`

### Standardized comparison output

Stop using ad hoc one-off analysis snippets for shadow comparisons. Add a
dedicated script that emits timestamped markdown + PNG artifacts for BTC/SPY
mode comparisons and reports:

- hold ratio
- hold-reason distribution
- live/shadow divergence
- entropy stats
- flat-mass support
- entropy vs future abs-return correlation
- contraction vs trend-strength correlation
- basin-edge vs next-move correlation

## Hold-remediation outcome

The structured-diagnostics + kernel-mode comparison rerun is captured at
`logs/shadow/*_20260313T061237Z.*` and
`logs/shadow/shadow_signal_report_20260313T061237Z.md`.

What landed:

- explicit primary hold causes are now logged
- `global`, `per_asset`, and `residual` modes are now comparable
- the entropy veto is no longer the dominant unexplained blocker

What the rerun showed:

- all six mode/market runs remained `shadow_hold = 1.0`
- hold attribution is now mostly `score`, with `flat_basin` secondary
- `entropy` did not remain the dominant hold cause after the calibrated gate
- `residual` did not outperform the simpler `global` / `per_asset` modes in the
  first comparison pass

That changes the diagnosis again:

- attribution and kernel comparison are no longer the blocker
- the next blocker is the action functional itself
- the next futures-policy milestone should retune `ActionWeights`

## ActionWeights retune target

The immediate goal of the next step is narrow:

- keep structured hold attribution unchanged
- keep kernel-mode comparison unchanged
- shift the default action functional so some high-directional-mass shadow states
  stop failing purely on score

This retune should prefer:

- higher reward for signed return, alignment, actionability, and contraction
- lower penalty for branch risk, diffusion, stress, drawdown, churn, and
  incoherent inventory

Success condition:

- at least one of `global`, `per_asset`, or `residual` should stop being
  100%-hold on the BTC/SPY rerun
- hold attribution should remain explicit and comparable

## First ActionWeights retune outcome

The first retune rerun is captured at `logs/shadow/*_20260313T062106Z.*` and
`logs/shadow/shadow_signal_report_20260313T062106Z.md`.

What improved:

- the score-dominated all-hold regime was broken
- hold attribution remained explicit
- kernel-mode comparison remained intact

What overshot:

- BTC `global` / `per_asset` act ratios are about `0.996`
- BTC `residual` is `1.0` act
- all SPY modes are `1.0` act
- flat-mass support collapsed toward zero in most modes
- live/shadow divergence jumped to about `1.0`, which is too far in the other
  direction

That means the retune worked mechanically but not yet behaviorally. The next
step is moderation:

- reduce the near-all-act bias
- preserve explicit hold attribution
- keep the multi-kernel comparison harness unchanged

## Calibration A/B modes (current milestone)

To stabilize the score distribution and avoid the all-hold/all-act collapse, the
shadow pipeline now supports explicit A/B modes for score normalization, gating,
and kernel structure:

### Score normalization modes

- `ratio`: `score = R / (1 + P)`
- `scaled_diff`: `score = scale * (R - P)` (scale chosen to center score)
- `logistic`: `score = tanh(scale * (R - P))`

Where `R` is the reward block and `P` is the penalty block.

### Gating modes

- `lex`: entropy → basin margin → score (with flat-mass score adjustment)
- `joint`: joint thresholds with flat-mass adjustment
- `score_only`: score threshold only (diagnostic)

Flat mass adjustment:

```
score_adjusted = score * (1 - flat_mass)
```

### Kernel modes

- `global`
- `per_asset`
- `residual` (kept only for comparison)
- `shrinkage` (Bayesian mixing of global and per-asset with lambda from sample size)

### Soft acceptance gates (tracked, not enforced)

- action rate target: `5%–25%`
- sign accuracy on ACT: `> 52%`
- Sharpe improvement vs baseline: `> 0.2` (proxy Sharpe from per-step returns)
- hold discipline: `E[|return| | HOLD] < E[|return| | ACT]`

These metrics are reported in the shadow comparison reports but do not gate
execution yet.

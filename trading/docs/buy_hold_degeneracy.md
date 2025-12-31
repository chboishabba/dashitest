# Buy-and-hold Degeneracy and Minimal Thesis Memory

Related context: `TRADER_CONTEXT.md:93128`, `TRADER_CONTEXT.md:93680`, `TRADER_CONTEXT.md:99346`, `TRADER_CONTEXT.md:100096`, `TRADER_CONTEXT.md:100108`, `TRADER_CONTEXT.md:100133`, `TRADER_CONTEXT.md:100197`, `TRADER_CONTEXT.md:100814`, `TRADER_CONTEXT.md:100947`, `TRADER_CONTEXT.md:100951`, `TRADER_CONTEXT.md:101004`, `TRADER_CONTEXT.md:101013`, `TRADER_CONTEXT.md:101034`, `TRADER_CONTEXT.md:101035`, `TRADER_CONTEXT.md:101107`, `TRADER_CONTEXT.md:101305`.

Below is a clean, formal answer to both parts. It is explicit and aligned with the current ternary controller.

---

# Part I - Why buy-and-hold is degenerate but optimal under monotone trends

## Setup

Let ((P_t)_{t=0}^T) be a price process with strictly monotone drift:

[
P_{t+1} = P_t(1+r_t), \qquad r_t >= \epsilon > 0
]

Ignore leverage, costs, and constraints for now.

Let a trading policy be a binary exposure process:

[
x_t in {0,1}
]

Wealth evolution:

[
W_T = W_0 \prod_{t=0}^{T-1} (1 + x_t r_t)
]

---

## Claim 1 - Buy-and-hold dominates any policy with exits

Buy-and-hold policy:
[
x_t \equiv 1
]

Any other policy has at least one t such that (x_t = 0).

### Proof

For any t with (x_t = 0),

[
(1 + x_t r_t) = 1 < (1 + r_t)
]

Thus:

[
\prod_{t=0}^{T-1} (1 + x_t r_t)
<
\prod_{t=0}^{T-1} (1 + r_t)
]

So:

[
W_T^{\text{any exits}} < W_T^{\text{buy-hold}}
\quad\blacksquare
]

---

## Claim 2 - Buy-and-hold is degenerate

A policy is degenerate if it ignores information without penalty.

Buy-and-hold:

* ignores volatility
* ignores drawdowns
* ignores risk
* ignores epistemic uncertainty

Yet it is optimal because the environment is monotone.

Optimality here is an artefact of the environment, not intelligence of the policy.

Formally:

* The optimal policy class collapses to a single constant control
* The control law has zero dependence on state

This is the definition of degeneracy.

---

## Claim 3 - Any cautious controller is strictly suboptimal under monotone drift

Your controller enforces:

* exits under uncertainty
* stress gating
* edge alignment

These induce (x_t = 0) on some set (S \subset {0,...,T-1}).

Even if |S| = 1,

[
\log W_T^{\text{controller}}
=
\sum_{t \notin S} \log(1+r_t)
<
\sum_{t=0}^{T-1} \log(1+r_t)
]

So:

Risk-aware controllers are provably dominated in monotone worlds.

This is not a failure -- it is the price of epistemic safety.

---

## Interpretation

* Buy-and-hold wins because it assumes the answer in advance
* Your system refuses that assumption
* Therefore it must underperform in monotone regimes

This is a theorem, not an opinion.

---

# Part II - Minimal thesis-memory extension that preserves your formalism

Now the constructive part.

You want compounding without abandoning ternarity.

The key constraint:

Direction remains ternary.
Memory must not introduce continuous control.

---

## Core idea

Add one scalar state variable:

[
m_t in {0,1,...,M}
]

Interpretation:

* m_t = thesis confidence depth
* Not direction
* Not size
* Not probability

This is ordinal memory, not metric memory.

---

## State variables (minimal)

You already have:

* A_t in {-1,0,+1} -- action
* P_t in {-1,0,+1} -- permission
* E_t in {-1,0,+1} -- edge
* S_t -- stress (continuous, veto only)

Add:

* m_t in {0,...,M}

That is it.

---

## Transition rules (precise)

### 1. Entry

If:

[
A_t != 0
and
m_{t-1} = 0
]

then:

[
m_t = 1
]

---

### 2. Reinforcement (thesis holds)

If:

[
A_t = A_{t-1} != 0
and
P_t = +1
]

then:

[
m_t = min(m_{t-1} + 1, M)
]

---

### 3. Soft veto (uncertainty)

If:

[
A_t = 0
but
m_{t-1} > 0
]

then:

[
m_t = m_{t-1} - 1
]

No exit yet.

---

### 4. Hard veto (risk / ban)

If:

[
P_t = -1 or stress > limit
]

then:

[
m_t = 0, A_t = 0
]

Immediate exit.

---

### 5. Exit condition (memory exhausted)

Exit only when:

[
A_t = 0 and m_{t-1} = 1
]

This guarantees:

* minimum hold duration
* hysteresis
* compounding

---

## Why this preserves your formalism

### Direction remains ternary

* No continuous sizing
* No probabilities
* No gradients

### Memory is ordinal

* You never compare magnitudes
* Only thresholds at 0 and M

### Risk veto remains absolute

* Stress can still kill everything instantly

### Buy-and-hold emerges only when justified

In monotone regimes:

[
A_t = +1 for all t
=> m_t -> M
]

Effective buy-and-hold, but earned, not assumed.

---

## Why this is minimal (proof sketch)

You cannot:

* eliminate memory (you already saw why)
* replace memory with thresholds (that is still memory)
* encode memory in continuous state without breaking ternarity

Therefore:

A bounded ordinal memory is the minimal extension.

Any less:

* no compounding

Any more:

* continuous control / RL

---

## Conceptual summary

* Buy-and-hold is optimal only because it cheats by assuming monotonicity
* Your controller is correct to reject that assumption
* To earn compounding, you must remember that you were right
* The smallest way to do that is a bounded thesis counter

---

## Empirical confirmation (15s sanity run)

This run confirms the proof and diagnosis in practice. See `README.md` for the run command and per-source summaries.

Invariant A - Duration collapse:

* Almost all trades have `dur=1`
* A few have `dur>1`, but only when the engine stalls briefly
* Exit reason is always `flat`

This implies a one-step greedy controller that re-evaluates each bar from scratch.
See `TRADER_CONTEXT.md:100096` and `TRADER_CONTEXT.md:100108` for the memoryless-controller framing.

Invariant B - Noise-scale PnL:

* Per-trade PnL is tiny and alternating
* Aggregate PnL is near zero over short windows
* BTC/YF produces large losses when volatility spikes

This matches a memoryless controller that monetizes variance, not trend.

Invariant C - Daily data leads to no trades:

* Yahoo daily series produce 0 trades and flat capital
* The edge signal collapses under coarse bars and the controller refuses exposure

Together these confirm the buy-and-hold dominance result in monotone regimes. See
`TRADER_CONTEXT.md:100133` for the theorem statement and `README.md` for the run.

---

## Plot readout (thesis-memory run)

This plot is the first empirical signal that the controller has left the 1-bar regime. The
key evidence is temporal coherence, not absolute values.

Panel: Thesis memory (depth / signal / hold)

* Depth is nonzero for long contiguous stretches.
* Depth is flat or slowly varying instead of flickering.
* Depth does not reset to zero every bar.

Interpretation: the controller now maintains temporal commitment, enabling compounding. See
`TRADER_CONTEXT.md:93128` for persistence framing.

Panel: Posture row (BAN / HOLD / ACT)

* Long stretches of HOLD with sparse ACT bursts.
* BAN events are rare.

Interpretation: decisions persist and only change at regime boundaries.
See `TRADER_CONTEXT.md:73737` for posture semantics.

Panel: Price + actions + bad_flag

* Actions cluster near regime boundaries.
* Actions align after stress spikes, not before.

Interpretation: causal alignment replaces noise chasing.

Panel: PnL vs HOLD%

* HOLD% is high and stable.
* PnL drifts with hold time rather than flip frequency.

Interpretation: returns are duration-driven rather than noise-driven.

Panel: p_bad + bad_flag

* p_bad fluctuates.
* bad_flag fires on sharper events.
* Thesis depth survives mild p_bad bumps.

Interpretation: hard risk veto, soft uncertainty decay, and hold behaviour are
working in the intended hierarchy.

Panel: MDL / Stress / GoalProb

* MDL decays smoothly.
* Stress trends downward.
* Goal probability does not spike artificially.

Interpretation: the system gains persistence without collapsing into buy-and-hold.

Panel: Plane rates

* Lower amplitude and frequency.
* More symmetric activity.

Interpretation: the policy requires less description length after adding memory.

---

## Implementation notes (non-code)

This document specifies behavior only. A minimal implementation should:

* Add an ordinal state `m_t` (thesis depth) with configurable max `M`.
* Update `m_t` via the transition rules above at each step.
* Gate exit decisions on `m_t` exhaustion rather than immediate soft veto.
* Log `m_t` per step and per trade when present.
* Expose `M` via CLI and defaults in `run_trader.py` config.

## Exit-depth interpretation and next steps

All exits at `thesis_depth = 0` mean the controller never promoted to a deeper thesis
before closing. This is consistent with an MDL-conservative policy on streams that do
not justify higher-order memory. See `TRADER_CONTEXT.md:100814` for the formal framing.

Minimal next steps:

1) Thesis promotion rule: promote depth when `p_bad` stays elevated and segmentation
   reduces MDL (`Delta MDL_split < 0`).
2) Shadow thesis (diagnostic): track what depth-1 would have been without acting.
3) Synthetic regime break: inject a controlled volatility or drift break and confirm
   depth promotion + non-zero exit depths under known structure.
4) Add temporal hysteresis: apply a switching penalty when the action state flips,
   and log action run length / time since last switch to measure churn reduction.
   See `TRADER_CONTEXT.md:101414` for the action-space penalty framing.

## Shadow thesis diagnostic (spec)

Shadow thesis is a logging-only diagnostic that never changes action or depth.
It answers: was a deeper thesis available but declined?

At each step (or candidate segmentation points), compute a depth-1 split and log:

* Use a rolling window of `SHADOW_MDL_WINDOW` steps (default 200).
* `MDL_current` = mean `mdl_rate` over the full window.
* `MDL_split` = average of the mean `mdl_rate` in the left and right halves.
* `shadow_delta_mdl = MDL_split - MDL_current`.
* `shadow_would_promote = (shadow_delta_mdl < -eps)`.
* `shadow_is_tie = (abs(shadow_delta_mdl) <= eps)`.

Expected outcomes:

1) `shadow_delta_mdl >= 0` almost everywhere: memory is unnecessary.
2) brief negatives: promotion needs persistence or hysteresis.
3) sustained negatives in stress regions: promotion is justified.

If `shadow_delta_mdl` collapses to floating noise (machine epsilon), the split is
algebraically identical to the unsplit proxy. In that case, upgrade the diagnostic to
a refit-based split (left/right window parameters) and report promote/tie/reject counts
instead of raw negative counts. See `TRADER_CONTEXT.md:101107` for the minimal design.

Refit-based split outline (minimal):

* Window `W` around `t` with left `[t-W, t)` and right `[t, t+W)`.
* Fit params on left/right separately and on the combined window.
* `MDL_current = L(params_all) + L(residuals_all)`.
* `MDL_split = L(params_left)+L(residuals_left)+L(params_right)+L(residuals_right)+split_penalty`.
* Use `eps = 1e-12 * max(1.0, abs(MDL_current))` for promote/tie/reject gating.
* Defaults: `W=64`, `split_penalty=log(n)` with `n=2W`.

## Action switching penalty + run-length diagnostics

Introduce a switch cost in action space so leaving a stable action state requires
sustained evidence. This is a temporal hysteresis term, not a sizing rule.
Reference: `TRADER_CONTEXT.md:101414`.

Log run length statistics alongside actions:

* `action_run_length`: consecutive steps with the same `action_t`.
* `time_since_last_switch`: steps since `action_t` last changed.

These allow attribution of PnL to long runs vs churn and make it easy to detect
excessive switching even when MDL promote rates are stable.

## Plane rates diagnostic (logging-only)

Plane rates should oscillate around zero in monotone assets; sustained deviations
in noisy assets (BTC_YF) signal curvature without guaranteed direction. Use this
as a diagnostic before enabling promotion. See `TRADER_CONTEXT.md:101510`.

Diagnostics to add (logging/analysis only; no behavior change):

* Promotion rate vs `plane_abs` (absolute plane rate) and `stress`.
* Promotion rate vs `p_bad` and `capital_pressure`.
* PnL contribution vs action run length in each plane bucket.
* Rolling sign-flip counts for plane rate (`plane_sign_flips_W`) and a
  "would-veto" count to estimate the impact of a stability gate.

Per-step fields to log:

* `plane_abs = abs(plane_rate)`
* `plane_sign = sign(plane_rate)`
* `plane_sign_flips_W` (rolling sign flips within window `W`)
* `plane_would_veto = (plane_sign_flips_W > 1)` (diagnostic only)

If promotions cluster where `plane_abs` rises and `stress` rises, the detector is
aligned. If promotions fire on plane jitter alone, the split penalty is too weak.
See `TRADER_CONTEXT.md:101616`.

Stability gate candidate (defer implementation until diagnostics confirm):

* Do not promote when plane rate flips sign more than once within window `W`.
* This is a veto rule, not a new signal; it preserves HOLD in monotone series and
  prevents churn in heavy-tail streams. See `TRADER_CONTEXT.md:101638`.

Plane-stability diagnostic (logging-only):

* Log `plane_sign_flips_W` and `plane_would_veto` and aggregate:
  * promotion rate vs `plane_sign_flips_W`
  * mean ΔPnL vs `plane_sign_flips_W`
  * `shadow_would_promote` vs `plane_would_veto`
* References: `TRADER_CONTEXT.md:102286`, `TRADER_CONTEXT.md:102292`.

Run-length conditional diagnostic (logging-only):

* Bucket by `action_run_length`, plane sign persistence (e.g., flips in `W`),
  and stress quartiles.
* Aggregate mean ΔPnL per bucket to test whether longer runs and stable sign
  correlate with positive PnL. References: `TRADER_CONTEXT.md:101447`,
  `TRADER_CONTEXT.md:103231`.

## Decision geometry plots (logging-only)

Externalize controller behavior as geometry before any gating or parameter
sweeps. This uses existing logs only; no behavior changes. References:
`TRADER_CONTEXT.md:102058`, `TRADER_CONTEXT.md:102190`.
Implementation: `scripts/plot_decision_geometry.py`.

Primary decision heatmaps (same bins across all maps):

* X-axis: `plane_abs`
* Y-axis: `stress` (or `p_bad`)
* Color maps (three views):
  * action rate (HOLD/ACT/BAN)
  * promotion rate (promote/tie/reject)
  * mean ΔPnL per bin

Binning (fixed and explicit to avoid lying with resolution):

* `plane_abs`: 20 bins using quantiles (q05..q95) with underflow/overflow bins.
* `stress`/`p_bad`: 20 bins using quantiles (q05..q95) with underflow/overflow bins.
* Drop/grey bins with count < 30 to avoid single-point artifacts.
* If quantile edges collapse (many identical values), fall back to linear bins
  across the finite min/max range to keep bins unique.

Metrics (per bin):

* `promotion_rate = promote / (promote + reject)` (ties excluded).
* `action_rate = act / total` (ACT vs HOLD/BAN).
* `mean_pnl = mean(realized_pnl_step)` (or per-trade ΔPnL if aggregating trades).

Overlay diagnostic (time-series):

* Price, `plane_rate`, and `plane_abs` with vertical markers for
  shadow promotion, action, and would-veto events. See
  `TRADER_CONTEXT.md:102095`.

Ternary simplex (diagnostic clarity):

* Normalize `p_bad`, `plane_abs`, `stress` to sum to 1 and plot in a simplex.
  Color by action and mark promote/tie/reject. See `TRADER_CONTEXT.md:102116`.

Minimum per-step fields needed for plots:

* `plane_abs`, `stress`, `p_bad`, `action_t`, `shadow_would_promote`.

Plane-rate proxy:

* If a signed `plane_rate` column is not present, use `delta_plane` as the
  signed proxy and define `plane_abs = abs(delta_plane)` for plotting.

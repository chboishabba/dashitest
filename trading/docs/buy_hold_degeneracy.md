# Buy-and-hold Degeneracy and Minimal Thesis Memory

Related context: `TRADER_CONTEXT.md:93128`, `TRADER_CONTEXT.md:93680`, `TRADER_CONTEXT.md:99346`, `TRADER_CONTEXT.md:100096`, `TRADER_CONTEXT.md:100108`, `TRADER_CONTEXT.md:100133`, `TRADER_CONTEXT.md:100197`.

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

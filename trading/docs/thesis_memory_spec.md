# Thesis Memory State Machine Spec (Ternary-First)

## 0) Purpose

Thesis memory is a minimal, discrete commitment layer between instantaneous gating and
long-horizon policy learning.

Goals:
1) Stop churn-to-flat when the environment is stable and risk is acceptable.
2) Exit quickly when thesis becomes invalidated or unstable.
3) Stay fully ternary in decisions (direction and evidence), with only a tiny amount
   of discrete memory.

Non-goals:
- No new predictors required (except optional realized vol feature in v2).
- Not a portfolio optimizer. It is a stateful governor.

## 1) Definitions

### 1.1 Ternary set
T = {-1, 0, +1}

### 1.2 Core thesis state (persistent)

At each time step t, the thesis memory holds:

- d_t in T : thesis direction
  - 0 = no thesis
  - +1 = thesis-long
  - -1 = thesis-short
- s_t in {0,1,2} : thesis strength (0 none, 1 weak, 2 strong)
- a_t in N : thesis age (steps since entry)
- c_t in N : cooldown (steps remaining before a new thesis can be entered)
- v_t in {0,1,2} : invalidation accumulator

State tuple: M_t := (d_t, s_t, a_t, c_t, v_t)

### 1.3 Inputs required (per step)

- plane_sign_t in T
- plane_sign_flips_W_t in N
- plane_abs_t >= 0
- stress_t >= 0
- p_bad_t in [0,1]
- shadow_would_promote_t in {0,1}
- plane_would_veto_t in {0,1}

### 1.4 Derived trits (per step)

Define three evidence trits: alignment, stability, risk.

#### 1.4.1 Proposed direction d*_t

- If d_t = 0: d*_t := plane_sign_t
- Else: d*_t := d_t

#### 1.4.2 Alignment trit alpha_t in T

Using desired := d*_t:
- If plane_sign_t = 0 -> alpha_t := 0
- Else if plane_sign_t == desired -> alpha_t := +1
- Else -> alpha_t := -1

#### 1.4.3 Stability trit beta_t in T

- If plane_would_veto_t = 1 OR plane_sign_flips_W_t > 1 -> beta_t := -1
- Else if plane_sign_flips_W_t = 1 -> beta_t := 0
- Else (plane_sign_flips_W_t = 0) -> beta_t := +1

#### 1.4.4 Risk trit rho_t in T

Define stable cutpoints (logged, constant per run):

- p_bad_hi, p_bad_lo with 0 <= p_bad_lo < p_bad_hi <= 1
- stress_hi, stress_lo with 0 <= stress_lo < stress_hi

Then:
- If p_bad_t >= p_bad_hi OR stress_t >= stress_hi -> rho_t := -1
- Else if p_bad_t <= p_bad_lo AND stress_t <= stress_lo -> rho_t := +1
- Else -> rho_t := 0

## 2) Transition function

(M_t, inputs_t) -> (M_{t+1}, events_t)

### 2.1 Cooldown update (always runs)

- If c_t > 0: c' := c_t - 1
- Else: c' := 0

### 2.2 Entry logic (only when no thesis)

Entry preconditions (all must be true):
- d_t = 0
- c_t = 0 (or c' = 0)
- shadow_would_promote_t = 1
- beta_t >= 0
- rho_t >= 0
- d*_t != 0

Then enter:
- d_{t+1} := d*_t
- s_{t+1} := 1
- a_{t+1} := 0
- v_{t+1} := 0
- c_{t+1} := 0

Emit event:
- event = thesis_enter
- reason = shadow_promote
- log: d_enter, s_enter = 1

If entry preconditions fail, remain flat:
- d_{t+1} = 0, s_{t+1} = 0, a_{t+1} = 0, v_{t+1} = 0, c_{t+1} = c'

### 2.3 Update logic (only when thesis active)

If d_t != 0:

#### 2.3.1 Strength update via signed trit

sum := alpha_t + beta_t + rho_t
Delta_s_t := clip_T(sum), where:
- +1 if sum >= 1
- 0 if sum = 0
- -1 if sum <= -1

s_tmp := clip_{0..2}(s_t + Delta_s_t)

#### 2.3.2 Age update

a_tmp := a_t + 1

#### 2.3.3 Invalidation accumulator update

bad := (alpha_t = -1) OR (beta_t = -1) OR (rho_t = -1)

- If bad: v_tmp := min(2, v_t + 1)
- Else: v_tmp := max(0, v_t - 1)

### 2.4 Exit logic (only when thesis active)

Exit if any of:
1) s_tmp = 0
2) v_tmp = 2
3) a_tmp >= A_max

If exit triggers:
- d_{t+1} := 0
- s_{t+1} := 0
- a_{t+1} := 0
- v_{t+1} := 0
- c_{t+1} := C

Exit reason (precedence):
1) v_tmp = 2 -> thesis_invalidated
2) s_tmp = 0 -> thesis_decay
3) else -> thesis_timeout

Emit event:
- event = thesis_exit
- reason = ...
- include snapshot: (d_t, s_t, a_t, v_t) and (alpha_t, beta_t, rho_t, Delta_s_t)

If no exit:
- d_{t+1} := d_t
- s_{t+1} := s_tmp
- a_{t+1} := a_tmp
- v_{t+1} := v_tmp
- c_{t+1} := c'

Optionally emit event:
- thesis_update with reason reinforce/decay/hold based on Delta_s_t

## 3) Action interaction spec

Let proposed_action_t in {-1,0,+1} and final x_{t+1} in {-1,0,+1}.

### 3.1 Minimal thesis constraint

If thesis active (d_t != 0):

- If beta_t >= 0 AND rho_t >= 0:
  - Do not allow flattening unless an explicit risk stop triggers.
  - If proposed_action_t = 0, override to d_t.
  - Log override = thesis_hold_bias.

- If beta_t = -1 OR rho_t = -1:
  - Flattening is allowed.
  - If proposed_action_t = 0, accept and log:
    - thesis_unstable if beta_t = -1
    - thesis_risk if rho_t = -1

### 3.2 Flip-flop prevention

If thesis active (d_t != 0), forbid direct reversal unless thesis exits first.

- If proposed_action_t = -d_t:
  - If exit will occur this step, allow reversal next step after exit.
  - Else override to d_t or 0:
    - if rho_t = -1 or beta_t = -1 -> override to 0
    - else override to d_t
  - Log override = thesis_no_flipflop.

## 4) Reward spec rewrite (benchmark regret)

### 4.1 Variables

- Price P_t
- One-step log return r_t := log(P_{t+1}/P_t)
- Position x_t in {-1,0,+1}
- Transaction cost coefficient k >= 0
- tc_t := k * |x_{t+1} - x_t|

### 4.2 Benchmark exposure

Choose constant benchmark exposure x_bar in [0,1].

### 4.3 Reward

Regret reward:
R_t := (x_t - x_bar) * r_t - tc_t

### 4.4 Optional risk-adjusted regret (v2)

When realized vol sigma_t is available:
R_t := (x_t - x_bar) * r_t - lambda * |x_t| * sigma_t - tc_t

## 5) Logging requirements

Every step log:

### 5.1 Thesis state
- thesis_d, thesis_s, thesis_a, thesis_c, thesis_v

### 5.2 Evidence trits
- thesis_alpha, thesis_beta, thesis_rho
- thesis_ds, thesis_sum

### 5.3 Events
- thesis_event (thesis_enter, thesis_update, thesis_exit, thesis_none)
- thesis_reason
- thesis_override (optional)

### 5.4 Reward components
- r_step
- tc_step
- benchmark_x
- reward_regret
- risk_penalty (optional, when v2 is implemented)

## 6) Parameters (constants per run)

- W : window used for plane_sign_flips_W
- A_max : max thesis duration
- C : cooldown duration
- p_bad_lo, p_bad_hi
- stress_lo, stress_hi
- k : transaction cost coefficient
- benchmark_x : x_bar

Suggested starting points (not binding):
- A_max = 50 to 300 steps (depends on bar size)
- C = 3 to 10 steps
- k small but non-zero so churn is visible

## 7) Acceptance tests

1) Entry only on promote + nonnegative stability/risk.
2) Invalidation exits fast (two consecutive bad evidences hit v=2).
3) Strength clips properly to {0,1,2}.
4) Cooldown blocks re-entry for C steps after exit.
5) No flip-flop (cannot go from d to -d in one step while active).
6) Reward penalizes flat in uptrend if benchmark_x = 1.

## 8) Implementation interface (recommended)

- ThesisState(d:int, s:int, a:int, c:int, v:int)
- Inputs(plane_sign:int, flips:int, plane_abs:float, stress:float, p_bad:float,
  shadow_promote:int, would_veto:int)
- Derived(alpha:int, beta:int, rho:int, ds:int, sum:int)

Pure function:
step_thesis(state, inputs, params) -> (new_state, derived, event)

Then:
apply_thesis_constraints(state, derived, proposed_action, risk_stop_flag) -> (final_action, override_event)

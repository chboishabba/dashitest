# Compactified Context

## Kernel & formal semantics (locked)
- Kernel is a ternary projection (not a feature extractor), involution- and Gamma-symmetry-equivariant, defined on the quotient (not representatives); learning selects among closed quotients.
- Kernel energy E(s) is a consistency diagnostic for closure/MDL within the kernel system; trading monotones are enforced on the action/capital channel by Meta-Witness and the capital kernel.
- Binary constructs are derived: carrier T={-1,0,+1}; ternary admits support/sign factorization; activity mask m_t = pi_supp(d_t) ("trade occurs") is not a label.
- Renormalisation tower compatibility: Rbar_j o Kbar(j) = Kbar(j+1) o Rbar_j and pi_j o K(j) = Kbar(j) o pi_j (quotient equality).
- Kernel updates: synchronous update is canonical; async/block-sequential are implementation variants only.
- Phases/Witness act only on the action channel and are idempotent supervisors; they never redefine K.
- Learner/adapter learns quotient invariants and outputs legitimacy/confidence scalars only; it cannot set direction/size and only gates permission.

## Kernel tower, commutation, and witness
- Nested/convolved kernels are the renormalisation tower Rbar_j with Kbar(j) commuting on the quotient: Rbar_j o Kbar(j) = Kbar(j+1) o Rbar_j.
- Representative space at scale j: S^(j) := T^{G^(j)} x C with gauge group G, quotient Q^(j) := S^(j)/G.
- Kernel is well-defined on the quotient: pi_j o K(j) = Kbar(j) o pi_j (learning/selection act on Q=S/G, not on representatives).
- Phase/Witness is an idempotent endomap on the action channel, not a second dynamics: W o W = W and W composes with A(s), not with K(s).
- Single commutative diagram (tower + quotient + witness):
```text
S^(j) --K^(j)--> S^(j)
 | \             / |
 |  \           /  |
 |   pi_j     pi_j  |
 v     \     /      v
Q^(j) --Kbar^(j)--> Q^(j)
 |                 |
 | Rbar_j          | Rbar_j
 v                 v
Q^(j+1) --Kbar^(j+1)--> Q^(j+1)
 ^                 ^
 | pi_{j+1}        | pi_{j+1}
 |                 |
S^(j+1) --K^(j+1)--> S^(j+1)

Action channel supervisor:
A --W--> A  (W o W = W; W composes with A(s), not K)
```

## Gates, regimes, and posture
- Phase-4 density monitor (see `scripts/phase4_density_monitor.py`, profile `configs/phase4_monitor_profiles.json`) enforces: per-ontology balance >=25%, bin persistence >=6/8, >=40 rows/bin, monotonic medians, effect-size >=0.0005. Gate success requires 3 consecutive OPENs; each iteration logs `blocking_reason` in `logs/phase4/density_monitor.log` and JSON outputs.
- Phase-4 density gate stays strict and opens only when Phase-07 readiness persists (rho_A threshold + persistence + boundary stability).
- Phase-07 asymmetry density (certificate, not signal): edge = x_{t-1} * Delta m_t; cost = (h_t + f_t) * m_t * |Delta x_t|; rho_A = sum edge / (sum cost + eps). Phase-04 unblocks only when rho_A clears threshold with persistence, survives small execution perturbations, and profit eigen-events pass special-code escalation (VOID/PARADOX).
- Control-law clarifications: UNKNOWN/M5 means HOLD (buffer), not forced FLAT; gate-first hazard semantics ("all-red spine" / Meta-Witness) guard exposure before commitment.
- Posture is derived from the quotient tower (not primitive): p_t := Pi_p((q_t(j))_j) in {-1,0,+1} (Hostile/Tense/Coherent). posture_observe documents the observe-only path in Phase-6 gating (`docs/stream_daemon.md`, `docs/phase6_capital_control.md`, `docs/posture_observe_notes.md`).
- Minimal profitable architecture = triadic condition classifier -> cheap strategy layer (only when +1) -> execution/risk clamps. Sense fast, commit slow (per-second sensing; minute/hour commitment).
- Live gate invariant: closed Phase-6 gate forces posture_observe (direction=0, target_exposure=0, hold=true, actionability=0), so Phase-07 is empty; only a `capital_controls_*.jsonl` with `allowed=true` wakes the stack. Decision payloads carry `phase6_gate` snapshots (`open`, `source`, `allowed_slip_bps`, `reason`) for refusal diagnostics; recommended wake-up proof is to inject a slip approval and restart the live daemon.
- Live Phase-07 finding: with gate open and coherent posture, the controller ramped long exposure every 1-3s (e.g., NEOUSDT), giving non-empty support, but Phase-07 net stayed <=0 because boundary costs dominate at that cadence (noise-scale returns vs cost on each Delta x). This is a boundary-dominated eigen-orbit, not a wiring bug. Experiments: lift horizon to 30-120s with cost on entry/exit, run zero-cost sanity to confirm wiring, and align actuation/measurement (slower clock, batched/impulse changes, or charge cost only on sign flips/regime exits).
- Phase-8 entry criteria (net-surplus gate; formal):
  Phase-8 is a supervisory predicate on the action-channel certificate and readiness state. It does not redefine K or the quotient tower.
  Define the Phase-7 boundary certificate:
    b_t := (rho_A(tau), persistence, stability, tau, cost_model_id)
  where rho_A(tau) is computed on an actuator/horizon pairing fixed to tau (decimated/impulse mode) and stability requires robustness under small execution/cost perturbations.

  Phase-8 OPEN iff all of:
    (E8.1) Actuator mode fixed+logged: actuator_mode in {decimate, impulse, batch} is constant over the evaluation window; tau is constant; cost_model_id is constant.
    (E8.2) Ledger-ready: for every action update, the log provides Delta x_t, m_t, and the cost inputs (half-spread, fees, slippage proxy) sufficient to recompute b_t offline.
    (E8.3) Non-empty support: activity support meets minimum density (e.g., n_support >= N_min AND activity_rate within bounds), preventing "rho_A from noise."
    (E8.4) Net-extractable: rho_A(tau) >= theta_A with persistence over W windows (or K consecutive evaluations), and remains >= theta_A under +/- eps perturbations of the cost model.
    (E8.5) Horizon existence: a sweep over tau in [10s, 300s] finds at least one tau* satisfying (E8.4) (this prevents locking into a boundary-dominated cadence).

  Phase-8 CLOSED otherwise, with boundary clamp: if ready=false at any point, force posture_observe (direction=0, exposure=0) and record blocking_reason.

  Required artifacts (Phase-7.* deliverables):
    - Phase-7.1 Actuator testability: daemon supports decimation/impulse/batch with logged mode/tau.
    - Phase-7.2 Execution ledger in Phase-7 schema: sufficient fields to recompute edge/cost and support.
    - Phase-7.3 Horizon sweep harness (10-300s): outputs rho_A vs tau and robustness classification.

  One-command audit report must output: gross vs net density, activity rate, cost-vs-move share, robustness verdict, worst drawdown proxy (if available), and blocking_reason when CLOSED.
- Phase-9 intent (certified operator): capital becomes a state variable; multi-regime certification proves profitability in >=1 regime and preservation elsewhere; learning is capital-budgeted; Meta-Witness can freeze or force OBSERVE; every action stores a justification chain. Success = unattended operation with automatic throttles, refusals, and justification artifacts. See `docs/phase9_legitimacy.md`.
- Phase-9 kernel details: capital update C_t = C_{t-1} + C_{t-1}x_{t-1}r_t - C_{t-1}kappa_t|Delta x_t| - C_{t-1}(lambda_dd[DD_t]^+ + lambda_sigma*sigma_hat_t + lambda_post*1_{p_t != +1}) with clamps and exposure bound |x_t| <= min(x_max, beta * C_t/(C_t(kappa_t+eps))). Phase-07 net certificate is edge/cost density; capital kernel accounting uses return-minus-cost (e.g., median(x_{t-1}r_t - kappa_t|Delta x_t|)). Meta-Witness refusals R0-R5 cover missing authority, net<=0, sparse support, drawdown breach, cost-dominated churn, policy inconsistency. Integration points: capital kernel + Meta-Witness intercept before action emission in `scripts/stream_daemon.py` and `engine/loop.py`; log capital/mw fields and justification chain.

## Eigen-events / Phase-07
- Eigen-events are quotient-stable, boundary-surviving objects (not raw profitable blips): they persist under friction, horizon variation, and small execution/cost perturbations.
- False eigen-event = numerator > 0 without boundary stability; true eigen-event = survives friction, horizon variation, and small perturbations (boundary-stable fixed object).
- Phase-04 unblocks only when rho_A clears threshold with persistence and special-code escalation (VOID/PARADOX) remains satisfied.
- Deliverable: `scripts/phase07_eigen_boundary_check.py` taking (d_t, m_t), returns stream, and friction model; outputs rho_A vs horizon, +/- eps cost robustness, and classification {false, boundary-unstable, boundary-stable}.
- Horizon decimation harness (Phase-7.3): sweep 10-300s to show boundary dominance vs scale and locate any positive net density.

## Formal deliverables & locks
- Math clarifications to keep (paper-grade): kernel definition section with synchronous update as canonical (async/block-seq only as variants); E(s) is a kernel-consistency diagnostic (MDL-optimal within the kernel system); trading monotones are on the action/capital channel; ternary admits support/sign factorization; short biological analogy tying kernel eigen-objects to stable biological modes (Kluver-style).
- Observability invariant: CLOSED gate => posture_observe; OPEN => triadic posture -> cheap strategy -> execution clamps; refusals carry reason codes; no learner->execution leakage.
- Do-not-change: Phase-4 density gate; no upstream binary labels; no learner bypass of gates; no treating short-horizon profitability as eigen-evidence.
- Status: conceptual alignment complete; remaining work is instrumentation and proof, not theory.
- Optional next steps: tighten Phase-07 parser alignment to the live daemon schema; add the Phase-7.3 horizon sweep harness; formalize VOID/PARADOX typing in Meta-Witness refusals.

## Motif/phase mapping & nested kernels
- Phases are supervisory predicates on quotient objects (operational gates), not a second dynamics or new primitives.
- The Phase ladder is consistent with M1..M9 motifs and nested/convolved kernels (renormalisation tower Rbar_j, commutation on Q).
- Phase-6 posture_observe maps to M5 buffer (UNKNOWN => HOLD, not forced FLAT).
- Phase-7 boundary asymmetry maps to boundary-quotiented eigen-objects.
- Phase-8 net-surplus gate maps to M8 construct-aware; requires actuator/horizon/ledger readiness.
- Phase-9 Meta-Witness/capital kernel maps to M9 supervisor with refusal levels and capital clamps.
- Dual-kernel tension (M6): Kernel A = legitimacy/structural health; Kernel B = exploitability/forced behavior; contradictions default to inactivity.
- Caveat: do not let "phase" become a second dynamics; phases gate admissibility and action, but never redefine K.

## M9 state & refusal logic (nested tensor view)
- Canonical M9 state variables (always present): kernel/quotient state (s in S, [s] in Q, q_t(j), Kbar(q_t(j))), posture p_t in {-1,0,+1} with support=pi_supp(p_t) and sign=pi_sgn(p_t) when support=1, Phase/Witness state (phase in {0..9}, gate_open, refusal in {NONE,HOLD,BAN}, witness_id), capital state (C_t, Delta C_t, drawdown, capital_at_risk, capital_kernel_state), action channel (d_t, x_t, m_t, slip_allow, horizon), boundary diagnostics (edge_t, cost_t, rho_A, persistence, boundary_stable).
- Single graded state: X_t = (q_t(j) tower in Q(j) with q_t(j+1)=Rbar_j(q_t(j)), kernel residuals kappa_t(j), posture p_t, boundary cert b_t=(rho_A, persistence, stability, tau, cost_model_id), capital state c_t=(C_t, Delta_C_t, drawdown, risk_budget, flags), witness w_t=(refusal in {NONE,HOLD,BAN}, rule_id, reason), action channel a_t=(d_t, x_t, m_t=pi_supp(d_t), slip_allow)).
- Tower coherence: q_t(j+1) = Rbar_j(q_t(j)) for all j; closure is certified when kappa_t(j) <= eps_j on required scales.
- Posture is derived from the tower (p_t = Pi_p((q_t(j))_j)); it is not a primitive label.
- Boundary certificate b_t is the Phase-07 object; capital kernel uses return-minus-cost accounting separately.
- Meta-Witness is idempotent on the action channel; applying it twice changes nothing further.
- Action channel is downstream of all above; m_t is derived via pi_supp(d_t).
- Projections: M5 = (p_t, refusal>=HOLD when p_t=0); M6 tension uses legitimacy score from (kappa,p) vs exploitability from b_t; M7 = b_t; M8 = (b_t, actuator_mode, ledger_ready, horizon_sweep_ready); M9 = (c_t, w_t, a_t) gated by (q, kappa, p, b).
- Positive M9 state (admissible/acting): tower closure on required scales (kappa_t(j) <= eps_j), posture=+1, rho_A stable/persistent > threshold, boundary_stable=1, capital SAFE, gate_open=1/refusal=NONE, action direction=sign(posture) with exposure>0 and m_t=1.
- Negative M9 state (inadmissible/refused): kernel instability or posture -1/0 or dual-kernel contradiction => HOLD; boundary failure or non-persistent rho_A => false eigen-event => HOLD; capital violation => BAN; witness override for actuator/horizon/ledger inconsistencies; enforced action = direction/exposure/m_t = 0 (posture_observe).
- Projection-only summary table (not the object): Kernel = vector closed(j), Boundary = b_t, Witness = w_t, Capital = c_t, Action = a_t; any "positive/negative" table is a coarse image of X_t.
- Coarse projection table (ops-only, not the mathematical object):
```text
Dimension | Positive M9                 | Negative M9
Kernel    | closed across required j    | open / unstable
Posture   | +1                          | 0 or -1
Boundary  | rho_A stable                | false / dominated
Capital   | SAFE                        | STRESSED / VIOLATED
Witness   | silent                      | HOLD / BAN
Action    | ACT                         | OBSERVE / REFUSE
```

## Learner / ell contract
- Learner outputs ell/legitimacy scalars only; it never sets direction/size and cannot bypass gates.
- Learner is PnL-free online; retrain from logged qfeat streams; gating and refusal logic remain on CPU.

## qfeat + GPU/Vulkan contract
- Frozen qfeat vector order (dimensionless, shift/scale invariant): [vol_ratio, curvature, drawdown, burstiness, acorr_1, var_ratio]. CPU reference (compute_qfeat, w1=64, w2=256, EPS=1e-6) uses log returns, std/range vol ratio, log std of second differences, normalized drawdown, L1/L2 burstiness, clipped lag-1 autocorr, fast/slow variance ratio; NaN/Inf squashed to 0.
- GPU/CPU split: GPU computes window->qfeat (and later qfeat prediction + ell); CPU handles hysteresis/regime gating, logging, and BAN logic. Only ell (plus tiny debug vector if enabled) crosses GPU->CPU. Keep learner PnL-free online; retrain from logged qfeat streams.
- Parity harness: `python tools/parity_qfeat.py --tape logs/qfeat_tape.memmap --backend vulkan --force --w2 16 --w1 4` and keep worst diffs <=2e-4 before shader edits. Shader boundary lives at `vulkan_shaders/qfeat.comp`; parity helper in `tools/parity_qfeat.py`; Vulkan adapter scaffolding in `trading/vk_qfeat.py` and `trading/strategy/vulkan_tape_adapter.py`.
- GPU implementation order: freeze window->sheet->qfeat path, add ell on GPU, wire ell into gate (CPU), backtest for ACT suppression in bad regimes, then add small predictor head if needed. Keep CPU fallback for determinism.
- Implementation contract appendix: `docs/appendix_kernel_vulkan.md` captures quotient-first semantics, GPU/CPU split, determinism, and audit requirements.

## Tape/replay workflow
- `scripts/run_with_vulkan_tape.py --force` rebuilds qfeat tape and overwrites slot 6 (ell); add/use `--reuse-tape` to replay without rebuilding (should not require --prices-csv when reusing). Default = build if missing; --force = rebuild; --reuse-tape = error if tape absent.
- Replays/plots: ensure ell/legitimacy_margin columns are written so acceptability/hysteresis plots show real signal; avoid caching confidence outside the adapter.

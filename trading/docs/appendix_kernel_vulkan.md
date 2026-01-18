# Appendix: Kernel-Vulkan Correspondence (Implementation Contract)

This appendix specifies the correspondence between the formal DASHI kernel tower and the Vulkan implementation path. The goal is to preserve quotient-first semantics, equivariance, and deterministic closure diagnostics while allowing GPU acceleration for feature construction and (optionally) learner-side scalar outputs.

## A. Formal objects (recap)

Let the ternary carrier be \(T=\{-1,0,+1\}\). At scale \(j\), the representative state is
\[
S^{(j)} := T^{G^{(j)}\times C},
\]
with gauge group \(G\) acting on representatives, yielding the quotient \(Q^{(j)} := S^{(j)}/G\).
The kernel is a projection-like operator \(K^{(j)}: S^{(j)}\to S^{(j)}\) that induces a well-defined map \(\bar K^{(j)}: Q^{(j)}\to Q^{(j)}\) such that
\[
\pi_j\circ K^{(j)} = \bar K^{(j)}\circ \pi_j.
\]
A renormalisation/coarse-graining tower is defined by maps \(R_j: S^{(j)}\to S^{(j+1)}\) inducing \(\bar R_j: Q^{(j)}\to Q^{(j+1)}\), with compatibility in the quotient:
\[
\bar R_j\circ \bar K^{(j)} = \bar K^{(j+1)}\circ \bar R_j.
\]

Implementation must preserve these equalities up to numeric tolerances only where the formalism explicitly permits (for example, float32 computation of derived diagnostics), and must not introduce any "kernel redefinition" via execution machinery.

## B. Implementation split: what runs on GPU vs CPU

### B1. GPU responsibilities (Vulkan)

The GPU is permitted to compute derived, gauge-invariant observables used for gating/monitoring and for learner inputs, for example:

- Windowed feature maps \(F: \text{prices}\mapsto \mathbb{R}^d\) (qfeat)
- Optional predictor head \(\hat F\) and scalar legitimacy/confidence \(\ell\)

The GPU is not permitted to:
- Emit action directives (direction/size)
- Encode hidden state that bypasses logging
- Mutate kernel state or quotient representatives

### B2. CPU responsibilities

The CPU retains:
- All supervisory predicates (Phase-4/6/7/8/9 gates)
- All action-channel semantics (hysteresis, posture_observe, Meta-Witness)
- Any BAN / refusal logic and justification-chain assembly
- Deterministic replay and audit report generation

This preserves the principle: learning can gate permission but cannot act.

## C. Correspondence of "nested tensors" to GPU buffers

The formal tower \((q^{(j)})_{j=0}^J\) is represented operationally by:
- A base observation stream (prices/bars) plus
- A multiscale derived stream (features at multiple windows / decimations)

Concretely, define a set of window pairs \((w_1^{(j)}, w_2^{(j)})\) for each scale \(j\). The GPU computes:
\[
f_t^{(j)} := F(\text{prices}; w_1^{(j)}, w_2^{(j)}),
\]
which is used as a coordinate chart for quotient-relevant observables, not as the quotient element itself. The CPU then constructs gating/posture functionals \(\Pi\) over the graded feature family \((f_t^{(j)})_j\), yielding posture \(p_t\in\{-1,0,+1\}\).

This is the implementation analogue of treating the tower as a graded object: the GPU produces the graded observables; the CPU uses them to evaluate supervisory predicates.

## D. Equivariance and determinism constraints

### D1. Determinism

GPU computation must be bitwise stable for a given device/driver configuration where feasible, and numerically stable across small perturbations. The required contract is:

- CPU reference implementation is canonical.
- GPU results must match CPU within tolerance <= 2e-4 (or tighter if achieved) for all qfeat dimensions.
- NaN/Inf must be squashed to 0 identically on CPU and GPU.

Parity harnesses must be maintained for every shader change.

### D2. Gauge / symmetry discipline

Any GPU-computed observable used for gating must be:
- Shift/scale invariant where claimed
- Independent of representative choice within gauge class (to the extent the observable intends to reflect quotient structure)

If an observable is not gauge-invariant, it must be explicitly typed as "representative-level diagnostic" and barred from gating.

## E. Closure diagnostics and kernel residuals

Kernel closure is evaluated in the quotient via residuals:
\[
\kappa_t^{(j)} := d(\bar K^{(j)}(q_t^{(j)}), q_t^{(j)}).
\]

Implementation note: GPU does not compute \(\bar K\) unless a GPU kernel implementation is provided and parity-checked. If GPU computes any proxy closure metric, it must be logged as a proxy and never used as the sole closure criterion without CPU verification.

## F. Action channel separation (Phases/Witness)

Supervisory predicates and Meta-Witness act on the action channel only. Formally, witness is an idempotent endomap \(W\) on action-intents \(A\):
\[
W\circ W = W,\quad \text{and } W \text{ composes with } A(\cdot)\text{, not with }K(\cdot).
\]

This means:
- GPU outputs may affect whether \(A(\cdot)\) is enabled (permission),
- but never define or modify \(K\) or the tower itself.

## G. Logging / justification chain

Every emitted action must carry a justification chain that includes:
- Graded observables (hashes or summaries per scale)
- Gate snapshots (Phase-6 open/closed, Phase-7 certificate, Phase-8 readiness)
- Capital kernel state and Meta-Witness refusal codes (Phase-9)

This appendix is satisfied when GPU acceleration can be toggled on/off without changing supervisory semantics (only performance), and when audits reproduce the same gate decisions under CPU fallback.

# Boundary-Stable Eigen-Events (Normative Definition)

> **Status:** Locked invariant  
> **Audience:** All agents touching Phase-07, Phase-04 gating, learner activation, or performance evaluation  
> **Purpose:** Prevent regression to frictionless (“before fees”) eigen-selection

This section defines **Boundary-Stable Eigen-Events**, the only admissible class of trajectories. All downstream components—Phase-07, Phase-04, learner activation, dashboards, reports—must conform to this invariant. Anything that looks like an edge without surviving boundary costs is a **false eigen-event** and must be rejected.

## 1. Motivation (non-negotiable)

The system must **never** treat apparent profitability in a frictionless projection as a valid eigen-event. Any regime, mode, or trajectory that fails after boundary costs are restored is **not admissible** for learning or optimisation. This document makes the definitions precise, authoritative, and executable.

## 2. Primitive data (fixed)

For each decision opportunity \(t\) in a finite rolling window \(W\) we observe:

- \(d_t \in \{-1,0,+1\}\): the intended direction
- \(r_t \in \mathbb{R}\): the realised mid-return over the commitment horizon
- \(m_t \in \{0,1\}\): activity mask (1 iff the trade or flip is actionable)
- \(c_t \ge 0\): boundary cost (fees + half-spread + ordering slippage proxy)

These quantities live on fixed tapes. The window \(W\) contains the latest \(n\) decisions or diagnostics; nothing in this definition ever mutates trading logic.

## 3. Gross, boundary, and net functionals

For a trajectory segment \(\pi(W)\) we define:

$$
E(\pi;W) := \sum_{t\in W} m_t \cdot d_t \cdot r_t \qquad\text{(gross edge)}
$$
$$
B(\pi;W) := \sum_{t\in W} m_t \cdot c_t \qquad\text{(boundary / friction)}
$$
$$
\Pi(\pi;W) := E(\pi;W) - B(\pi;W) \qquad\text{(net surplus)}
$$

These quantities are **not interchangeable**: \(E\) omits friction, \(B\) isolates it, and \(\Pi\) is the only admissible invariant. Phase-07, Phase-04, and any learner activation must evaluate \(\Pi\) rather than \(E\).

## 4. Asymmetry densities (support-separated, robust)

Let the **active index set** over a window \(W\) be

$$
W^{+} := \{\,t \in W \mid m_t = 1\,\},
$$

and only those active timesteps participate in asymmetry estimation. Inactivity is treated as **absence of support**, not as a zero-valued event.

Define robust asymmetry densities as

$$
\rho_A^{\text{gross}}(W) := \operatorname{median}_{t \in W^{+}}\big(d_t \cdot r_t\big),
$$
$$
\rho_A^{\text{net}}(W) := \operatorname{median}_{t \in W^{+}}\big(d_t \cdot r_t - c_t\big).
$$

These medians operate **only on supported events** and therefore measure directional edge **conditional on activity**, not diluted by inactivity.

### 4.1 Activity rate (support density)

Separately define the **activity rate**

$$
\alpha(W) := \frac{|W^{+}|}{|W|}.
$$

This quantity is tracked **independently** and must never be folded into \(\rho_A\). It answers a different question:

* \(\rho_A\): *“Is there directional surplus when the system acts?”*
* \(\alpha\): *“How often does the system act?”*

Phase-07 and Phase-04 may impose **minimum activity thresholds** (e.g., \(\alpha(W) \ge \alpha_{\min}\)) as gating conditions, but **activity never substitutes for edge**, and edge never substitutes for activity.

## 🔐 Normative clarifications (binding)

1. The binary mask \(m_t \in \{0,1\}\) is a **derived support carrier**, not a truth label or classifier.
2. Inactive timesteps **do not contribute zeros** to asymmetry medians.
3. Gross and net asymmetry densities are evaluated **only on supported events**.
4. Any implementation that computes medians over \(m_t \cdot x_t\) without defining this equivalence is **non-compliant** with this document.

## 🧠 Reviewer invariant (one line)

> **Asymmetry is measured conditional on action; activity and edge are orthogonal and never conflated.**

### Why this matters (informal, for humans)

This prevents a classic failure mode:

> “Looks profitable, but only because it almost never trades.”

By separating **support** from **surplus**, you ensure that:

* sparse regimes don’t fake stability,
* dense churn doesn’t fake edge,
* and Phase-07 remains a *true boundary-stable eigen detector*, not a volume meter.

## 5. Eigen-event classifications (authoritative)

### 5.1 False eigen-event (boundary-unstable)

$$
\boxed{
E(\pi;W) > 0
;\quad
\Pi(\pi;W) \le 0
}
$$

Equivalently:

$$
\rho_A^{\text{gross}}(W) > 0
;\quad
\rho_A^{\text{net}}(W) \le 0
$$

**Interpretation:** the trajectory only appears profitable in a frictionless projection. These events **must be excluded** from Phase-04, Phase-07, learners, and reports.

### 5.2 Boundary-stable eigen-event (candidate)

$$
\boxed{
\Pi(\pi;W) > 0
;\quad
\rho_A^{\text{net}}(W) > 0
}
$$

This is the **minimal admissibility** condition for training or optimisation.

### 5.3 Boundary-stable profit eigen-event (full DASHI sense)

A trajectory class \(\pi\) is a profit eigen-event iff there exists a family of windows \(\{W_k\}\) such that:

$$
\boxed{
\begin{aligned}
&\Pi(\pi;W_k) > 0 \\
&\rho_A^{\text{net}}(W_k) > 0 \\
&\text{the condition persists over a supermajority of }W_k \\
&\text{no VOID/PARADOX escalation dominates the class}
\end{aligned}
}
$$

Only such classes may unlock Phase-04, activate the learner, or be labelled “profitable.”

## 6. Relationship to Phase-07 and Phase-04 (binding)

* **Phase-07** performs the pointwise detection of boundary-stable eigen-events. The emitter writes:
  ```
  phase7_ready := (\rho_A^{\text{net}}(W) > 0)
  ```
* **Phase-04** enforces temporal persistence of this predicate before any training payload is allowed.
* **Training admissibility** is equivalent to the existence of a boundary-stable eigen-event under persistence.

Phase-07 and Phase-04 block **learning only**; the execution loop continues to emit ACT/HOLD decisions and trade on its existing permission logic even when `phase7_ready` stays false. Runtime controllers may enable the boundary-aware abstain gate (e.g., `run_trader.py --boundary-gate`) which compares the running edge estimate (`pred_edge`) to the same cost proxy Phase-07 uses, forcing HOLDS when the estimated surplus fails to cover the boundary cost and keeping execution aligned with the training gate.

If Phase-07 is missing, negative, or non-persistent, Phase-04 **must** remain closed.

## 7. Prohibited regressions (explicit)

1. Using gross \(\rho_A^{\text{gross}}\) or gross PnL to admit training.
2. Treating “made money before fees” as evidence of an eigen-event.
3. Allowing any learner, optimiser, or controller to run when Phase-07 is blocked.
4. Collapsing UNKNOWN (⊥) into FLAT (0) as a surrogate for boundary control.

Violations are formal regressions, not tuning choices.

## 8. One-line invariant (for reviewers)

> **All learning and optimisation operate only on boundary-stable eigen-events; frictionless eigen-events are explicitly inadmissible.**

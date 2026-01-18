# Boundary-Stable Eigen-Events (Normative Definition)

> **Status:** Locked invariant  
> **Audience:** All agents touching Phase-07, Phase-04 gating, learner activation, or performance evaluation  
> **Purpose:** Prevent regression to frictionless (“before fees”) eigen-selection

This section defines **Boundary-Stable Eigen-Events**, the only admissible class of trajectories. All downstream components—Phase-07, Phase-04, learner activation, dashboards, reports—must conform to this invariant. Anything that looks like an edge without surviving boundary costs is a **false eigen-event** and must be rejected.

## 1. Motivation (non-negotiable)

The system must **never** treat apparent profitability in a frictionless projection as a valid eigen-event. Any regime, mode, or trajectory that fails after boundary costs are restored is **not admissible** for learning or optimisation. This document makes the definitions precise, authoritative, and executable.

## 2. Primitive data (fixed)

For each decision opportunity \(t\) in a finite rolling window \(W\) we observe:

- \(d_t \in \{-1,0,+1\}\): intended direction (action-channel signal)
- \(x_t \in \mathbb{R}\): signed target exposure (action-channel state)
- \(m_t \in \{0,1\}\): activity mask, derived as \(m_t := \pi_{\text{supp}}(d_t)\)
- \(\Delta m_t := m_t - m_{t-1}\)
- \(\Delta x_t := x_t - x_{t-1}\)
- cost inputs \(h_t, f_t, s_t \ge 0\) (half-spread, fees, slippage proxy), with \(\kappa_t := h_t + f_t + s_t\)

These quantities live on fixed tapes. The window \(W\) contains the latest \(n\) decisions or diagnostics; nothing in this definition ever mutates trading logic. Returns \(r_t\) are used by the capital kernel, not by Phase-07.

## 3. Edge and boundary aggregates

For a trajectory segment \(\pi(W)\) we define:

$$
e_t := x_{t-1} \cdot \Delta m_t \qquad\text{(action-channel edge)}
$$
$$
c_t := \kappa_t \cdot m_t \cdot |\Delta x_t| \qquad\text{(boundary / friction)}
$$
$$
E(\pi;W) := \sum_{t\in W} e_t \qquad\text{(edge aggregate)}
$$
$$
B(\pi;W) := \sum_{t\in W} c_t \qquad\text{(boundary aggregate)}
$$

These quantities are **not interchangeable**: \(E\) tracks action-channel edge and \(B\) isolates boundary costs. Phase-07 evaluates their ratio; capital accounting uses return-minus-cost separately.

## 4. Asymmetry densities (support-separated, robust)

Let the **active index set** over a window \(W\) be

$$
W^{+} := \{\,t \in W \mid m_t = 1\,\},
$$

and only those active timesteps participate in asymmetry estimation. Inactivity is treated as **absence of support**, not as a zero-valued event.

Define the boundary asymmetry density as

$$
\rho_A(W) := \frac{E(\pi;W)}{B(\pi;W) + \epsilon}.
$$

This ratio is **action-channel only** and must never be confused with return-based PnL. It measures the edge of the action stream relative to boundary cost, not market movement.

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
2. Inactive timesteps **do not contribute** to edge or cost aggregates.
3. \(\rho_A\) is computed from \((x_{t-1}, \Delta m_t, \Delta x_t, \kappa_t)\) only; **returns do not enter Phase-07**.
4. Any implementation that collapses activity into edge (or vice versa) is **non-compliant** with this document.

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
\rho_A(W) \le \theta_A \text{ or fails robustness}
}
$$

**Interpretation:** the action stream shows edge, but it does not survive boundary costs, persistence, or small perturbations. These events **must be excluded** from Phase-04, Phase-07, learners, and reports.

### 5.2 Boundary-stable eigen-event (candidate)

$$
\boxed{
\Pi(\pi;W) > 0
;\quad
\rho_A(W) \ge \theta_A \text{ and robust}
}
$$

This is the **minimal admissibility** condition for training or optimisation.

### 5.3 Boundary-stable profit eigen-event (full DASHI sense)

A trajectory class \(\pi\) is a profit eigen-event iff there exists a family of windows \(\{W_k\}\) such that:

$$
\boxed{
\begin{aligned}
&E(\pi;W_k) > 0 \\
&\rho_A(W_k) \ge \theta_A \text{ and robust} \\
&\text{the condition persists over a supermajority of }W_k \\
&\text{no VOID/PARADOX escalation dominates the class}
\end{aligned}
}
$$

Only such classes may unlock Phase-04, activate the learner, or be labelled “profitable.”

## 6. Relationship to Phase-07 and Phase-04 (binding)

* **Phase-07** performs the pointwise detection of boundary-stable eigen-events. The emitter writes:
  ```
  phase7_ready := (\rho_A(W) \ge \theta_A \text{ and robust})
  ```
* **Phase-04** enforces temporal persistence of this predicate before any training payload is allowed.
* **Training admissibility** is equivalent to the existence of a boundary-stable eigen-event under persistence.

Phase-07 and Phase-04 block **learning only**; the execution loop continues to emit ACT/HOLD decisions and trade on its existing permission logic even when `phase7_ready` stays false. Runtime controllers may enable the boundary-aware abstain gate (e.g., `run_trader.py --boundary-gate`) which compares the running edge estimate (`pred_edge`) to the same cost proxy Phase-07 uses, forcing HOLDS when the estimated surplus fails to cover the boundary cost and keeping execution aligned with the training gate.

If Phase-07 is missing, negative, or non-persistent, Phase-04 **must** remain closed.

## 7. Prohibited regressions (explicit)

1. Using return-based PnL or frictionless projections to admit training.
2. Treating “made money before fees” as evidence of an eigen-event.
3. Allowing any learner, optimiser, or controller to run when Phase-07 is blocked.
4. Collapsing UNKNOWN (⊥) into FLAT (0) as a surrogate for boundary control.

Violations are formal regressions, not tuning choices.

## 8. One-line invariant (for reviewers)

> **All learning and optimisation operate only on boundary-stable eigen-events; frictionless eigen-events are explicitly inadmissible.**

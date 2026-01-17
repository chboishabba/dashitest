# Phase-9: Certified Economic Operator

Phase-9 elevates the system from “trading engine” to a continuously certified economic operator. Net surplus, risk, capital, and learning are coupled under explicit governance instead of ad-hoc tuning.

## Definition

Trade only when the action stream is net-extractable **and** the system can justify continuous existence: capital is a state variable, regimes are certified, learning is budgeted, and refusal paths are enforced without human intervention.

## Pillars and deliverables

- **Capital as state (Capital Ledger Kernel)**: track \(C_t\) with \(C_{t+1} = C_t + \Pi_t - \Lambda_t\), where \(\Pi_t\) is realized surplus and \(\Lambda_t\) is legitimacy tax (risk/drawdown/volatility/opportunity). If capital cannot be justified under stress, throttle or terminate exposure.
- **Multi-regime certification (Eigen-regime census)**: demonstrate profitability in ≥1 regime and preservation in all others. Hostile regimes collapse exposure to convex-only or zero; Phase-07 negative edge must be obeyed.
- **Capital-constrained learning**: learning updates are budgeted; if variance rises without surplus, learning halts. Learning drawdown monitor is tied to capital, not abstract loss.
- **Self-audit / refusal (Meta-Witness)**: independent process can freeze learning, force OBSERVE, or require re-certification when assumptions break, over-fitting is detected, or costs exceed structure.
- **Justification chain**: every action stores `regime → posture → actuator → cost model → expected surplus → realized surplus`; broken links gate future actions.

## Success criteria (Phase-9 entry)

1. No trade occurs without a stored justification chain.
2. Capital drawdown triggers structural response (throttle/exit), not parameter tweaks.
3. Learning auto-halts under uncertainty or variance without surplus.
4. Regime shifts reduce exposure automatically (obey Phase-07 negative edge).
5. System can operate unattended for weeks with continuous certification artifacts.

## First concrete step

Add an explicit capital state machine: log \(C_t\), enforce \(\Delta x_t \le f(C_t, \text{drawdown}, \text{Phase-07})\), and record capital justification per action. This forces Phase-9 framing before further feature work.

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

## Capital kernel equations (operational sketch)

- State: capital \(C_t>0\); exposure \(x_t\in[-x_{\max}, x_{\max}]\); mid \(m_t\); \(\Delta x_t = x_t - x_{t-1}\); return \(r_t = (m_t - m_{t-1})/m_{t-1}\).
- PnL: \(\Pi^{\text{m2m}}_t = C_{t-1} \cdot x_{t-1} \cdot r_t\).
- Friction: \(\Lambda^{\text{fric}}_t = C_{t-1} \cdot \kappa_t \cdot |\Delta x_t|\), with \(\kappa_t = \kappa_{\text{fee}} + \kappa_{\text{slip}} + \kappa_{\text{margin}}\).
- Risk/legitimacy tax: \(\Lambda^{\text{risk}}_t = C_{t-1}(\lambda_{\text{dd}}[\text{DD}_t]^+ + \lambda_{\sigma}\hat\sigma_t + \lambda_{\text{post}}\mathbf{1}_{p_t \neq +1})\).
- Surplus: \(\Pi_t = \Pi^{\text{m2m}}_t - \Lambda^{\text{fric}}_t\).
- Capital update: \(C_t = C_{t-1} + \Pi_t - \Lambda^{\text{risk}}_t\); clamp to \(C_{\min}\).
- Budgeted exposure clamp: \(B_t = \beta C_t\); enforce \(|x_t| \le \min(x_{\max}, B_t/(C_t(\kappa_t + \epsilon)))\). If costs dominate, exposure collapses automatically.
- Phase-07 framing: \(\rho_A^{\text{net}}(W) = \operatorname{median}_{t\in W^+}(x_{t-1}r_t - \kappa_t|\Delta x_t|)\); Phase-07 ready ⇔ \(\rho_A^{\text{net}}(W) > 0\) with persistence.

## Meta-Witness refusal conditions (binding)

- **R0 Missing authority**: phase6 gate closed/missing → OBSERVE, reason `no_phase6_authority`.
- **R1 Net surplus ≤ 0**: phase7_ready=false (`net_asymmetry_nonpositive`, etc.) → HOLD; if turnover continues, escalate to BAN.
- **R2 Sparse support**: \( |W^+|<N_{\min} \) or \(\alpha(W)<\alpha_{\min}\) → HOLD/OBSERVE (no unblock by silence).
- **R3 Capital drawdown breach**: drawdown > limit → BAN, freeze learning, require re-certification.
- **R4 Cost-dominated churn**: high turnover with \(\rho_A^{\text{net}}\le 0\) → BAN, flag churn failure.
- **R5 Policy inconsistency**: refusal active but action emitted → kill-switch (force BAN) and emit `SPECIAL_CODE=PARADOX`.

## Wiring into existing paths

- **Capital kernel**: place in `phase9/capital_kernel.py` (or `policy/`). Functions: `update_capital(C_prev, x_prev, r_t, dx_t, kappa_t, posture, dd, sigma) -> C_t`; `clamp_exposure(x_raw, C_t, kappa_t, dd, posture) -> x_clamped`; `ledger_row(...)`.
- **Meta-Witness**: `phase9/meta_witness.py` with `evaluate(state) -> directives` (`force_hold`, `force_posture`, `freeze_learning`, `max_abs_exposure`, `reason`).
- **Interception point**: run Meta-Witness + capital clamp right before actions are emitted:
  - Live: `scripts/stream_daemon.py` decision emission (`live:decision`).
  - Loop: `engine/loop.py` before execution/Phase-5 logging.
- **Log fields to add**: `capital_C`, `capital_dd`, `kappa_t`, `mw_reason`, `mw_refusal_level`, `mw_forced_hold/ban`, `mw_max_exposure`, plus justification chain `regime → posture → actuator → cost model → expected surplus → realized surplus`.

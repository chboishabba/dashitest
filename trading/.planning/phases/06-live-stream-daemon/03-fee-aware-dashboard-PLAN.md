## Phase 06.03 - Fee-Aware Live Dashboard + Decision Cost Gate (Plan)

### Objective
Make live decision visualization economically honest and interpretable by:
- Plotting decision intents as stepwise targets (not executions)
- Surfacing delta-exposure and fee pressure explicitly
- Ensuring fee modeling uses notional deltas (price * exposure change)

### Scope Options (pick one to implement first)
1) **Dashboard-only corrections**
   - Step plot for signed target exposure
   - Delta exposure panel (impulses/bars)
   - Fee model uses notional (`abs(delta) * price * fee_rate`)
   - Optional gross vs net PnL overlay in dashboard
2) **Decision cost gate (consumer)**
   - Insert a cost-aware clamp between decision emission and sinks
   - Estimate cost from delta exposure + last price
   - Hold or damp actionability when expected edge < cost
3) **Both (dashboard + gate)**

### Constraints
- Keep triadic strategy semantics unchanged.
- No learning feedback loops; cost model is a downstream consumer.
- Avoid GUI regressions in `pyqtgraph` dashboard mode.

### Success Criteria
- Live dashboard clearly separates intent vs execution impacts.
- Fees visibly scale with churn (delta exposure) on notional basis.
- Optional cost gate reduces micro-churn without touching strategy logic.

### Assumptions
- `target_exposure` is a fraction of portfolio notional.
- Last close price in OHLC stream is acceptable for fee estimation.
- Fee/slippage rates are configured via existing CLI flags.

### Open Questions
- Which scope option should land first (1, 2, or 3)?
- Should fee pressure be plotted as a separate series or stacked under PnL?
- Do we want a hard block or soft decay when cost > benefit in the gate?

### Plan
1) Confirm scope option and decision-gate policy (hard/soft/budgeted).
2) Update docs to reflect intent (dashboard semantics + fee model).
3) Update TODO/plan items to match the chosen scope.
4) Implement code changes + targeted tests/verification.
5) Update changelog if behavior changes are user-visible.

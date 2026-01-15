# Phase-6 Capital Controls

Phase-6 does not relearn direction or size. It enforces **capital discipline** once Phase-4 has supplied size weights—so the control logic generally consumes execution logs (from Phase-5) and applies exposure caps, drawdown-aware damping, or hazard-aware size shrinkage before the next trading session.

## Objectives

1. **Protect capital** by limiting exposure per ontology and conditioning size on hazard proximity.
2. **Keep size decisions conservative** when drawdowns, volatility spikes, or hazard counts increase.
3. **Remain downstream of Phase-5**: capital controls only read proposals/emulation logs, never modify `size_pred` before Phase-4 has locked it.

## Inputs

* Phase-5 execution log (JSONL): includes `i`, `action`, `size_pred`, `realized_pnl`, `slippage_cost`, `execution_cost`.
* Proposal log: to recompute on-the-fly exposures if needed.
* Tape-level stats: `hazard`, `veto`, drawdown metrics, realized vs. nominal PnL per bin.

## Controls to apply

### Exposure caps

* `max_exposure[T]`, `max_exposure[R]`: limit the total notional (size × price) per ontology over each window (rolling 1h / 4h).
* Continue rejecting proposals that would exceed the cap even if Phase-4 suggests a large size bin.

### Drawdown-aware damping

* Track worst drawdown in the last `N` bars. When drawdown exceeds threshold, scale `size_pred` probability mass toward smaller bins via a map (e.g., multiply Phase-4 weights by `[(1 − dd_scale), dd_scale]`).

### Hazard proximity

* Monitor consecutive hazard vetoes or `H` density spikes.
* When hazard rate > `hazard_limit`, temporarily clamp Phase-4 size choices to `min(size_pred, 0.5)` while still logging the clamp for audit.

## Phase-6 iteration plan

1. **Consumption**: read Phase-5 JSONL, summarize realized exposure per ontology/size bin, and compute moving average drawdown/hazard.
2. **Decision**: determine whether upcoming proposals should be throttled (cap reached) or damped (drawdown/hazard triggered).
3. **Enforcement**: either rewrite a `phase6_capital_log` describing adjustments or write a decorator that `phase5_execution_simulator` and production runners can apply before the next trade.

## Outputs

* `logs/phase6/capital_controls_<stamp>.jsonl`: each entry records the incoming proposal, the capital condition triggered, and the final acceptance/size modification (if any).
* Summary table: exposures, drawdown, hazard ratio, gating status per tape.

## Next steps

* Align this logic with Phase-5 so the control decisions are auditable before any live deployment.
* Build simple simulators that feed synthetic drawdown/hazard sequences to ensure the controls behave as expected.

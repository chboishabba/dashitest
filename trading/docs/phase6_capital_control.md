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

### Friction guard (0.5 bps)

* The Phase-5 friction sweep proved only the 0.5 bps slip track stays profitable; Phase-6 therefore gates exposure on that exact regime.
* Run `scripts/phase6_exposure_control.py --slip-threshold 0.5` to collapse every Phase-5 log into `logs/phase6/capital_controls_<stamp>.jsonl`, marking slip levels >0.5 bps as `clamped` and the 0.5 bps run as `allowed`.
* Capital controls only propagate exposures and drawdown stats from the allowed slip track; anything else is treated as an automatic clamp (reduce size or shift to `OBSERVE/UNWIND`).

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
* Live gate wiring: `phase6/Phase6ExposureGate` scans the latest `capital_controls_*.jsonl` and reports `gate_open` only when an `allowed=true` slip entry exists. If the log is missing or no slip is approved, live posture defaults to `OBSERVE`.

## Live gate invariants (current state)

- A closed gate forces `posture_observe`, which pins `direction=0`, `target_exposure=0`, `hold=true`, and `actionability=0`; Phase-07 is empty because it is conditional on actions, not on market movement alone.
- The only wake-up condition is a Phase-6 capital-control log with at least one `allowed=true` entry. Missing logs or logs without `allowed=true` keep the system frozen by design.
- Recommended wake-up proof: write a small slip approval (e.g., `{"timestamp": "<iso>", "allowed": true, "reason": "manual_test"}`) into `logs/phase6/capital_controls_<stamp>.jsonl`, restart `scripts/stream_daemon_live.py`, and verify `gate_open=true`, posture leaves OBSERVE, and Phase-07 starts accumulating actions. If the gate stays closed, investigate wiring before touching asymmetry logic.

## Next steps

* Align this logic with Phase-5 and the runner’s gating hook (`phase6/Phase6ExposureGate`) so only the allowed 0.5 bps slip track ever escapes to the executor.
* Build simple simulators that feed synthetic drawdown/hazard sequences to ensure the controls behave as expected.

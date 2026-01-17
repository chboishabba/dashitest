# Phase-4.1 → Phase-5 → Asymmetry Plan

**Objective:** Follow the documented fork by doing **A → B → C** in sequence so this conservative learner only acts when evidence exists. This plan captures the deliverables, inputs, decisions, and exit criteria for each milestone.

## Option A: Phase-4.1 on ES (size only)

* **Goal:** Learn stable T/R size weights after the Phase-4 density gate opened for ES (gate open twice with no debounce fail).
* **Inputs:** ES proposals + prices (same as gate targets), existing Phase-4 size trainer, priors (λ = 0.25), and the pinned H (flat/hysteresis anchor) configuration.
* **Steps:**
  1. Run `scripts/train_size_per_ontology.py` with `--target ES` and `--mode size` (no directional learning).
  2. Freeze the resulting size weights, record schema, checksum, and metadata (timestamped log).
  3. Capture diagnostics: per-bin PnL, size gradient, `phase4_gate_status.md` snapshot.
* **Exit criteria:** locked weights produce repeatable `logs/phase4/size_weights_es.csv`, gate status remains OPEN, and the asset is ready for Phase-5 friction inputs.
* **Decision point:** If size training collapses (zero weights, extreme variance), document that and do not proceed to Phase-5.

## Option B: Phase-5 friction sweep (reality check)

* **Goal:** Confirm whether the ES size edge survives market taxes (slippage, fees) before calling it profitable.
* **Inputs:** Locked ES size weights, proposals annotated with action/loss info, Phase-5 simulator parameters.
* **Steps:**
  1. Run `scripts/phase5_execution_simulator.py` with slippage levels (e.g., 0.5, 1.0, 1.5, 2.0 bps) and standard fee model.
  2. Log `edge_decay.csv` showing size vs PnL vs slippage, tail-loss comparisons vs FLAT, and action flip frequency.
  3. Evaluate whether any friction level keeps edge positive after costs.
* **Exit criteria:** either (a) at least one friction level retains edge (retain for Phase-6) or (b) all frictioned runs lose money → signal to pivot to asymmetry sensing instead of forcing trade.
* **Decision point:** Document the highest slippage level that stays positive; propagate that param to any future Phase-6 controls.

## Option C: Asymmetry sensing (profitable M9 unlock)

* **Goal:** Build the first asymmetry sensor (pressure field/influence tensor) so certain M9 regimes become actionable.
* **Candidate sensors:** funding–price divergence persistence, forced liquidation proxy, cross-asset influence (ES → BTC) lag map, or options convexity mispricing. Pick one with available data.
* **Steps:**
  1. Select sensor and define its input stream, transformation, and threshold (e.g., normalized funding spread + divergence persistence > θ).
  2. Encode the sensor into the influence tensor (per-symbol heads) and feed a triadic projection (+1/−1/0/⊥) into `signals/triadic_ops` so permission logic can escalate.
  3. Log sensor activations + downstream approvals (`logs/asymmetry_sensor.jsonl`) and prove it fires before obvious moves.
* **Exit criteria:** sensor regularly fires on historical periods where asymmetric behavior later confirms the thesis (e.g., persist funding/price divergence preceding forced flows), and the controller can conditionally treat that state as eligible for convex posture.
* **Decision point:** If sensor never triggers, record why and either expand to a new instrument or accept the asymmetry cannot be observed with the current data.

## Sequence & assumptions

1. **A before B:** Without Phase-4 size weights locked, Phase-5 lacks calibrated inputs.
2. **B before C:** Friction kills the naive edge, so we only invest in asymmetry sensing when needed.
3. **Assumptions:** ES tape remains stable, BTC data stays blocked, GPU logic and triadic ops continue to behave, and gating logic refuses to open on insufficient density.
4. **Artifacts:** update `docs/phase4_density_monitor.md`, `docs/phase4_gate_status.md`, and `docs/m4_m9_status.md` with run results after each milestone; capture relevant logs in `logs/phase4/` and `logs/phase5/` as timestamped files.

## Risks & mitigations

* **Risk:** ES friction tests show zero edge even after size tuning. *Mitigation:* pivot immediately to asymmetry sensor hypothesis (Option C) and capture data for Aberdeen regression tests.
* **Risk:** BTC remains blocked, tempting shortcuts. *Mitigation:* keep `docs/m4_m9_status.md` updated and record any data-substrate improvements attempted.
* **Risk:** running Phase-5 before weights freeze wastes compute. *Mitigation:* verify weight file checksum and gate openness before launching the simulator.

If you want me to update TODOs or assist in running any of these scripts, say which option should be executed first—A, B, or C—and I’ll proceed accordingly. Otherwise, I can help turn this plan into tracked work items now.

## Tracking & checkpoints

| Task | Owner | Due | Artifact |
| --- | --- | --- | --- |
| Option A weight lock + diagnostics | Dev | 2026-01-15 | `logs/esnq/ES_5m/phase4_size_run.json`, `logs/esnq/ES_5m/weights_size_run.json`, `phase4_gate_status.md` snapshot |
| Option B friction sweep | Dev | 2026-01-15 | `logs/phase5/es/phase5_execution_*.jsonl`, slip vs PnL summary table |
| Option C influence sensor | Dev | 2026-01-15 | `logs/asymmetry/influence_tensor_*.jsonl`, `scripts/asymmetry_influence_sensor.py` |
| BTC Path commitment | Dev | 2026-01-17 | `docs/btc_phase4_path.md` |
| Ternary invariants memo | Dev | 2026-01-17 | `docs/ternary_invariants.md` |
| Next-phase checklist review | Dev | 2026-01-20 | updated `docs/m4_m9_status.md`, TODOs |

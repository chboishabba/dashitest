---
phase: 05-esnq-ingestion
type: execute
domain: trading
---

<objective>
Document the ES/NQ ingestion contract so the Phase-4 gate finally sees real density, then add tooling that proves the contract is honored before Phase-4.1 is ever unblocked.
Purpose: unblock the data dependency without touching gate math by making ingestion visible and verifiable.
Output: refreshed ingestion doc, updated TODO entry, and a small `scripts/check_esnq_ingestion.py` utility that surveys the proposal/prices contract plus monitor log.
</objective>

<execution_context>
~/.codex/skills/get-shit-done/workflows/execute-phase.md
./summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@docs/es_nq_ingestion_plan.md
@docs/phase4_density_monitor.md
@configs/phase4_monitor_targets.json
@scripts/phase4_density_monitor.py
@logs/phase4/density_monitor.log
</context>

<tasks>

<task type="auto">
  <name>Task 1: Embed ingestion contract + tooling notes in docs</name>
  <files>docs/es_nq_ingestion_plan.md</files>
  <action>Expand the ingestion plan to list the concrete contract checks (proposal schema, monitor-target wiring, blocking reason logging) and describe how the new helper script verifies those checks.</action>
  <verify>Read `docs/es_nq_ingestion_plan.md` and confirm the ES/NQ contract now references the helper script and the Phase-4 gate’s blocking reasons.</verify>
  <done>The doc lays out the ingestion invariants and points operators to the helper script for diagnostics.</done>
</task>

<task type="auto">
  <name>Task 2: Record the ingestion checklist in TODO</name>
  <files>TODO.md</files>
  <action>Add entries that mark Phase-4.1 as data-blocked until ES/NQ proposals + prices are confirmed, and call out the new check script as part of that gating story.</action>
  <verify>TODO includes tasks referencing ingestion verification and links to the helper script or doc.</verify>
  <done>TODO clearly states the remaining data work and where to find the verification helper.</done>
</task>

<task type="auto">
  <name>Task 3: Ship the ES/NQ ingestion verifier</name>
  <files>scripts/check_esnq_ingestion.py</files>
  <action>Implement a CLI that reads `configs/phase4_monitor_targets.json`, verifies each proposal/prices path exists and contains the required columns, and checks the monitor log for recent entries per target (reporting the latest blocking reason).</action>
  <verify>Run `python scripts/check_esnq_ingestion.py --config configs/phase4_monitor_targets.json` (using the BTC target today) and confirm it reports the contract status without crashing.</verify>
  <done>The helper script exists and can be run daily to prove ES/NQ data flows through the existing monitor contract.</done>
</task>

</tasks>

<verification>
Before declaring phase complete:
- [ ] `python scripts/check_esnq_ingestion.py --config configs/phase4_monitor_targets.json` runs and reports the gate-state for each target.
- [ ] `docs/es_nq_ingestion_plan.md` mentions the helper script and the ingestion invariants, plus how blocking reasons tie back to the gate.
- [ ] `TODO.md` reflects the ingestion work and the data lock.
</verification>

<success_criteria>
- Plan tasks done and verified
- Ingestion guardrail clearly recorded
- Helper script proves the contract end-to-end
  </success_criteria>

<output>
After completion, create `.planning/phases/05-esnq-ingestion/01-ingest-contract-SUMMARY.md`:

# Phase 05 Plan 01: ES/NQ ingestion contract

**Documented the ingestion contract and shipped a quick CLI that proves the Phase-4 gate sees the right density inputs.**

## Accomplishments

- [List on completion]

## Files Created/Modified

- `docs/es_nq_ingestion_plan.md` - ...
- `TODO.md` - ...
- `scripts/check_esnq_ingestion.py` - ...

## Decisions Made

[Record decisions or "None"]

## Issues Encountered

[Record issues or "None"]

## Next Step

[Ready / mention dependency]
</output>

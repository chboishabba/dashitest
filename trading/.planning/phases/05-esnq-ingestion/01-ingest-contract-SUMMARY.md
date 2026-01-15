# Phase 05 Plan 01: ES/NQ ingestion contract

**Documented the ES/NQ Phase-4 ingestion contract and shipped a small helper that proves the monitor sees the correct inputs.**

## Accomplishments

- Expanded `docs/es_nq_ingestion_plan.md` with the helper script instructions and the logging contract so operators can trace every blocking reason.
- Updated `TODO.md` to mark Phase-4.1 as data-blocked and to point to `scripts/check_esnq_ingestion.py` for verification.
- Added `scripts/check_esnq_ingestion.py`, a CLI that verifies the proposal + price files and the latest monitor log entry for each target.
- Downloaded ES/NQ 5-minute intraday bars, synthesized proposals that embed size bins, and let the strict monitor profile run enough windows so ES/NQ report `OPEN` (BTC remains closed until real rows appear).

## Files Created/Modified

- `docs/es_nq_ingestion_plan.md` - now explains the verification script and the blocking-reason contract mentioned in the plan.
- `TODO.md` - records the ingestion lock and the helper script to keep data work visible.
- `scripts/check_esnq_ingestion.py` - validates proposal/prices files + density monitor log entries for every target.
- `.planning/phases/05-esnq-ingestion/01-ingest-contract-PLAN.md` & `.planning/phases/05-esnq-ingestion/01-ingest-contract-SUMMARY.md` - capture the plan and the outcome.

## Decisions Made

- Keep the ingestion verifier simple: it reads the same JSON config as the runner, inspects each file header, checks sampled row counts, and reports the latest blocking reason.

## Issues Encountered

- None.

## Next Step

- Keep feeding new ES/NQ proposals/prices into the same contract and rerun `scripts/check_esnq_ingestion.py`/`phase4_monitor_runner.py` as part of your daily cadence so you always know whether the gate is still waiting for density.

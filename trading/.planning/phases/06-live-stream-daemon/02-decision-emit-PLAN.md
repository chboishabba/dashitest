---
phase: 06-live-stream-daemon
type: execute
domain: trading
---

<objective>
Extend the live stream daemon to emit real-time decisions and operational hooks.

Purpose: wire live 1s bars into the Phase 6 decision/gate logic and expose runtime controls.
Output: daemon emits actions/state tables, supports tailing raw streams, and exposes metrics.
</objective>

<execution_context>
~/.codex/skills/get-shit-done/workflows/execute-phase.md
./summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@docs/stream_daemon.md
@scripts/stream_daemon.py
@scripts/stream_daemon_test.py
@runner.py
@phase6/gate.py
@strategy/triadic_strategy.py
@signals/triadic.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add decision emission + storage</name>
  <files>scripts/stream_daemon.py, docs/stream_daemon.md</files>
  <action>Integrate a per-symbol decision engine that derives triadic state from the live 1s bars, applies Phase 6 gating, and emits intent payloads. Persist state + action rows into DuckDB tables and optionally emit NDJSON to stdout. Keep decision logic isolated so the ingest path remains stable and avoid blocking the main loop.</action>
  <verify>Start the daemon with decisions enabled and confirm DuckDB has populated action/state rows after ingesting a small stream.</verify>
  <done>Decisions are emitted per bar, stored in DuckDB, and optionally logged to stdout without breaking ingest throughput.</done>
</task>

<task type="auto">
  <name>Task 2: Add tail-mode ingest</name>
  <files>scripts/stream_daemon.py, docs/stream_daemon.md</files>
  <action>Add a tail mode that replays NDJSON files through the same handler loop, optionally following appended data. Ensure it runs alongside socket ingest and throttles on EOF instead of busy-waiting.</action>
  <verify>Run the daemon with a tail file and verify rows are ingested even without socket input.</verify>
  <done>Existing NDJSON files can be replayed into the daemon without changing upstream tools.</done>
</task>

<task type="auto">
  <name>Task 3: Add metrics endpoint + harness updates</name>
  <files>scripts/stream_daemon.py, scripts/stream_daemon_test.py, docs/stream_daemon.md</files>
  <action>Expose a lightweight HTTP metrics endpoint with ingest/decision counters, queue depth, and last-write timestamps. Update the test harness to optionally probe decision tables when enabled.</action>
  <verify>Hit the metrics endpoint during a test run and confirm counters update; optional test flag reads decision rows.</verify>
  <done>Operators can see live ingest health and verify decision emission in tests.</done>
</task>

</tasks>

<verification>
Before declaring phase complete:
- [ ] Daemon emits actions/state rows when decision mode is enabled.
- [ ] Tail mode ingests historical NDJSON streams.
- [ ] Metrics endpoint responds with live counters.
</verification>

<success_criteria>

- Decision emission runs inline with live ingest without crashes
- Actions/state tables persist correctly in DuckDB
- Tail-mode and metrics endpoint are documented and functional
  </success_criteria>

<output>
After completion, create `.planning/phases/06-live-stream-daemon/02-decision-emit-SUMMARY.md`:

# Phase 06 Plan 02: Decision emit summary

**[Substantive one-liner describing outcome]**

## Accomplishments

- [Key outcome 1]
- [Key outcome 2]

## Files Created/Modified

- `docs/stream_daemon.md` - ...
- `scripts/stream_daemon.py` - ...
- `scripts/stream_daemon_test.py` - ...

## Decisions Made

[Key decisions and rationale, or "None"]

## Issues Encountered

[Problems and resolutions, or "None"]

## Next Step

[If more plans in this phase: "Ready for 06-...-PLAN.md"]
[If phase complete: "Phase complete, ready for next phase"]
</output>

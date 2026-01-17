---
phase: 07-live-density
type: execute
domain: trading
---

<objective>
Feed Phase-4 density gates from live stream data without relaxing strictness.

Purpose: materialize live density inputs so Phase-4 can open on real-time evidence.
Output: live density feeder, observability logs, and a replay-safe materialization path.
</objective>

<execution_context>
~/.codex/skills/get-shit-done/workflows/execute-phase.md
./summary.md
~/.codex/skills/get-shit-done/references/checkpoints.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@scripts/phase4_density_monitor.py
@configs/phase4_monitor_profiles.json
@docs/hierarchical_summarisation.md
@docs/stream_daemon.md
</context>

<tasks>

<task type="checkpoint:decision" gate="blocking">
  <decision>Choose the live density materialization path</decision>
  <context>We need live density inputs to match offline Phase-4 strictness without logic drift or manual overrides.</context>
  <options>
    <option id="memmap-parity">
      <name>Memmap parity with offline tooling</name>
      <pros>Maximum parity with existing Phase-4 tooling; avoids logic drift.</pros>
      <cons>Requires memmap management and more file IO.</cons>
    </option>
    <option id="duckdb-only">
      <name>DuckDB-only rolling tables</name>
      <pros>Simple operational surface; fewer formats to manage.</pros>
      <cons>Risk of divergence from existing memmap logic.</cons>
    </option>
  </options>
  <resume-signal>Select: memmap-parity or duckdb-only</resume-signal>
</task>

<task type="auto">
  <name>Task 1: Implement live density feeder</name>
  <files>scripts/live_density_feeder.py, docs/phase7_live_density.md</files>
  <action>Consume the live 1s bar stream (or summarised tables) and emit rolling density inputs aligned to Phase-4 schema. Use the chosen materialization path, and avoid shortcuts that would weaken strict checks.</action>
  <verify>Run the feeder on a short live sample and confirm density artifacts are created.</verify>
  <done>Phase-4 tooling can read the live density artifacts without code changes.</done>
</task>

<task type="auto">
  <name>Task 2: Add observability for density failures</name>
  <files>scripts/live_density_feeder.py, docs/phase7_live_density.md</files>
  <action>Log per-symbol failure reasons (balance, persistence, effect size) to a structured JSONL so it is always clear why the gate remains closed.</action>
  <verify>Inspect the log during a short run and see failure reasons emitted.</verify>
  <done>Operators can attribute closed gates to explicit failed checks.</done>
</task>

</tasks>

<verification>
Before declaring phase complete:
- [ ] Live density artifacts are created on a short run.
- [ ] Phase-4 strict profile reads the live artifacts without changes.
- [ ] Failure logs capture explicit reasons for closed gates.
</verification>

<success_criteria>

- Phase-4 strictness remains unchanged
- Live data can open the Phase-4 gate when criteria are met
- Observability shows why the gate is closed when it is
  </success_criteria>

<output>
After completion, create `.planning/phases/07-live-density/01-live-density-SUMMARY.md`.
</output>

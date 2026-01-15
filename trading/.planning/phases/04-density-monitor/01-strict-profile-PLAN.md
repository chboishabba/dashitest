---
phase: 04-density-monitor
type: execute
domain: trading
---

<objective>
Describe how the Phase-4 gate now has a canonical `strict` profile and an automated daily status log, then deliver the new script that materializes `phase4_gate_status.md` so operators know *why* the gate is still closed.
Purpose: make the gate behavior reproducible, communicative, and easy to review before Phase-4.1 training.
Output: updated docs, a TODO entry, the status-generation script, and a fresh `phase4_gate_status.md` snapshot.
</objective>

<execution_context>
~/.codex/skills/get-shit-done/workflows/execute-phase.md
./summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@docs/phase4_density_monitor.md
@scripts/phase4_density_monitor.py
@logs/phase4/density_monitor.log
</context>

<tasks>

<task type="auto">
  <name>Task 1: Document the gate status summary workflow</name>
  <files>docs/phase4_density_monitor.md, TODO.md</files>
  <action>Explain how `configs/phase4_monitor_profiles.json` carries the `strict` profile, describe the new `phase4_gate_status.md` and how to regenerate it, and add a TODO entry so operators remember to capture the log each day.</action>
  <verify>Read `docs/phase4_density_monitor.md` and `TODO.md` to confirm the new sections mention the status file and script.</verify>
  <done>Docs mention `phase4_gate_status.md`, the script, and the strict profile; TODO references the status log task.</done>
</task>

<task type="auto">
  <name>Task 2: Ship the Phase-4 gate status generator</name>
  <files>scripts/phase4_gate_status.py, phase4_gate_status.md</files>
  <action>Implement a script that reads `logs/phase4/density_monitor.log`, selects the latest entry per date/target, and writes a Markdown summary; run it locally to emit `phase4_gate_status.md` for the current data.</action>
  <verify>Run the new script to regenerate `phase4_gate_status.md` and ensure it contains the latest blocking reasons.</verify>
  <done>The repo contains a working script and an up-to-date `phase4_gate_status.md` describing the gate status.</done>
</task>

</tasks>

<verification>
Before declaring phase complete:
- [ ] `python scripts/phase4_gate_status.py` runs without errors and updates `phase4_gate_status.md`
- [ ] `docs/phase4_density_monitor.md` references the new status log workflow
- [ ] `TODO.md` notes the maintenance task for the gate status log
</verification>

<success_criteria>
- All tasks completed
- Script output is human-readable and mentions blocking reasons
- Documentation and TODO reflect the new workflow
  </success_criteria>

<output>
After completion, create `.planning/phases/04-density-monitor/01-strict-profile-SUMMARY.md`:

# Phase 04 Plan 01: Density gate status summary

**Ensured Phase-4 gate transparency by documenting the strict profile and generating a daily status Markdown file.**

## Accomplishments

- Added prose describing the strict profile and gate status log in the density monitor doc
- Introduced `scripts/phase4_gate_status.py` and produced a snapshot `phase4_gate_status.md`
- Logged the gate-status maintenance step in `TODO.md`

## Files Created/Modified

- `docs/phase4_density_monitor.md` - Mentioned the strict profile and status log
- `TODO.md` - Added the daily status generation reminder
- `scripts/phase4_gate_status.py` - Computes the Markdown summary
- `phase4_gate_status.md` - Records the latest gate state

## Decisions Made

- Keep the generator script simple: one run scans the log and writes the latest per-date/target reasoning

## Issues Encountered

- None

## Next Step

Ready to hand off to the monitor operator for daily refresh.
</output>

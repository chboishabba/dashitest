---
phase: 06-live-stream-daemon
type: execute
domain: trading
---

<objective>
Define and implement the always-on stream daemon contract so arbitrary-rate inputs are ingested, summarised, and persisted with minimal latency.

Purpose: move from file-based chunk ingestion to a continuous stream processor that can accept any input rate and push results to DuckDB immediately.
Output: stream contract doc, daemon CLI implementation, and a reproducible test harness.
</objective>

<execution_context>
~/.codex/skills/get-shit-done/workflows/execute-phase.md
./summary.md
~/.codex/skills/get-shit-done/references/checkpoints.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@data_downloader.py
@tools/ingest_archives_to_duckdb.py
@trading/summarisation/summariser.py
@trading/summarisation/storage.py
</context>

<tasks>

<task type="checkpoint:decision" gate="blocking">
  <decision>Choose the daemon input contract</decision>
  <context>We need one canonical format/transport to support high-rate inputs while keeping parsing and backpressure manageable.</context>
  <options>
    <option id="ndjson-stdin">
      <name>NDJSON over stdin</name>
      <pros>Simple streaming, low overhead, easy to feed from any producer, works with pipes and sockets.</pros>
      <cons>Requires explicit schema versioning and field validation per line.</cons>
    </option>
    <option id="tcp-ndjson">
      <name>TCP socket + NDJSON</name>
      <pros>Daemon handles connections directly; producers push without spawning new processes.</pros>
      <cons>Requires socket management, reconnect logic, and service discovery.</cons>
    </option>
  </options>
  <resume-signal>Select: ndjson-stdin or tcp-ndjson</resume-signal>
</task>

<task type="auto">
  <name>Task 1: Document the stream daemon contract</name>
  <files>docs/stream_daemon.md</files>
  <action>Specify input message schema(s), required fields, time units, symbol labeling, and how trade vs ohlc messages are handled. Include output destinations (DuckDB tables, summaries), batching rules, error handling, and backpressure behavior. Avoid mixing raw ticks with summary outputs; emphasize quotient/summariser boundaries.</action>
  <verify>Read docs/stream_daemon.md and confirm schema + output behavior are explicit, with examples.</verify>
  <done>The contract doc defines the live stream format, expected outputs, and failure modes.</done>
</task>

<task type="auto">
  <name>Task 2: Implement the always-on daemon CLI</name>
  <files>scripts/stream_daemon.py, tools/ingest_archives_to_duckdb.py, trading/summarisation/summariser.py</files>
  <action>Build a CLI that consumes the chosen stream format, maintains per-symbol buffers, aggregates trades into 1s bars, writes to DuckDB via ingest_dataframe, and (optionally) calls the summariser per window. Include a rate-safe flush loop, signal handling, and minimal logging. Keep raw ingestion and summariser output separated to avoid reprocessing raw ticks.</action>
  <verify>Run the CLI with a sample input file and confirm DuckDB tables update without errors.</verify>
  <done>Daemon runs continuously, accepts arbitrary-rate inputs, and writes live bars + summaries to DuckDB.</done>
</task>

<task type="auto">
  <name>Task 3: Add a reproducible test harness</name>
  <files>scripts/stream_daemon_test.py</files>
  <action>Create a small generator that emits synthetic trade and ohlc messages at variable rates to validate throughput and ordering. Provide a single command that can be run to populate DuckDB and print row counts.</action>
  <verify>Run the test harness and see DuckDB row counts increase; no crashes under burst load.</verify>
  <done>One command can simulate high-rate inputs and prove ingest correctness.</done>
</task>

</tasks>

<verification>
Before declaring phase complete:
- [ ] docs/stream_daemon.md describes input schema, outputs, and backpressure.
- [ ] Daemon CLI ingests a sample stream and writes to DuckDB without crashes.
- [ ] Test harness produces rows and verifies counts in DuckDB.
</verification>

<success_criteria>

- Stream contract finalized and documented
- Daemon ingests arbitrary-rate inputs and persists results
- Test harness validates throughput and correctness
  </success_criteria>

<output>
After completion, create `.planning/phases/06-live-stream-daemon/01-daemon-contract-SUMMARY.md`:

# Phase 06 Plan 01: Live stream daemon summary

**Shipped a stream daemon contract + CLI that ingests arbitrary-rate inputs into DuckDB with optional summarisation.**

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

[Phase complete or next plan]
</output>

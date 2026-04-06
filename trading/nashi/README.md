# Nashi

`nashi/` is the new trader surface for a stricter Agda-formalism-based runtime.

It is intentionally separate from the current `engine/loop.py` path. The goal is
to reuse useful existing signals and market plumbing, while replacing heuristic
execution permissions with explicit admissibility contracts.

## Design rules

- Start from the Agda execution contract, not from the legacy fill loop.
- Treat refusal and severity as first-class runtime data.
- Allow learning components only behind the admissibility gate.
- Keep bridge code thin and observable so mismatches against `../../dashi_agda`
  are easy to audit.

## Current contents

- `contracts.py`: runtime mirror of the five-clause execution contract.
- `schema.py`: canonical closure embedding axes, arrow profiles, and witness-class names.
- `bridge_agda.py`: thin bridge to the current Agda/DASL-backed source semantics.
- `severity.py`: monotone refusal/severity handling.
- `state.py`: small policy state surface.
- `policy.py`: contract-first policy runtime.
- `phase9.py`: capital kernel, meta-witness directives, and justification-chain builder.
- `bridges.py`: basic adapters for embedding- and signal-based experiments.
- `proposals.py`: bounded candidate-generation layer over the canonical embedding.
- `family_memory.py`: lightweight runtime memory for recent boundary/family context.
- `family_certification.py`: post-step family witness classifier and preservation-vs-trade certification.
- `runtime.py`: observation adapter/runner that emits repo-compatible intents and per-step logs.
- `telemetry.py`: CSV + NDJSON + DuckDB writer for the existing dashboards.

## Execution pricing

- If input bars include `bid` and `ask`, `nashi` executes against those quotes.
- If quotes are absent, `nashi` synthesizes a full spread from `max(default_spread_bps, 0.1 * (high - low))` around `close`.
- The Agda-side histogram/tail machinery is relevant to observable and family certification, not to the execution fill primitive itself.

## Family certification

- `nashi` now carries a rolling post-step family witness over recent admissible decisions.
- The exported family class vocabulary matches the Agda constructors: `InteriorFamily`, `ArrowLadderFamily`, `SingleArrowBreakFamily`, and `MDLTailBoundaryFamily`.
- Spread-aware hostile windows are separated from structural failures:
  - `trade_certified`: the family remains admissible and the edge survives costs.
  - `preserve_certified`: the family remains structurally admissible, but microstructure kills tradeability, so hold/preservation is the correct certified behavior.
- DuckDB now includes `nashi_family_certifications` plus step-level family columns in `nashi_steps`.

## Next steps

1. Replace the bootstrap proposal adapter with a stricter Agda-backed proposal surface.
2. Add parity artifacts for step/family witness export against the sibling Agda work.
3. Replace the current heuristic Phase-9 regime proxy with actual Phase-6/7 authority and readiness inputs.
4. Extend the family witness from rolling-window certification to full regime census and backtest certification.

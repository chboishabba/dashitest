# Compactified Context

- Phase-4 density monitor now enforces per-ontology balance ≥25%, bin persistence across ≥6 of the last 8 runs, ≥40 rows per bin, monotonic medians, and effect-size ≥0.0005 (`scripts/phase4_density_monitor.py`); the `strict` profile in `configs/phase4_monitor_profiles.json` captures that bundle and `phase4_monitor_runner.py` applies it for any tape.
- Gate status now requires 3 consecutive OPENs before reporting success, and every iteration records a `blocking_reason` string in `logs/phase4/density_monitor.log`/JSON payloads so you know exactly why training is still gated.
- Candidate density target: ES/NQ intraday futures (see `docs/es_nq_ingestion_plan.md`) keep the trainable regime clean while reusing the existing proposals/prices contract.
- Phase-5 execution simulator (`scripts/phase5_execution_simulator.py`) consumes proposals + prices, applies slip/fee bps, and emits timestamped JSONL entries with `realized_pnl`, `slippage_cost`, and `execution_cost`.
- TODO tracker now reflects the tightened gate, ES/NQ tape plan, Phase-5 simulator, and the Phase-6 capital-control sketch so the backlog knows what still needs multi-step completion.
- Added ES/NQ ingestion artifact: 5-minute price files, synthetic proposals, `scripts/check_esnq_ingestion.py`, and a monitor run that now reports ES/NQ gates as `OPEN` once the strict profile runs for ≥6 iterations.
- Added a clarification in `docs/phase4_phase5_strategy.md` about synthetic/amplitude-injected OPENs being mechanical tests and noted how to flag those runs so Phase-4.1 only fires off real data.
- `scripts/phase4_density_monitor.py` now accepts `--test-vector` so amplitude-injected runs mark `test_vector` in `logs/phase4/density_monitor.log`, and `docs/phase4_density_monitor.md` explains how to interpret that tag plus how to refresh BTC/ES/NQ data using `python data_downloader.py`.

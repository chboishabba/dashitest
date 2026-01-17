# Phase-4 Density Monitor

`scripts/phase4_density_monitor.py` turns the Phase-4 diagnostics script into an event-driven gate watcher:

* Runs `scripts/train_size_per_ontology.py` against every tape you care about.
* Logs `T/R` counts, per-bin counts/medians, and the gate status (`OPEN`/`closed`) to `logs/phase4/density_monitor.log`.
* Keeps the most recent `--history` entries in memory so the terminal always reports a rolling open-rate.

## Gate logic

The monitor evaluates the Phase-4 gate for each tape on every iteration:

* Require at least `--min-total-tr` `T+R` rows (default `200`).
* Require `--min-bins` distinct size bins with ≥ `--min-rows-per-bin` rows (defaults `2` bins with ≥20 rows each) per ontology.
* Require `min(T,R) / (T+R) ≥ --min-ontology-ratio` (default `0.25`) so learning isn’t dominated by a single ontology.
* Require each bin to be present in ≥ `--bin-persistence-required` of the last `--bin-persistence-window` runs, preventing single bursts from opening the gate.
* Gate opens once any ontology (`T` or `R`) meets the density criteria and has ≥ `--min-effect-size` median spread (default `0.0003`).
* By default the gate only reports `OPEN` when two consecutive checks pass; set `--no-debounce` to disable the hysteresis.
* Phase-4 only unblocks after Phase-07 is ready; the monitor requires a Phase-07 status log and a persistence window of `phase7_ready=true` before it will open.

You can tweak those parameters to make the gate more or less conservative, but the defaults match the Phase-4 acceptance checklist.

## Phase-07 gating (asymmetry density)

Phase-07 is the asymmetry census: Phase-4 cannot open unless Phase-07 reports persistent readiness. The monitor reads a JSONL status log (default `logs/phase7/density_status.log`) and requires `phase7_ready=true` to appear in at least `--phase7-persistence-required` of the last `--phase7-persistence-window` entries for the same `target`.

The emitter comes from `scripts/phase7_status_emitter.py`, which follows the definitions in `docs/boundary_stable_eigen.md` and writes the contract described in `docs/phase7_status_emitter.md`. Phase-04 only unblocks after the net asymmetry density survives the boundary cost proxy.

Expected JSONL fields (one per line):

```json
{"timestamp": "2024-01-01T00:00:00Z", "target": "BTC", "phase7_ready": true, "phase7_reason": "asymmetry_density_ok", "phase7_metrics": {"density": 0.62}}
```

If the Phase-07 log is missing or does not contain matching entries for the target, the Phase-4 gate stays closed with a `phase7_status_missing` blocking reason.

## Usage

```bash
python scripts/phase4_density_monitor.py \
  --target BTC=data/btc/proposals_intraday.csv,data/btc/close.csv \
  --target SPY=data/spy/proposals_daily.csv,data/spy/close.csv \
  --interval 600 \
  --phase7-log logs/phase7/density_status.log \
  --diag-out-dir logs/phase4/density_monitor \
  --monitor-log logs/phase4/density_monitor.log
```

Running without `--interval` keeps the monitor in a single pass, which is useful when you want a quick snapshot. With `--interval`, the script sleeps between iterations and keeps appending to the log.

## Multi-tape runner

Use `scripts/phase4_monitor_runner.py` when you want to start the monitor across all of your tapes with one command:

```
python scripts/phase4_monitor_runner.py \
  --config configs/phase4_monitor_targets.json \
  --monitor-arg "--interval 600" \
  --monitor-arg "--diag-out-dir logs/phase4/density_monitor"
```

The runner reads `configs/phase4_monitor_targets.json` (each entry needs `name`, `proposal_log`, `prices_csv`) and turns those into `--target NAME=proposal,prices` arguments before invoking the monitor.

## Profiles & blocking reason

Named flag bundles live in `configs/phase4_monitor_profiles.json`. The `strict` profile already matches the new toughness you asked for, so run:

```
python scripts/phase4_monitor_runner.py \
  --config configs/phase4_monitor_targets.json \
  --profile strict
```

The monitor appends a `blocking_reason` field to each JSON line so you can see the first condition that kept the gate closed; when the gate opens, `blocking_reason` is empty and `gate_reasons` instead contains the success message.

### Strict profile (canonical)

Phase-4.1 training can only run after the `strict` profile returns an `OPEN` verdict; every condition it checks is recorded in `configs/phase4_monitor_profiles.json`, so the script keeps a single, reproducible definition of “safe to train.” The profile enforces:

* `min_rows_per_ontology=120` to prevent single-ontology domination.
* `min_bins=3`, `min_rows_per_bin=40`, and `min_bin_balance=0.35` so every bin is substantive and both ontologies participate.
* `min_effect_size=0.0005` to guarantee the size geometry survives Phase-5 slip/fee modeling.
* `min_monotone_bins=2` so medians have meaningful order rather than jitter.
* `bin_persistence_window=8`, `bin_persistence_required=6`, and `median_flip_max=1` to ensure stability before declaring density.
* `consecutive_passes=3` (or `--no-debounce` to override) so the gate only reports `OPEN` after three successive successful probes.

If any of those checks fail, you keep waiting. The gate is a quality filter, not a race—use the monitor log and `phase4_gate_status.md` to understand which condition is still pending before Phase-4.1 can start.

## Flagging mechanical tests

When you manually inject amplitude separation (or otherwise massage your proposals) to exercise the stack, pass `--test-vector <tag>` to `scripts/phase4_density_monitor.py` (or wrap it via `scripts/phase4_monitor_runner.py --monitor-arg "--test-vector synthetic_amplitude"`). The monitor writes a `test_vector` field to `logs/phase4/density_monitor.log`, prints `test_vector=<tag>` in the terminal, and keeps the tag in each iteration payload under `logs/phase4/density_monitor/*.json`. That way you can filter for those entries and avoid mistaking the resulting `OPEN` for a data-driven readiness signal.

Continue to treat `OPEN` as real only when no `test_vector` tag exists and the rows came from intact BTC/ES/NQ proposals.

## Daily gate status summary

`phase4_gate_status.md` captures the most recent status per tape so you do not have to hunt through the JSON log. Run:

```
python scripts/phase4_gate_status.py \
  --log logs/phase4/density_monitor.log \
  --output phase4_gate_status.md
```

to refresh the summary. The generated Markdown highlights the latest `gate_open` decision, `gate_reasons`, and `blocking_reason` for each target across recent dates so you can tell at a glance what prevented `OPEN` today.

## Outputs you can watch

* `logs/phase4/density_monitor.log` (JSON lines): each entry records the gate status, counts, bin medians, blocking_reason, and the `train_size_per_ontology.py` payload that drove the decision.
* `logs/phase4/density_monitor/<tape>-<timestamp>.json`: the raw Phase-4 diagnostics payload you can inspect later.
* Terminal output: summary lines showing the gate, counts, bin coverage, recent open rate, and reasons for closing.

Let the monitor collect data and trigger Phase-4.1 automatically when the gate goes `OPEN`. If it stays closed, focus on collecting more BTC intraday history or other tapes until your data passes the gate.

## Data availability checklist

Blocking on `T rows X < 120`, `R rows ...`, or “bins=0/0” is the monitor doing its job — it is telling you that the underlying proposals still lack the density required. Instead of loosening thresholds, refresh the raw price data so the ingestion pipeline can generate new proposals:

1. Run `python data_downloader.py` to pull the latest BTC/ES/major equity bars. The script caches files under `data/raw/` (Stooq, Binance, Yahoo) so re-running it keeps those caches fresh. The Binance intraday cache now contains tens of thousands of 1m rows that feed the Phase-4 pipeline.
2. Re-run whatever proposal generation step you use (e.g., `scripts/run_proposals.py` targeting `data/btc/` etc.) so the monitor’s configuration files (`configs/phase4_monitor_targets.json`) point at logs that were generated from the refreshed data.
3. Resume `scripts/phase4_monitor_runner.py --profile strict` with `--monitor-arg "--interval 600"` (or your chosen cadence), keep it running until ≥6 persistence windows accumulate, and watch the `logs/phase4/density_monitor.log` entries with an empty `test_vector` tag.

If any download step fails (CoinGecko now requires API keys, so it may return 401), the other sources already provide the Δ-1 high-frequency BTC history we rely on. `data_downloader.py` also prints warnings when a cache hit satisfies the request so you know whether the source actually refreshed.

# Phase-07 Status Emitter

Purpose: emit a friction-aware asymmetry readiness signal so Phase-04 only opens after **boundary-stable eigen-events** occur. This is a **read-only** status stream; it never alters trading or learning behavior directly. Phase-07 & Phase-04 gate **learning** only—trading and execution permissions remain driven by the existing controller even when the emitter reports `phase7_ready=false`.

## Normative background

Phase-07 operates on the invariant defined in `docs/boundary_stable_eigen.md`. The emitter reports whether the latest asymmetry density survives the **boundary cost** proxy so Phase-04 and downstream learners never awaken on frictionless lifts.

## Output contract

Append JSONL entries to:

```
logs/phase7/density_status.log
```

Each entry has the form:

```json
{
  "timestamp": "2026-01-17T01:23:45Z",
  "target": "BTC",
  "phase7_ready": true,
  "phase7_reason": "asymmetry_density_ok",
  "phase7_metrics": {
    "rho_A_net": 0.031,
    "rho_A_gross": 0.084,
    "cost_est": 0.012,
    "window": 256,
    "count": 192
  }
}
```

The Phase-04 monitor consumes this log via `--phase7-log` and enforces persistence itself. The emitter only writes the **instantaneous test**; persistence is Phase-04’s job.

## Runtime emitter (`scripts/phase7_status_emitter.py`)

The script:

1. Reads the latest window of Phase-5 execution rows (JSONL) for a given `--target`.
2. Builds per-entry diagnostics:
   * \(d_t =\) decision direction
   * \(r_t =\) realised return per unit (price horizon minus fill price)
   * \(c_t =\) cost proxy (slippage_cost + execution_cost per unit)
   * \(m_t = 1\) whenever a sized action exists (size > 0 and direction ≠ 0)
3. Computes medians:
   * `rho_A_gross`: median of \(d_t \cdot r_t\)
   * `rho_A_net`: median of \(d_t \cdot r_t - c_t\)
   * `cost_est`: median cost per unit
4. Marks `phase7_ready = (rho_A_net > 0)` and records `phase7_reason` (`"asymmetry_density_ok"` vs `"net_asymmetry_nonpositive"` or `"insufficient_density"` when data is missing).
5. Appends a JSON line with the payload above to `logs/phase7/density_status.log`.

### Runtime flags

* `--execution-log`: Phase-5 JSONL file (default must contain `direction`, `size`, `price_t_h`, `fill_price`, `slippage_cost`, `execution_cost`).  
* `--target`: symbol written to each payload (default `default`).  
* `--phase7-log`: output path (default `logs/phase7/density_status.log`).  
* `--window`: number of entries to read from the end of the execution log (default `256`).  

The script is safe to run from cron, the dashboard, or manually; every invocation appends a single canonical status line.

### Sample run

```bash
python scripts/phase7_status_emitter.py \
  --target BTC \
  --execution-log logs/phase5/phase5_execution_20260114T122144Z.jsonl \
  --phase7-log logs/phase7/density_status.log \
  --window 256
```

## Metrics and Phase-04 integration

`phase7_metrics` includes:

- `rho_A_net`, `rho_A_gross`: per-unit medians for net/gross asymmetry density.
- `cost_est`: median boundary cost (slippage + fee) per action.
- `window`: number of entries scanned.
- `count`: number of eligible rows (sum of \(m_t\)).

Phase-5 execution logs now emit `pred_edge` (score-margin × direction when available) so the boundary gate has an explicit surrogate for the predicted net edge; the `--boundary-gate` option in `run_trader.py` compares this (via a running edge EMA) to the same cost proxy used by Phase-07 and forces HOLDS when the boundary test fails. This keeps execution aligned with the training gate without changing the core Phase-07 contract.

When the log is missing or contains no eligible rows, `phase7_ready` stays `false`; Phase-04 reports `phase7_status_missing` or `net_asymmetry_nonpositive` accordingly.

Phase-04 stays closed until enough `phase7_ready=true` lines accumulate (see `scripts/phase4_density_monitor.py`), keeping the entire stack grounded on boundary-stable eigen-events.

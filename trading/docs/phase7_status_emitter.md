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

## Diagnostics: why net asymmetry is missing

Use `scripts/phase7_asymmetry_diagnostics.py report` to separate “no edge here” from “measurement/gating is miswired.” The report mirrors Phase-07’s median logic and adds scale checks.

```bash
python scripts/phase7_asymmetry_diagnostics.py report \
  --execution-log logs/phase5/phase5_execution_20260114T122144Z.jsonl \
  --window 256 \
  --group-by-horizon
```

The report highlights:

- Horizon mismatch: if horizons differ inside the log, the per-horizon medians show whether net asymmetry only appears at longer holds.
- Cost dominance: compare `median_abs_move` vs `cost_est`. If `median_abs_move < cost_est`, net asymmetry is structurally negative at that horizon.
- Pred-edge scale checks: if `pred_edge` exists, the report prints its scale relative to realized moves and cost.

If the report is negative even after removing costs (see below), you likely have a direction/horizon mismatch rather than a fee issue.

## Controlled wake-up tests (known asymmetry)

Use the same diagnostics tool to generate “known asymmetry” logs for sanity tests. These do **not** change trading behavior; they only validate the Phase-07 → Phase-04 wiring.

### Test 1: zero-cost sanity

```bash
python scripts/phase7_asymmetry_diagnostics.py zero-cost \
  --execution-log logs/phase5/phase5_execution_20260114T122144Z.jsonl \
  --output logs/phase7/phase5_zero_cost.jsonl
```

Run the emitter on the zero-cost copy. If `rho_A_net` still stays ≤ 0, the issue is not fees but direction/horizon.

### Test 2: injected drift

```bash
python scripts/phase7_asymmetry_diagnostics.py inject-drift \
  --execution-log logs/phase5/phase5_execution_20260114T122144Z.jsonl \
  --output logs/phase7/phase5_injected_drift.jsonl \
  --drift-margin 5.0
```

This rewrites `price_t_h` so every active row is profitable by `cost_per_unit + drift_margin`. Running the emitter against this log should flip `rho_A_net` positive and unblock Phase-04 after persistence. If it does not, the wiring (parsing, sign conventions, support definition) is wrong.

## Live finding: boundary-dominated orbit at 1–3s cadence

The first live run with `phase6_gate.open=true` and coherent posture (+1) showed monotone long ramps (e.g., NEOUSDT) with non-empty support \(m_t=1\). Phase-07 still reports net asymmetry ≤ 0 because boundary costs dominate: exposure changes arrive every 1–3 seconds, price moves at that horizon are spread/noise-sized, and cost is paid on each \(\Delta x_t\). This is a valid eigen-orbit that collapses inside the friction boundary, not a wiring bug.

Experiments to confirm and resolve:

- **Horizon lift**: hold exposure for 30–120s, charge cost only on entry/exit, and recompute medians. Expect net to flip if a slow trend exists.
- **Zero-cost sanity**: set boundary cost to 0 with all else identical; net should turn positive immediately, proving the measurement path.
- **Actuation alignment**: choose one of (a) slower controller clock, (b) batched/impulse exposure changes, or (c) charge boundary cost only on sign flips/regime exits. This aligns continuous-time control with discrete-time surplus measurement and prevents cost-only decay.

## Phase-8 intent: net-surplus gate on live

Phase-8 = a live system that trades only when Phase-07 shows the action stream is net extractable (actuator + horizon + costs), and it proves that claim on real execution logs end-to-end.

Prereqs (Phase-7.x hardening):

- **Phase-7.1 (Actuator testability)**: add live decision decimation/impulse modes so supported events are sparse and boundary-aware. Examples: `--hold-seconds {30,60,120}` to freeze exposure between updates; `--impulse`/`--deadband` to jump once then hold until exit.
- **Phase-7.2 (Live execution ledger)**: emit a live ledger (even paper) in the Phase-07 schema (`timestamp, symbol, x_prev, x, mid, delta_mid, cost_est, realized_pnl, pred_edge`) so `scripts/phase7_status_emitter.py` runs unmodified on live logs.
- **Phase-7.3 (Horizon sweep)**: harness to run the same live log through Phase-07 at multiple horizons (e.g., 10s/30s/60s/120s/300s) to locate any horizon with positive net density or prove absence.

Phase-8 entry gate (all must be true):

- Actuator mode fixed and logged (decimation/impulse parameters).
- Phase-07 ready persists on live logs (your persistence policy, e.g., N of last M windows).
- Boundary gate enforced at runtime: if `phase7_ready=false`, trades clamp to HOLD/OBSERVE.
- One-command audit emits session summary (net/gross density, activity rate, cost share vs move share, worst drawdown).

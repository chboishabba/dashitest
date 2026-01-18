# Phase-07 Status Emitter

Purpose: emit a friction-aware asymmetry readiness signal so Phase-04 only opens after **boundary-stable eigen-events** occur. This is a **read-only** status stream; it never alters trading or learning behavior directly. Phase-07 & Phase-04 gate **learning** only—trading and execution permissions remain driven by the existing controller even when the emitter reports `phase7_ready=false`.

## Normative background

Phase-07 operates on the invariant defined in `docs/boundary_stable_eigen.md`. The emitter reports whether the latest action-channel asymmetry density survives the **boundary cost** proxy so Phase-04 and downstream learners never awaken on frictionless lifts.

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
    "rho_A": 1.42,
    "sum_edge": 12.0,
    "sum_cost": 8.4,
    "n_support": 192,
    "activity_rate": 0.75,
    "robust_pass": true,
    "class_label": "boundary-stable",
    "window": 256,
    "count": 192
  }
}
```

The Phase-04 monitor consumes this log via `--phase7-log` and enforces persistence itself. The emitter only writes the **instantaneous test**; persistence is Phase-04’s job.

## Runtime emitter (`scripts/phase7_status_emitter.py`)

The script:

1. Reads the latest window of decision/action rows (NDJSON) for a given `--symbol`.
2. Builds action-channel deltas:
   * `m_t = pi_supp(d_t)`
   * `edge_t = x_{t-1} * Delta m_t`
   * `cost_t = (half_spread + fee + slip) * m_t * |Delta x_t|`
3. Computes action-channel asymmetry density:
   * `rho_A = sum(edge) / (sum(cost) + eps)`
4. Applies robustness gates (support threshold, cost perturbation, threshold).
5. Marks `phase7_ready` and appends a JSON line to `logs/phase7/density_status.log`.

### Runtime flags

* `--decisions-ndjson`: decision/action NDJSON file (must contain `timestamp`, `symbol`, `direction`, `target_exposure`).
* `--symbol`: symbol filter for multi-symbol logs (optional).
* `--target`: label written to each payload (default `default`).
* `--phase7-log`: output path (default `logs/phase7/density_status.log`).
* `--window`: number of entries to read from the end of the decisions log (default `256`).
* `--half-spread`, `--fee`, `--slip`: boundary cost inputs.
* `--rho-thresh`, `--persist-min`, `--eps-cost-frac`: readiness and robustness parameters.

The script is safe to run from cron, the dashboard, or manually; every invocation appends a single canonical status line.

### Sample run

```bash
python scripts/phase7_status_emitter.py \
  --decisions-ndjson logs/decisions.ndjson \
  --symbol BTCUSDT \
  --target BTC \
  --half-spread 0.0003 \
  --fee 0.0005 \
  --slip 0.0 \
  --window 256
```

## Metrics and Phase-04 integration

`phase7_metrics` includes:

- `rho_A`: edge/cost density for the action stream.
- `sum_edge`, `sum_cost`: aggregates used for the ratio.
- `n_support`, `activity_rate`: support size and rate (separate from edge).
- `robust_pass`, `class_label`: robustness verdict and coarse classification.
- `window`, `count`: total rows scanned and supported rows.

When the log is missing or contains no eligible rows, `phase7_ready` stays `false`; Phase-04 reports `phase7_status_missing` or the latest `phase7_reason` accordingly.

Phase-04 stays closed until enough `phase7_ready=true` lines accumulate (see `scripts/phase4_density_monitor.py`), keeping the entire stack grounded on boundary-stable eigen-events.

## Diagnostics: why asymmetry is missing

Use `scripts/phase7_asymmetry_diagnostics.py report` to separate “no edge here” from “measurement/gating is miswired.” The report mirrors Phase-07’s edge/cost logic and adds scale checks.

```bash
python scripts/phase7_asymmetry_diagnostics.py report \
  --decisions-ndjson logs/decisions.ndjson \
  --symbol BTCUSDT \
  --window 256
```

If the report is negative even after removing costs, you likely have a sign/support mismatch rather than a fee issue.

## Controlled sanity checks

### Test 1: zero-cost sanity

Run the emitter with cost inputs set to zero:

```bash
python scripts/phase7_status_emitter.py \
  --decisions-ndjson logs/decisions.ndjson \
  --symbol BTCUSDT \
  --target BTC \
  --half-spread 0.0 \
  --fee 0.0 \
  --slip 0.0
```

If `rho_A` still stays <= 0, the issue is not fees but direction/support wiring.

### Test 2: cost perturbation check

Increase `--eps-cost-frac` to confirm the robustness gate is sensitive to small cost shifts.

## Live finding: boundary-dominated orbit at 1–3s cadence

The first live run with `phase6_gate.open=true` and coherent posture (+1) showed monotone long ramps (e.g., NEOUSDT) with non-empty support (`m_t=1`). Phase-07 still reports `rho_A <= 0` because boundary costs dominate: exposure changes arrive every 1–3 seconds, price moves at that horizon are spread/noise-sized, and cost is paid on each `Delta x_t`. This is a boundary-dominated eigen-orbit, not a wiring bug.

Experiments to confirm and resolve:

- **Horizon lift**: hold exposure for 30–120s, charge cost only on entry/exit, and recompute `rho_A`. Expect `rho_A` to flip if a slow trend exists.
- **Zero-cost sanity**: set boundary cost to 0 with all else identical; `rho_A` should turn positive immediately, proving the measurement path.
- **Actuation alignment**: choose one of (a) slower controller clock, (b) batched/impulse exposure changes, or (c) charge boundary cost only on sign flips/regime exits. This aligns continuous-time control with discrete-time surplus measurement and prevents cost-only decay.

## Phase-8 intent: net-surplus gate on live

Phase-8 = a live system that trades only when Phase-07 shows the action stream is net extractable (actuator + horizon + costs), and it proves that claim on real decision logs end-to-end.

Prereqs (Phase-7.x hardening):

- **Phase-7.1 (Actuator testability)**: add live decision decimation/impulse modes so supported events are sparse and boundary-aware. Examples: `--hold-seconds {30,60,120}` to freeze exposure between updates; `--impulse`/`--deadband` to jump once then hold until exit.
- **Phase-7.2 (Decision ledger)**: emit a live decision ledger matching the Phase-07 schema (`timestamp, symbol, direction, target_exposure, phase6_gate`) so `scripts/phase7_status_emitter.py` runs unchanged on live logs.
- **Phase-7.3 (Horizon sweep)**: harness to run the same live log through Phase-07 at multiple horizons (e.g., 10s/30s/60s/120s/300s) to locate any horizon with positive net density or prove absence.

### Phase-7.3 horizon sweep: noise-floor annotation

Phase-7 is epistemic, not economic. The horizon sweep must annotate short-horizon inflation without introducing capital-equivalent deltas.

For each horizon `tau`, compute a null-normalized score:

1. **Shuffle control**: compute `rho_A_null(tau)` using one of:
   - shuffled `m_t` or `d_t`
   - shuffled `x_t`
   - time-shifted action stream (shift >> tau)
2. **Excess signal**:
   `delta_rho_A(tau) = rho_A(tau) - rho_A_null(tau)`
3. **Noise ratio**:
   `noise_ratio = rho_A(tau) / (rho_A_null(tau) + eps)`

Add these columns to the Phase-7.3 output (CSV/JSONL):

```csv
horizon_s,
rho_A,
rho_A_null,
delta_rho_A,
noise_ratio,
robust_pass,
class_label
```

Interpretation rule (documented, not gated):

- High `rho_A` at small `tau` is **non-diagnostic** unless `delta_rho_A(tau)` remains positive and monotone as `tau` increases.
- Small-`tau` spikes are kinematic; `tau`-stable plateaus are eigen-structural.

Do **not** add capital-equivalent deltas in Phase-7. Those belong to Phase-8+ after horizon, persistence, and boundary stability are fixed.

### Phase-7.3 horizon sweep emitter (implemented)

Use `scripts/phase73_horizon_sweep.py` to batch a tower log through Phase-07 at multiple horizons.

Inputs:

- tower NDJSON rows with `ts`, `symbol`, `run_id`, and action-channel fields (configurable paths).
- a tau grid (e.g., 10s/30s/60s/120s/300s)
- a friction model (cost_est or explicit fee/slip parameters)

Output NDJSON (one line per tau):

```json
{
  "run_id": "trading_log",
  "symbol": "BTC",
  "tau_s": 60,
  "rho_A": 0.0142,
  "rho_A_null": 0.0119,
  "delta_rho_A": 0.0023,
  "noise_ratio": 1.19,
  "robust_pass": true,
  "class_label": "boundary-stable"
}
```

Certification rule (per tau):

```
net_positive = (rho_A > rho_A_null + eps) and robust_pass and stable_under_tau_variation
```

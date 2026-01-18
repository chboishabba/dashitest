# Tower Projection Dashboard (M1-M9)

Purpose: plot typed projections of the same tower state, not nine unrelated scores. The dashboard is diagnostic only and preserves the phase rule: projections supervise the action channel, never steer it.

## Core contract

- M1..M9 are typed projections P_k(X_t) of the same nested state X_t.
- Never collapse M1..M9 into a single confidence score or ladder.
- Any activation A_k is a certificate satisfaction functional, not belief.
- The internals view is optional and gated by a flag (see below), so the default dashboard layout stays unchanged.

## Plot groups (tower-aligned)

Group A - kernel / tower closure (M1-M4):
- kappa across a small set of scales (min/median/max or explicit j's).
- closed_scales = count(kappa_j <= eps_j).

Group B - posture / buffer (M5):
- posture in {-1, 0, +1}
- HOLD or observe flags, plus hysteresis state if present.

Group C - dual-kernel tension (M6):
- legitimacy and exploitability side by side.
- highlight disagreement regions (legitimacy high, exploitability low, and vice versa).

Group D - boundary certificate + Phase-7.3 horizon sweep (M7):
- rho_A(tau) across horizons.
- robust pass/fail per tau.
- noise-floor annotation (rho_A_null, delta_rho_A, noise_ratio).

Group E - readiness gate (M8):
- ready_count / required / window.
- phase8_gate.open and reason codes.

Group F - witness / capital (M9):
- refusal state (NONE/HOLD/BAN).
- capital curve, drawdown, and clamp activations if simulated.

## Activation is certificate satisfaction (diagnostic only)

Define A_k(t) = sat_k(P_k(X_t)), with layer-specific satisfaction functions. Examples:

- A1 = closed_scales / total_scales
- A5 = posture coherence (not sign), e.g. stable -> 1.0, neutral -> 0.5, oscillatory -> 0.0
- A6 = 1 - |legitimacy - exploitability| / (max(legitimacy, exploitability) + delta)
- A7 = (# taus where (rho_A - rho_A_null) > 0 and robust) / |taus|
- A8 = min(1, ready_count / required)
- A9 = ACT=1.0, HOLD=0.5, BAN=0.0

Plot A_k only as a diagnostic stack. Do not feed A_k into action selection.

## Near-miss signatures (why this view exists)

- M7 strong but M8 low: readiness/persistence issue.
- M5 stable but M7 weak: coherent posture, boundary instability or horizon mismatch.
- M8 open but M9 refusing: witness or capital clamp.

These are interpretation guides, not triggers.

## PyQtGraph integration (non-cluttering)

The internals view should be a PyQtGraph-only extension activated by a flag such as:

```
PYTHONPATH=. python training_dashboard_pg.py --log logs/trading_log.csv --refresh 1.0 --graph-internals
```

Default layout remains unchanged unless `--graph-internals` is passed.

## Tower projection log (NDJSON)

The runner emits a tower projection log alongside the per-step CSV. It is NDJSON so nested P1..P9 stay typed.

- Default path: `logs/trading_log*_tower.ndjson` (same stem as the CSV log).
- Disable emission with `run_trader.py --no-tower-log`.
- Fields are null when the projection is not computed. Each P_k carries `available` to avoid semantic leakage.

Example record (current emission contract):

```json
{
  "t": 120,
  "ts": "2024-01-01 00:02:00",
  "symbol": "btc",
  "run_id": "trading_log",
  "P1": {
    "available": false,
    "q": {"e64": 0.12, "c64": 0.05, "s64": 0.44, "delta_e": 0.01, "delta_c": 0.02, "delta_s": 0.03},
    "kappa": null,
    "eps": null,
    "closed_scales": null,
    "total_scales": null,
    "A1": null
  },
  "P5": {
    "available": true,
    "posture": 1,
    "posture_source": "direction",
    "hold": 0,
    "state_age": 4,
    "align_age": 3,
    "stable": true,
    "A5": 1.0
  },
  "P6": {
    "available": false,
    "legitimacy_proxy": 0.78,
    "exploitability_proxy": 0.004,
    "agreement": null,
    "A6": null
  },
  "P7": {
    "available": false,
    "tau_s": null,
    "rho_A": null,
    "rho_A_null": null,
    "delta_rho_A": null,
    "robust": null,
    "A7": null,
    "boundary_gate": {
      "enabled": 1,
      "abstain": 0,
      "edge_confidence": 0.003,
      "cost_threshold": 0.0005,
      "reason": ""
    }
  },
  "P8": {"available": false, "ready_count": null, "required": null, "window": null, "open": null, "reason": null, "A8": null},
  "P9": {
    "available": true,
    "permission": 1,
    "refusal": "NONE",
    "A9": 1.0,
    "capital_pressure": 0,
    "risk_budget": 0.7,
    "cap": 0.9,
    "equity": 100120.0,
    "cash": 100050.0
  }
}
```

Notes:

- P1 uses quotient feature proxies until kappa/eps closure is computed.
- P6 proxies come from `1 - p_bad` and `pred_edge`; they are labeled as proxies and keep `available=false`.
- P7 boundary certificate remains empty until Phase-7.3 horizon sweeps are wired; boundary gate metrics are logged under `boundary_gate` as a proxy only.

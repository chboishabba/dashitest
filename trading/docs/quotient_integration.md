# Quotient Invariant Integration (Triadic Trader)

## Purpose

Train on rich price trajectories, but integrate learning only through
projection-invariant quotient features. Apparent instability in raw coordinates
is treated as gauge drift, not failure.

This follows the projection-invariance framing in `CONTEXT.md#L3337` through
`CONTEXT.md#L3342` and the projection-invariants framing in
`CONTEXT.md#L1055`.

## Objects

- Representation (learner input): `s_t` = rolling price trajectory window.
- Projection (trader uses): `Pi(s_t)` = existing triadic signals/thresholds.
- Equivalence: `s ~ s'` iff `Pi(s) == Pi(s')`.
- Quotient: `Q(s_t)` = invariant feature extractor that removes gauge degrees.

## Success criterion

Evaluate with quotient distance, not raw-space error:

```
d_Q(s, s') = ||Q(s) - Q(s')||^2
```

Interpretation rule:
stability in `Q` with drift in raw coordinates is expected and acceptable.

## Quotient feature set v0

### Design constraints

- Invariant under price scale and affine shifts.
- Stable across reasonable window choices.
- Regime/legitimacy signal only (no direction).
- Online and cheap to compute.
- Degrades gracefully under noise.

### Input signal

Use log-returns only:

```
r_t = log(p_t / p_{t-1})
```

### Windowing

Fixed windows:

- `W1 = 64` bars (primary regime)
- `W2 = 256` bars (persistence/background)

### Normalization

Robust volatility per window:

```
sigma_W = MAD(r_{t-W:t})
tilde_r_i = r_i / (sigma_W + eps)
```

### Window features

Energy (activity, not direction):

```
E_W = (1 / W) * sum(tilde_r_i^2)
```

Signed cancellation ratio (structure vs noise, direction-free):

```
C_W = abs(sum(tilde_r_i)) / sum(abs(tilde_r_i))
```

Spectral concentration (regime texture):

```
S_W = max(P_k) / sum(P_k)
```

### Multiscale consistency

For each `f` in `{E, C, S}`:

```
Delta_f = abs(f_W1 - f_W2)
```

### Final quotient vector

```
Q_t = [E_64, C_64, S_64, Delta_E, Delta_C, Delta_S]
```

## Integration paths (ordered by risk)

1) Quotient gating (recommended start)
- Predict `Q(s_{t+1})` or compute `Q(s_t)` directly.
- Feed into `strategy/triadic_strategy.py` as ACCEPT/HOLD/BAN gating only.
- Do not change action direction logic.

2) Quotient-loss evaluator (diagnostic)
- Compare predicted vs realized `Q` to emit a confidence scalar.
- Use to modulate aggressiveness, not direction.

3) One-step latent transition (only after stability)
- Blend predicted next triadic latent state with heuristics.
- Requires long-run quotient stability.

## Logging and evaluation requirements

- Log `Q(s_t)` and optional `Q_pred(s_{t+1})`.
- Log `quotient_error = d_Q(Q_pred, Q_real)` when applicable.
- Keep all action logic unchanged until quotient stability is verified.

## Implementation posture

- Log `Q_t` alongside existing signals.
- Do not change action logic while validating quotient stability.

## Stability check (BTC/SPY)

Run a quotient stability report that compares `q_*` drift inside stable
permission segments vs transition points. This is a diagnostics-only check and
should not gate actions yet.

Suggested command:

```
PYTHONPATH=. python trading/scripts/score_quotient_stability.py --log logs/trading_log.csv --min-run 20
```

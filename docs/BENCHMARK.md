# Benchmark Protocol (Dynamics + Quotient Evaluation)

This protocol packages the current workflow so it can be ported to new domains
without the Gray-Scott or primes backstory. The core idea: learn the one-step
map in a rich space, but evaluate multi-step rollouts with quotient/invariant
metrics to avoid projection artifacts.

Context: projection/invariants framing in `CONTEXT.md#L3337` and
`CONTEXT.md#L1055`.

## 1) Objects and equations

### Dynamics target

- State: `x_t in R^d`
- True transition: `x_{t+1} = F(x_t)`
- Dataset: `D = {(x_i, y_i)}_{i=1..n}` with `y_i = x_{i+1}`

### One-step learner (KRR)

```
F_hat(x) = sum_i alpha_i * k(x, x_i)
alpha = (K + lambda I)^{-1} Y
```

### Kernels compared

- RBF (Euclidean)
- Periodic RBF (grid wrap)
- Dashifine kernel (ultrametric / valuation-like similarity)

### Evaluation (two-layer)

1. One-step MSE
2. Multi-step rollout:

```
x_hat_{t+1} = F_hat(x_hat_t), x_hat_0 = x_0
```

3. Quotient/invariant metric (evaluate meaning, not raw coordinates)

For Gray-Scott, the chosen quotient is `V + radial(U)`:

```
MSE_quot(t) = MSE(V_t, V_hat_t) + MSE(rho(U_t), rho(U_hat_t))
```

## 2) Theorem statement (informal)

Gauge orbit drift under projection:

If dynamics are equivariant under a symmetry group `G`, and the observation
projection is not injective on `G`-orbits, then a learner can be accurate on
`X/G` (the quotient) while appearing unstable in raw coordinates. The correct
diagnostic is a quotient/invariant evaluation functional.

## 3) Problem-agnostic checklist

1. Define one-step target and rollout length `T`.
2. Compare at least two geometries (Euclidean baseline + geometry-aware kernel).
3. Record one-step and rollout MSE separately.
4. Inspect rollout for structured orbit artifacts (swirls/rings).
5. Define a quotient metric before arguing about failure.
6. Only then modify training or regularization.

## 4) Replicable benchmark protocol

Minimum bundle for a new domain:

- Dataset generation seed(s)
- Train/test split
- Hyperparameter grid (lambda, lengthscale/temperature)
- Outputs:
  - one-step MSE
  - rollout curves (raw)
  - quotient-rollout curves
  - snapshot panels
- kernel spectrum plot

## 6) Recommended next task (tree diffusion)

See `docs/tree_diffusion_benchmark.md` for a minimal ultrametric transport task
with Euclidean projection, quotient evaluation, and tree-intrinsic metrics.

## 5) Primes status (current)

Active tasks:

- Divisibility indicators: `1[p^k | n]`
- Valuation regression: `v_p(n)` (capped or normalized)

Observed behavior:

- Periodic RBF dominates simple divisibility MSE for small primes.
- Dashifine improves on higher powers, consistent with hierarchy-aligned loss.

Next step in primes:

- Use hierarchy-aware metrics (weighted by `k`, calibration by residue class,
  or predict sieve-step state) to avoid base-rate artifacts in MSE.

# Valuation-only primes benchmark (plan)

Purpose: focus primes benchmarking on valuation-structured targets rather than
Euclidean residue indicators.

Context anchors:
- CONTEXT.md#L4466 notes Dashifine tracking valuation depth over residue smoothness.
- CONTEXT.md#L4994 calls for a valuation-only rollout to expose the tree structure.

## Intended targets
- Full valuation vector: n -> (v_2(n), v_3(n), v_5(n), ...)
- Max prime power dividing n: max_k {p^k | n}
- Ultrametric distance: d_p(n, m) = p^{-v_p(n-m)}

## Evaluation focus
- Measure error by valuation depth, not raw divisibility.
- Prefer quotient-consistent metrics (tree depth accuracy, depth-weighted loss).

## Notes
- Keep divisibility indicators only as diagnostics, not primary loss/selection.

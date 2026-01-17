# Influence Tensor on Quotient Representatives

This note defines the influence tensor (math + operational) that lives strictly on the quotient representatives produced by the summariser. Raw 1s prices/ticks never enter this layer; only `qfeat` and legitimacy (`ℓ`) streams do.

## Formal definition

Let `q_t^{(i)} ∈ ℝ^d` be the quotient representative for instrument `i` at aligned time `t`. Define its delta:

```
Δq_t^{(i)} = q_t^{(i)} − q_{t−1}^{(i)}
```

The influence tensor is the regularised linear map:

```
I_{i→j}(τ) ≈ argmin_W E_t[‖Δq_t^{(j)} − W Δq_{t−τ}^{(i)}‖^2 + λ‖W‖^2]
```

for each lag `τ` (can be positive or negative). The tensor carries dimensions `[lag, source_dim, target_dim]` but is sparse in practice because only stable, high-legitimacy samples contribute.

### Semantic variants

* `I^ℓ_{i→j}(τ)` — influence on legitimacy: regress `Δℓ_t^{(j)}` on `Δq_{t−τ}^{(i)}`.
* `I^Δ_{i→j}(τ)` — influence on transitions: regress action-state deltas (e.g., HOLD→ACT) on source quotient changes.
* Weights `W` are conditioned on both streams having `ℓ > ℓ_min` to focus on meaningful regimes.

## Operational rules

1. **Alignment**: Build a canonical UTC second grid. Each instrument fills the grid with its latest `q` value and records a `staleness` counter. Downstream uses can downweight stale sources.
2. **Legitimacy weighting**: Multiply each sample by `ℓ_src × ℓ_tgt` so the tensor learns from honest regimes.
3. **Batch refresh / online**: Update the tensors via incremental ridge regression (or incremental pseudo-inverse) whenever new summarised windows close.
4. **Storage**: Persist as DuckDB tables/views with columns `(lag, source_dim, target_dim, weight, artifact_ts)` and optional Parquet exports for heavy analysis.

## Notes

* Influence tensors are **controllers**, not predictions. They inform Phase-4/5 density gating but do not directly issue trades.
* The tensor is defined only over quotient reps; referencing raw ticks here would violate the epistemic boundary.

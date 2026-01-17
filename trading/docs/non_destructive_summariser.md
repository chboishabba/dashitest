# Non-Destructive Summariser Spec

This doc describes what it means for a summariser to be allowed: it must operate on quotient representatives, preserve the invariants we care about, and never collapse contradiction into noise.

## Plane A vs Plane B

- **Plane A (raw field)**: 1s OHLCV chunks, gzipped, immutable. Joins across instruments only for alignment. No inference happens here.
- **Plane B (quotient reps)**: streams of `q_t` that have already collapsed nuisance symmetries. Summaries live here; the summariser is a homomorphism on these reps.

## Summary operator requirements

Let `Q(x)` produce the quotient stream and `S` be a summary operator over a time window `W`. `S` is admissible if and only if it preserves the ability to recompute:

1. The quotient feature sequence `q_t` (within tolerable error) for downsampled times.
2. Legitimacy indicators `ℓ_t` (density, `UNKNOWN` flags, edge_t signals) used by gates.
3. Persistence counts and contradiction markers needed for Phase-4 strict geometry.
4. Audit signals (résidual surprise, MDL) that the learning loop uses as integrity checks.

### Minimal summary fields (per window)

* `count`, `mean`, `median`, `MAD` for each `q_dim`
* `p01/p05/p95/p99` quantiles for curvature/volatility dims
* `run_length_pos`, `run_length_neg` for persistence dims
* `event_marker` flag for extreme curvature or drawdown on the window
* `legitimacy_density` (average `ℓ`) and `unknown_frac`

These fields are enough to reconstruct all gating logics without touching raw bars.

## Process outline

1. DuckDB tables contain raw 1s data (not summaries).
2. Summariser modules read DuckDB, compute `q_t` (via `QuotientMap`), and emit summary documents that conform to the `non-destructive` contract.
3. Summaries are written back as Parquet or DuckDB views for Phase-4/5 gating.

The key is that the summariser does not look at raw ticks except through `Q`; it never aggregates raw OHLC by itself.

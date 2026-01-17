# Quotient Map Interface and Invariance Rules

This note locks the semantics for every summariser, influence tensor, and gating decision that follows from the raw 1s field. Implementations must respect the quotient-map contract before any DuckDB loading or summarisation starts.

## 1. Quotient map definition

A **quotient map** is a deterministic function

```
q : (instrument, time, raw_field) → representative
```

such that it collapses nuisance symmetries while preserving forced behavior geometry and contradiction persistence. The following rules are mandatory:

- **Translation invariance**: shifting the time origin by any multiple of `Δt` that preserves chunk boundaries must not change the representative up to a deterministic label (e.g., adding a tag for the chunk ID).
- **Scale invariance**: rescaling price units (BTC/ETH vs USDT) or volume units must either be normalized in the representative or documented as an admissible shift.
- **Microstructure permutation**: the particular order of trades within a closed chunk is noise; the representative must be invariant under reordering that preserves OHLC counts/volumes.
- **Non-destructive contradictions**: signs of contradiction (e.g., wick asymmetry, imbalance flips) must survive projection; representatives cannot collapse opposing density into a single scalar without capturing their polarity.
- **Symmetry metadata**: each representative carries metadata describing the nuisance group it quotients out (e.g., `scale_invariance=USDT`, `microstructure_hash=chunk1`).

Quotient maps may be composed—e.g., raw → minutely features → regime packets—as long as the composition obeys the same invariance contract.

## 2. Representative schema (example)

A hypothetical `qfeat` record could include:

- `start_ts`, `end_ts`, `instrument`
- Market geometry: OHLCV, range, trade count
- Pressure: signed return, volatility, persistence ratio
- Contradiction signals: wick asymmetry, imbalance polarity, `UNKNOWN` flag
- Symmetry metadata: `scale_normalized`, `chunk_id`, `num_trades`

The exact schema is defined later by the summariser. For now, this note asserts that representatives are the **only objects** that get compared, joined, and stored in DuckDB.

## 3. Rotation/compression helper meaning

`tools/rotate_chunks.py` will:

1. Rename the current live chunk (e.g., `live.csv`) to a timestamped archive (e.g., `2026-01-16T03.csv`).
2. Gzip the archive to produce an immutable artifact and optionally update a `rotation.log` with the start/end bounds.
3. Re-create an empty `live.csv` ready for the next write.

It operates only on closed chunks, does no aggregation, and never writes to DuckDB. This tool belongs entirely to the **raw field layer**.

## 4. DuckDB ingestion helper meaning

`tools/ingest_archives_to_duckdb.py` will:

1. Read archived `.csv.gz` files produced by the rotation helper.
2. Append new data into DuckDB tables (e.g., `btc_1s_raw`) and write optional Parquet mirrors for fast analytics.
3. Maintain an ingestion manifest (e.g., `ingested_chunks.json`) so retries are idempotent.
4. Never expose data to live gates; this script is for research queries and as the staging ground for summarisation.

The resulting DuckDB tables are not yet quotient representatives—they remain high-resolution records that may be used by subsequent summarisation passes.

## 5. Sequence recommendation

1. Formalise the quotient map interface (this document).
2. Rebuild the raw helpers (`rotate_chunks.py` and `ingest_archives_to_duckdb.py`) according to the contracts above.
3. Implement summariser modules (`trading/summarisation/quotient_map.py`, `trading/summarisation/summariser.py`) that read from DuckDB tables and output verified representatives.
4. Define influence tensors over those representatives and nowhere else.

Only after step 3 do the Phase-4/5 gating sensors consume the summaries; the live gates continue reading raw field-derived legitimacy indicators.

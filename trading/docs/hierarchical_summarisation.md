# Hierarchical Summarisation and Epistemic Memory

This note records the design the team has been circling around: a clear boundary between sensory exhaust and learnable structure, and a disciplined stack of summaries that feed Phase‑4/5/6 gates.

## Why do this?

Raw 1-second OHLCV is a *sensory firehose*—it tells us what happened, but makes no claim about what mattered. The moment we try to build persistent hypotheses directly from that stream we risk inventing structure (M9 hallucinations) or losing coherence in the face of noise (M4 paralysis). Hierarchical summarisation introduces **epistemic renormalisation**: we compress evidence to the point where contradiction can be resolved, persistence can be measured, and influence can be learned without amplifying noise.

The stack is aligned with the current gating layout:

- **1s** = micro-events (local force)
- **1m** = local coherence / contradiction
- **5–15m** = regime geometry and influence tensors
- **1h+** = posture validity, capital logic, size sensitivity

Streaming and chunked storage stay in place; everything the gates, tips, and dashboards touch must live in the summarised strata.

## The summarisation stack

1. **Level 0 — Raw sensory field**
   - AggTrades → gzipped 1s CSV chunks in `logs/binance_stream/`.
   - Used only for replay, diagnostics, forensics. Never read directly by live gates or cross-instrument logic.

2. **Level 1 — Minutely synthesis**
   - Produce 1m Parquet records that contain:
     - Market structure: OHLCV, trade count, buy/sell imbalance, range.
     - Pressure & contradiction: signed return, realised volatility, wick asymmetry, micro-trend persistence.
     - Epistemic flags: `UNKNOWN` when volume is low, the spread proxy explodes, or flip-flops dominate.
   - This is the **first join point** for influence sensors: features are local claims, not raw ticks.
   - Phase‑4 density controllers read from these rows.

3. **Level 2 — 5–15m regime packets**
   - Aggregate minutely summaries (not raw bars) to compute:
     - Trend stability, reversal density, entropy across minutes.
     - Persistence counters aligned with the strict profile used for gating.
   - This is where BTC can leave M4 and equity futures begin to show consistent behaviour.
   - Phase‑5 friction logic and influence tensors live here.

4. **Level 3 — Hourly/session rollups**
   - Control only—no signals. Use for Phase‑6 capital logic, exposure curves, and drawdown checks.
   - Track realised vs expected slippage, size-weighted PnL curvature, inventory pressure persistence, drawdown elasticity.

## Data strata and tooling

| Layer | Storage | Join policy | Primary consumers |
| --- | --- | --- | --- |
| Level 0 | `logs/binance_stream/BTCUSDT_1s_*.csv.gz` | no joins for decision logic | replay, summariser ingest |
| Level 1 | `data/market.duckdb` & `data/btc/minutely.parquet` | cross-instrument joins, influence tensors, lag/lead reasoning | Phase-4 density, gating sensors |
| Level 2 | derived views over minutely summaries (`btc_15m`, `es_5m`, etc.) | joins + persistence features | Phase-5 friction, influence monitors |
| Level 3 | DuckDB hourly/session rollups, capital logs | no raw data | Phase-6 capital control, posture reviews |

Never query raw 1s ducks directly—doing so inflates write amplification and invites noise-driven decisions. The summariser is the promotion gate.

## What gets graphed

Graphs should answer: *What evidence accumulated that justified the next state change?* Always graph price (multi-resolution), volatility, imbalance, posture/intention/gate state. Debug/research graphs may expose contradiction density, influence strength, and liquidation proxies. Raw ticks should only be plotted for diagnostics (never live dashboards without explicit hindsight markers).

## Implementation notes

Deliverables for the next sprint:

- **Minutely summariser script** (`tools/summarise_1s_to_1m.py`): reads new gz chunks, emits 1m Parquet rows with the schema above, appends only new minutes.
- **DuckDB views**: register `btc_1m`, `btc_15m`, `es_5m`, `nq_5m` (plus whatever cross-asset tables are needed) so SQL like `SELECT * FROM btc_1m WHERE imbalance > 0.8` can be interrogated.
- **Gate wiring**: Phase-4 monitors read `btc_1m`; Phase-5 friction uses `btc_5m`/`15m`; Phase-6 capital logic uses hourly rollups.

## Promotion rule

- Raw ticks stay gzipped.
- Summaries live in DuckDB/Parquet and are the only inputs to live gates.
- Cross-instrument joins happen **after** summarisation; this is where claims, contradictions, and influence tensors are defined.
- Influence tensors are computed over summarised feature vectors (e.g., correlations between `f_i(1m)` and `g_j(1m)`), never raw ticks.

## Next steps

1. Finalise the 1m schema and ephemerality rules for the `UNKNOWN` flag.
2. Extend the `tools/summarise_1s_to_1m.py` helper per the schema above.
3. Define the influence tensor tables/views and label their consumers.
4. Wire the gates to the correct summarised tables so `phase4_density`, `phase5_friction`, and `phase6_capital` have unambiguous inputs.

Structured summarisation is the missing glue—this note captures the “why, what, how” story so the work stays documented before the implementation rush begins.

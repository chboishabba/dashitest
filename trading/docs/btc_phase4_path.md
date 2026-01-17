# BTC Phase-4 Path (Data-Limited)

**Decision:** BTC remains a non-expressible tape for Phase-4. The gate is correctly *closed* because the substrate lacks forced behavior density. We will not lower thresholds or invent heuristics; instead we either raise the data quality or leave BTC in a conservative posture.

## Why this choice
* BTC proposals currently have T/R counts ≪ 120 and bins stay sparse even after persistence windows. The monitor deliberately reports `blocking_reason: T rows 2 < 120`. That is the **material substrate** issue, not a semantic failure.
* Replaying BTC at higher resolution or over longer horizons could eventually produce forced behavior, but we lack that data right now. In the absence of evidence, the controller should stay CLOSED and treat BTC as non-tradable for Phase-4.

## What we will do next
1. **Data refresh** – Capture higher-density BTC bars (maybe 1m/5m intraday) spanning >20 trading days. Ingest via `data_downloader.py` and re-run proposal generation so the gate sees new rows.
   * For example, reuse the existing `data/btc/close.csv` by resampling it to `1min` via:
     ```bash
     python data_downloader.py resample \
       --source data/btc/close.csv \
       --target data/btc/high_density/close.csv \
       --freq 1min
     ```
     Point the Phase-4 monitor at `data/btc/high_density/close.csv` so the gate ingests the denser tape before deciding if BTC can leave M4.
   * When real-time streaming matters, run the kline helper:
     ```bash
     python data_downloader.py stream-binance \
       --symbol BTCUSDT \
       --interval 1m \
       --out data/raw/binance_stream.csv \
       --duration-minutes 180 \
       --poll-interval 20 \
       --limit 500
     ```
     Or if you need real per-second bars, use the trade-based helper:
     ```bash
     python data_downloader.py stream-binance-trades \
       --symbol BTCUSDT \
       --out-dir logs/binance_stream \
       --duration-minutes 120 \
       --poll-interval 2 \
       --chunk-size-minutes 5
     ```
     That helper resamples aggTrades to 1s OHLC, writes gzipped chunks every few minutes, and keeps the latest chunk available for the gate without ever leaving a single monolithic log.
2. **Alternative posture** – Document this restriction as a convex-only track: if we ever revisit BTC we must treat it as a source of optionality (e.g., hex-convex scaling) rather than forcing directional Phase-4 weights.
3. **Monitoring** – Maintain `phase4_gate_status.md` and `phase4_density_monitor` logs for BTC; any change in gate status is a signal to reevaluate.

## Future decision points
* If higher-density BTC data arrives and the gate still blocks, escalate to a structural limit (watch for hazard/entropy collapse). Otherwise, when the gate reports OPEN twice, we can reapply the Phase-4 pipelines.
* Until then, treat BTC as `OBSERVE`/`UNWIND` only. No active size training or Phase-5 friction sweeps should include it without explicit data evidence.

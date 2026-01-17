# Stream Daemon Contract (TCP + NDJSON)

This daemon ingests an arbitrary-rate stream of market events over a TCP socket, aggregates to 1s OHLC when needed, and persists to DuckDB. It is always-on and should respond to backpressure by buffering and batching rather than dropping data.

## Transport

- TCP socket (default `0.0.0.0:9009`)
- Protocol: NDJSON (one JSON object per line)
- UTF-8, newline-delimited

## Input message schemas

### Trade (raw tick)

```json
{"type":"trade","symbol":"BTCUSDT","ts_ms":1700000000123,"price":95473.29,"qty":0.001}
```

Required fields:
- `type`: `trade`
- `symbol`: string
- `ts_ms`: integer (epoch millis)
- `price`: float
- `qty`: float

Behavior:
- Trades are aggregated into 1s OHLCV bars per symbol.
- Each second produces one `ohlc_1s` row when the second closes.

### OHLC 1s (already resampled)

```json
{
  "type":"ohlc1s",
  "symbol":"ES",
  "ts_ms":1700000000000,
  "open":5000.25,
  "high":5000.75,
  "low":4999.75,
  "close":5000.50,
  "volume":12.0,
  "trades":4
}
```

Required fields:
- `type`: `ohlc1s`
- `symbol`: string
- `ts_ms`: integer (epoch millis, second-aligned)
- `open`, `high`, `low`, `close`, `volume`: float

Optional fields:
- `trades`: integer (defaults to 0)

Behavior:
- Rows are inserted into `ohlc_1s` immediately.
- The daemon does not re-resample `ohlc1s` inputs.

## Output persistence

### DuckDB

Target table:

`ohlc_1s(symbol, timestamp, open, high, low, close, volume, trades, source_file)`

- `source_file` is set to `live:{symbol}` or the connection label for traceability.
- Ingest is batched for throughput.

### Summariser (optional)

When enabled, the daemon maintains a rolling close series per symbol and emits non-destructive summaries to:

`quotient_summaries` + supporting views (`legitimacy_stream`, `quotient_deltas`, `influence_gate_feed`)

The summariser never reads raw ticks directly; it consumes the 1s bar stream.

### Decision emission (optional)

When `--emit-decisions` is enabled, the daemon derives a triadic state per symbol from the live 1s bars, applies the Phase 6 exposure gate, and emits intents via the triadic strategy. Decisions are persisted to DuckDB and can optionally stream as NDJSON to stdout.

Tables:

- `stream_state(timestamp, symbol, close, state, gate_open, posture, source_file)`
- `stream_actions(timestamp, symbol, state, direction, target_exposure, urgency, hold, actionability, reason, gate_open, posture, source_file)`

Gate behavior:

- Gate uses `logs/phase6` by default and sets posture to `OBSERVE` when closed.
- Influence sensor inputs are read from `logs/asymmetry` if enabled.
- Decisions are derived per completed 1s bar (no intrabar emission).

### Decision cost gate (optional)

When enabled, a cost-aware consumer clamps decisions before they reach sinks.
It estimates the **notional** cost of changing target exposure using the last close:

```
cost = abs(delta_exposure) * price * (fee_rate + slippage)
```

Budgeted policy (rolling window):

- If the new cost would exceed the per-window budget, the decision is held at the prior target.
- Reductions in absolute exposure are always allowed (avoid trapping risk).
- Optionally require a minimum edge proxy (in bps) to cover estimated costs.

Flags:

- `--decision-fee-rate` (default `0.0005`)
- `--decision-slippage` (default `0.0003`)
- `--decision-edge-bps` (default `0.0`, disabled when `0`)
- `--decision-cost-budget` (default `0.0`, disabled when `0`)
- `--decision-cost-window` (seconds, default `60`)

### Decision sinks (optional)

Use `--decision-sink` to fan out emitted decisions. Supported sinks:

- `file:/path/to/decisions.ndjson` (append-only NDJSON)
- `tcp://host:port` (NDJSON stream to subscribers)

Delivery semantics:

- At-least-once delivery; consumers must be idempotent.
- Deduplicate on `run_id + timestamp + symbol` for sink payloads.
- `stream_actions_latest` is a DuckDB view of the latest action per symbol.

## Backpressure + failure handling

- If the producer outpaces the daemon, the daemon buffers per connection and processes in batches.
- Invalid JSON lines are logged and skipped.
- Missing required fields cause the line to be rejected; the daemon continues.
- On shutdown, the daemon flushes any partial OHLC buckets.
- Periodic flush ensures decisions persist even when no new bars arrive.

## Tail mode

Use `--tail` to replay existing NDJSON files through the same handler loop. `--tail-follow` keeps reading as new lines are appended.

## Metrics endpoint

Provide `--metrics-host` and `--metrics-port` to expose a lightweight HTTP endpoint that returns ingest counters, queue depth, and last-write timestamps.

## Example usage

Start daemon:

```
python scripts/stream_daemon.py --host 0.0.0.0 --port 9009
```

Send NDJSON:

```
printf '{"type":"trade","symbol":"BTCUSDT","ts_ms":1700000000123,"price":95473.29,"qty":0.001}\n' | nc localhost 9009
```

Enable decisions + metrics:

```
python scripts/stream_daemon.py --port 9009 --emit-decisions --metrics-host 127.0.0.1 --metrics-port 9100
```

Enable decision cost gate (budgeted):

```
python scripts/stream_daemon.py --port 9009 --emit-decisions \
  --decision-cost-budget 50 --decision-cost-window 60 --decision-edge-bps 0.8
```

Enable decision sinks:

```
python scripts/stream_daemon.py --port 9009 --emit-decisions \
  --decision-sink file:logs/decisions.ndjson \
  --decision-sink tcp://127.0.0.1:9200
```

Probe decision tables while the daemon is running:

```
python scripts/stream_daemon_test.py --port 9009 --seconds 2 --rate 200 --check-decisions --read-only
```

Plot decision exposure (NDJSON):

```
python scripts/plot_stream_decisions.py --ndjson logs/decisions.ndjson --out logs/decisions_plot.png
```

Add state markers + urgency thickness:

```
python scripts/plot_stream_decisions.py --ndjson logs/decisions.ndjson --show-state --urgency-thickness
```

Live follow (updates while daemon runs):

```
python scripts/plot_stream_decisions.py --ndjson logs/decisions.ndjson --follow --interval-ms 1000 --show-state --urgency-thickness
```

Dashboard semantics:

- Decision plots show **target exposure**, not executions.
- The dashboard renders target exposure as **stepwise** changes (no interpolation).
- Delta exposure is plotted separately to show churn and fee pressure.
- PnL uses **notional** fee/slippage estimates (price * exposure change).

WebM time-lapse instead of many PNGs:

```
python scripts/plot_stream_decisions.py --ndjson logs/decisions.ndjson --webm logs/decisions.webm --fps 10 --frame-step 5
```

Replay a raw stream:

```
python scripts/stream_daemon.py --port 9009 --tail logs/binance_stream/raw.ndjson --tail-follow
```

Live Binance feed into the daemon (short run):

```
python scripts/stream_daemon_live.py --port 9009 --symbols BTCUSDT,ETHUSDT --runtime 20 --db logs/research/market_live_binance.duckdb
```

Stream all USDT spot symbols (cap with --max-symbols to reduce load):

```
python scripts/stream_daemon_live.py --port 9009 --all-symbols --quote-asset USDT --max-symbols 10 --runtime 20
```

One command: live feed + live plot:

```
python scripts/stream_daemon_live.py --port 9009 --emit-decisions --symbols BTCUSDT,ETHUSDT --runtime 20 \
  --plot-follow --plot-show-state --plot-urgency-thickness --plot-posture
```

One command: live feed + WebM timelapse:

```
python scripts/stream_daemon_live.py --port 9009 --emit-decisions --symbols BTCUSDT,ETHUSDT --runtime 20 \
  --plot-webm logs/decisions.webm --plot-fps 10 --plot-frame-step 5
```

## Notes

- The daemon is designed to accept arbitrary input rates given sufficient compute.
- Raw ticks are never joined across instruments; joins happen on summarised quotient representations.

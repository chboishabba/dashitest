# Decision Alignment Check (Losing Trades)

Purpose: for each losing trade, find the closest profitable trade entry in historical logs based on
decision inputs (no price or data changes). This is a sanity check for regime alignment and
decision logic, not a search for alternate price outcomes.

Context references:
- TRADER_CONTEXT.md:4245 (epistemically correct but unprofitable)
- TRADER_CONTEXT.md:94517 (calm/surprising + unprofitable regime framing)

## What it does

The analysis script:
- reads the per-step log (default: `logs/trading_log.csv`)
- reads the trade-close log (default: `logs/trade_log.csv`) or falls back to `trade_closed` rows
- finds each losing trade entry (using close PnL from trade log)
- z-scores a conservative input-feature set using step-log statistics (stable even with few trades)
- returns the nearest profitable trade entry and prints that input snapshot

## Default feature set (conservative)

These are used only if present in the log:

```
p_bad, edge_t, edge_ema, stress, plane_abs, plane_sign, actionability,
acceptable, belief_state, thesis_depth, capital_pressure, permission, action_t
```

Missing columns are skipped; non-numeric features are mapped (e.g., belief_state) or encoded.

## Usage

Run from this directory with `PYTHONPATH=.`:

```
PYTHONPATH=. python scripts/compare_trade_alignment.py \
  --log logs/trading_log.csv \
  --trade-log logs/trade_log.csv \
  --top-k 1
```

Optional:
- `--features` override the default list (comma-separated).
- `--top-k` prints the top N closest profitable entries per losing trade.

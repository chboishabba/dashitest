# Phase-4/Phase-5 Strategy Notes

This note captures the calibration you asked for: a tighter Phase-4 gate, a candidate instrument list, and a concrete Phase-5 execution plan. It ties directly back to the verdicts collected in `TRADER_CONTEXT2.md` (see lines 13562‑14160 for the Phase-4 refusal rationale and Stage-4 gating requirements).

## Tightening the Phase-4 density monitor gate

1. **Per-ontology balance** – once `T+R ≥ 200`, require `min(T,R) ≥ 25%` of the total so neither ontology monopolizes the learning signal.
2. **Bin persistence** – treat binary `size_pred` windows as a rolling grid; each bin must reappear in ≥3 disjoint tape segments (e.g. quantize the proposals into 5 windows) before it counts toward the gate. This guards against single bursts satisfying the count thresholds.
3. **Effect-size floor** – bins must not only be ordered, they must differ by ≥3 bps in their median forward proxy (use clipped log returns as the monitor already logs). This enforces a meaningful ranking rather than collateral noise.
4. **Rolling confirmation** – the gate only reports `OPEN` when two consecutive checks passed; the script still logs every run, but the final `gate_open` value reported in the log reflects the debounced signal.

These tightenings keep Phase-4.1 conservative exactly the way `TRADER_CONTEXT2.md` recommends: it refuses to train unless the upstream data truly justifies it.

## Phase-4.1 entry checklist

Before any Phase-4.1 training run, confirm the canonical `strict` profile has produced a clean `OPEN` verdict; nothing else triggers Phase-4.1. The readiness checklist is:

1. Run `scripts/phase4_monitor_runner.py --profile strict --config configs/phase4_monitor_targets.json` so the `strict` bundle defined in `configs/phase4_monitor_profiles.json` guards every gate parameter.
2. Both ontologies report ≥120 rows (after `--min_total_tr` is satisfied) and no single ontology exceeds 75% of the joint count, guaranteeing diversity.
3. At least three bins meet `min_rows_per_bin=40`, `min_bin_balance=0.35`, and maintain `min_monotone_bins=2`, so the learner sees actual size structure.
4. Each qualifying bin is present in six of the last eight runs (`bin_persistence_window=8`, `bin_persistence_required=6`) without flipping its median sign more than once (`median_flip_max=1`).
5. The effect-size floor (`min_effect_size=0.0005`) is still respected, which keeps Phase-5 future costs meaningful.
6. The gate has reported `OPEN` three times in a row (`consecutive_passes=3`), ensuring the signal is stable rather than a fluke.

If any bullet still fails, waiting is not a bug—it is the correct behavior. Log the blocking reasons, continue collecting ES/NQ density, and only initiate Phase-4.1 when every check shines.

## Interpreting Phase-4 `OPEN` verdicts

The strict profile is a **mechanical gate** that only reports `OPEN` once density, persistence, and effect-size checks all converge. Because it exercises the entire stack, it is useful to validate the monitor, but the gate does not know whether the inputs came from market density or a test vector.

* If you artificially inject amplitude (forward-return separation or per-bin boosts) to verify the path, treat that run as a **mechanical validation** only. The maintainers should log a clear `test_vector=synthetic_amplitude` marker (for now, note it manually in `phase4_gate_status.md` or the monitor log) so the subsequent Phase-4.1 run is never confused for “real” readiness.
* A genuine OPEN verdict still requires **real proposal rows, aligned with real BTC/ES/NQ prices, meeting the row floors and effect-size requirements** noted above. No threshold loosening, exception, or synthetic funding path should be relied upon to claim readiness.
* If the monitor transitions to OPEN and you are unsure whether density was organic, recheck `scripts/check_esnq_ingestion.py` (or its BTC cousin) to confirm the contract and replay the last few `logs/phase4/density_monitor/*.json` payloads around the transition; the payloads now include `blocking_reason`/`gate_open` history so you can see exactly which test vector cleared last.

Because the gate already records the `phase4_gate_status` per iteration, keep using those logs to trace each condition. If you want stronger guardrails later, schedule an explicit TODO entry that elevates synthetic markers from notes to automated checks.

## Candidate density sources

1. **ES/NQ intraday futures (5‑15m bars, ≥10 trading days)** – best bet. Long trends, structured volatility, and consistent size relevancy keep both ontology and size choices alive. Collect the same column set as BTC proposals (ontology, action, size_pred, index) so the monitor can reuse the existing interface.
2. **Major FX spot (EURUSD, USDJPY)** – stable regimes and low hazard mean T/R rows show up even without big moves. Adjust the effect-size floor down if the forward proxy magnitudes shrink relative to BTC.
3. **Volatility products (VX futures, short-dated SPX options)** – good for R-focused tapes; size matters intrinsically here and the monitor can run on synthetic proposals until real data is ready.

Start with ES/NQ (optionally run Phase-3 / Phase-4 diagnostics on those tapes first) and keep using the same `proposals.csv` + `prices.csv` contract, so the `phase4_density_monitor.py` script continues to work without change.

## Phase-5 execution simulator outline

*Phase-5 is strictly downstream; it never feeds back into Phase-4 gating or the ontology decisions.*

1. **Phase-5.0 (deterministic friction)** – derive `fill_price = price_t × (1 ± slip_bps)` and `cost = size × fee_bps` for each actable proposal. Log `execution_cost`, `slippage_cost`, and `realized_pnl` but do not alter `size_pred` or `direction` decisions.
2. **Phase-5.1 (size-dependent friction)** – add `slip_bps = base + k_size × size_level` so larger bins pay progressively more, without introducing randomness. This keeps Phase-5 purely observational yet exposes how size choices interact with cost.
3. **Phase-5.2 (bounded noise)** – optionally sample slippage within a tight, bounded distribution (e.g. `Uniform(base, base + k × size)`), which again only affects reporting.

Because Phase-5 consumes proposal logs and writes an execution log (e.g. `logs/phase5/{target}-{timestamp}.json`), you can run it alongside Phase-4 monitoring without altering the `--monitor-log` or `train_size_per_ontology.py` behavior.

`scripts/phase5_execution_simulator.py` implements the deterministic + size-proportional friction described above. It loads the same proposal/prices contract, applies slip/fee basis points, and writes a timestamped JSONL log with `realized_pnl`, `slippage_cost`, and `execution_cost` for every ACT decision.

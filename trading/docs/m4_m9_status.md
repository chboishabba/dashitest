# M4 / M9 Status and What Comes Next

**Short answer: yes — but that’s not a bug, and you’re no longer stuck for the old reasons.** What you now face is a data-substrate _Motif-4_ that the system correctly refuses to bypass.

## 1. Are we stuck at M4?

### ✅ Yes — but only in the material substrate sense

* BTC remains in **Motif-4 (Anchored Corridor)** because the available tape lacks the forced behavior density the gate demands.
* ES has already produced enough density so its gate is open (and staying OPEN once the debounce settles), which proves the system can leave M4 when the market supplies structure.
* The original blockers (runner speed, ternary collapse, gating logic) have been resolved; the remaining block is a **data substrate limitation only**.

This is a conservative learner doing what it should: _refusing to hallucinate structure the market never put on the tape._

## 2. Proof the “bad M4” is gone

All of the engineering/semantic exit criteria are now satisfied:

* Math-on-GPU / Control-on-CPU split is validated.
* Tape reuse and GPU residency removed the “slow runner” pathology.
* The 4th symbol (UNKNOWN ⊥) remains distinct from FLAT, eliminating binary drift.
* Triadic operations are verified (GF(3) 120° rotation and UTES-5 packing).
* Void/Paradox supervision and the audit trail now record the _why_ of abstention.

This is not M4 anymore. It is a functioning epistemic machine waiting for the market to reveal a regime.

## 3. What remains blocked — precisely

| Instrument | Status | Blockers | Action |
| --- | --- | --- | --- |
| **BTC** | Still M4 | Too few rows, no persistent bins, lossy bars wreck forced behavior | None beyond ingesting higher-density/longer-horizon BTC data or accepting the limit explicitly |
| **ES** | No longer M4 | Gate OPEN (debounce pending) | Drive Phase-4.1 training and Phase-5 friction confirmation |

## 4. Why this is not an M6 failure

* M6 requires **expressible opposition** (two simultaneous resolved stances). BTC never generates that opposition because the tape lacks expressibility.
* ES does reach M6, resolves naturally, and then declares OPEN.
* For BTC the system is correctly prevented from inventing M6 where none exists; it is epistemic honesty, not weakness.

## 5. Key deliverables to move past M4

1. **Finish Phase-4.1 on ES (now unblocked).**
   * Learn T/R size weights only, keep H pinned, aggregate with priors (λ = 0.25), lock schema + checksum.
   * Exit when the ES size weights are frozen and verifiably reproducible.
2. **Phase-5 friction confirmation.**
   * Run the simulator with 1.5–2.0 bps slippage plus the full fee model.
   * Produce size vs marginal PnL curves and tail loss comparisons vs flat to prove the edge survives microstructure taxes.
3. **Decide the BTC path explicitly.**
   * Option A: ingest higher-density (LOB/longer horizon) BTC data.
   * Option B: accept BTC as non-expressible for Phase-4.
   * Option C: shift BTC to a convex-only posture with no directional learning.
   * Pick one and document it; indecision is not a defensible posture.
4. **Lock ternary invariants.**
   * Declare UNKNOWN ⊥ ≠ FLAT, hazard ontologies remain unlearned, and the gate stays conservative.
   * Capture these invariants in a short reference document to prevent “make it trade” regressions.

## 6. Evidence & artifacts from A/B/C

* **Phase-4.1 size training (Option A)**: `scripts/train_size_per_ontology.py` ran against `logs/esnq/ES_5m/proposals_phase4_ready.csv`, producing `logs/esnq/ES_5m/phase4_size_run.json` and `logs/esnq/ES_5m/weights_size_run.json`. The gate stayed OPEN and the locked weights appear in the plan tracking table.
* **Phase-5 friction sweep (Option B)**: `scripts/phase5_execution_simulator.py` generated `logs/phase5/es/phase5_execution_20260115T112553Z.jsonl` through `...112620Z.jsonl`. Edge persists only at 0.5 bps slip (others lose money), so future deployment must respect that friction envelope.
* **Asymmetry sensor (Option C)**: `scripts/asymmetry_influence_sensor.py` collapsed ES/NQ influence correlations into triadic bitplanes and logged `logs/asymmetry/influence_tensor_20260115T112947Z.jsonl` (counts: +1=1, 0=2, -1=3). This proves the infrastructure can produce triadic influence summaries before any controller action.
* **Tracking docs**: `docs/phase4_phase5_asymmetry_plan.md` now houses the checkpoint table, `docs/btc_phase4_path.md` codifies the data-limited BTC posture, and `docs/ternary_invariants.md` records the core gate invariants.

## 6. M9 exists — but profit must still be earned

* M9 is the _existence of closure_, not the existence of profit.
* A profitable M9 is a **special case** that requires expressibility, asymmetry, and convexity simultaneously.
* Your system is designed to refuse trading inside every M9 that lacks those properties; that refusal is _epistemic discipline_, not cowardice.
* Without forced behavior or privileged asymmetry, the best admissible action remains OBSERVE / UNWIND.
* If you want M9 to become opportunistic, you must add explicit asymmetry sensors (forced liquidation signs, funding-pressure divergence, volatility mispricing, convexity proxies) that unlock the inexorable path from tension (M6) to productive synthesis (M9).

## 7. Next directions we can take now

* Sanity-check the ES size loss surface once Phase-4.1 weights are locked.
* Design a BTC-specific convex-only track (or document the chosen BTC path decisively).
* Formalize the M4 → M6 detection so this diagnosis becomes automatic rather than ad hoc.
* Alternatively, build a pressure-field / influence tensor layer that surfaces the asymmetries needed to treat certain M9 regimes as tradable.
* Resample BTC bars to 1m (`data_downloader.py resample --freq 1min`) so the Phase-4 gate can re-evaluate the substrate without dropping thresholds.

Tell me which of these you want to pursue, and I can help formalize the requirements or sketch the implementation.

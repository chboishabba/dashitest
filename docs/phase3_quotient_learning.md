# Phase 3: Quotient-first training

This file records the Phase-3 breakthrough described around `CONTEXT.md#L39825-L40250`: no more chasing observers, just optimize on the invariants (the quotient `V`) while treating everything else (`U`) as gauge.

## Guiding principle

1. **Define the quotient explicitly.**  
   * `U` is the learner-internal schedule/plan/gate randomness.  
   * `V` is the semantic output you actually care about (the snapshot frame/energy map you can log).  
   The goal is to optimize using only `V` plus a gauge penalty, as spelled out in `CONTEXT.md#L39825-L39998`.

2. **Plan-equivalence objective (high ROI, minimal disruption).**  
   * Run the same input under two regimes: the current plan `r₁` and a canonical reference plan `r₀`.  
   * Use `V(r₁)` for the task loss and penalize `|V(r₁) - stopgrad(V(r₀))|²` to collapse regime directions (`CONTEXT.md#L40120-L40220`).  
   * Keep a cheap MDL-style gauge cost (active tiles, plan changes, cache hits) so the learner prefers stable gauges.

3. **Label entropy check for observer experiments.**  
   * Before running Stage B tests, ensure label distributions have enough support (`CONTEXT.md#L39890-L39920`).  
   * For plan-phase: raise `--plan-stable-length` or merge the `-1` class into `0` so you avoid 63/64 imbalances.  
   * For cache-hit: lower the number of bins or the thresholds so each bin is populated.  
   These steps keep blocked-permutation tests informative instead of degenerate.

4. **Logging checklist for Phase 3 runs.**  
   * Track `task_loss`, `quotient_loss` (`|V₁ - V₀|²`), `mdl_cost`, `plan_hit_rate`, and canonical-plan statistics.  
   * Report when `Phase 3` is activated and how `alpha`/`beta` evolve, per the warm-up schedule discussed around `CONTEXT.md#L40130-L40230`.

## Timestamped run artifacts

Each Phase-3 invocation automatically persists its diagnostics so you can compare runs later:

1. A JSON log saved under `logs/bsmoe_train/bsmoe_phase3_<timestamp>.json` that records per-epoch `{epoch, task_loss, quotient_loss, mdl_cost, alpha, plan_hit}` plus the UTC timestamp of the run.
2. A summary graph plotted to `outputs/bsmoe_phase3_<timestamp>.png` that overlays the three main losses so you see the quotient binding progress at a glance.

Timestamping ensures you never overwrite earlier diagnostics and matches the “timestamped output” pattern referenced in `docs/energy_landscape_vis.md` and `CONTEXT.md#L39995-L40215`.

## Normalized latitude for the quotient

`V` is a geometric object, not a raw energy meter, so keep it normalized. The minimal change that keeps the quotient interpretable is to map the tile-energy map through a logarithmic or mean-rescaling filter before measuring `|V₁ - V₀|²`. In practice we compute `V = log1p(tile_energy)` (with matching logic on both `C₁` and `C₀`) so the `quotient_loss` stays in a reasonable scale even though `tile_energy` grows quartically with `C`. This normalization converts the invariant into a bona fide chart while preserving the Phase-3 story from `CONTEXT.md#L39825-L40797`.

## Minimal quotient gradient (Phase 3 + VJP)

The Phase-3 loss is still compatible with little gradient surgery: once `V₁` and `V₀` are normalized, backpropagate only through the invariant difference. A lightweight vector-Jacobian-product multiplies `delta_V = V₁ - V₀` with each tile’s `C₁` block (and a small constant) and adds that correction to the existing `task` error. The result is a surgical gauge push that honors the MDL objective without reshaping the learner architecture; the recipe generalizes the “stop trying to decode” narrative in `CONTEXT.md#L39825-L40797` to an active gradient signal.

## Workflow summary

1. Implement the canonical plan (`r₀`) and compute its invariant once per run; treat it as a `stopgrad` target.  
2. Keep the existing task loss (e.g., drive `C` toward zero) and add the quotient penalty plus gauge cost with independent weights.  
3. Run the same training script; expose `--phase3` plus `--phase3-alpha`, `--phase3-beta`, and a warm-up flag so you can ramp up the quotient term gradually.  
4. Keep Stage B alive if desired, but stop chasing tiny gains and log the quotient metrics instead.

Applying this setup keeps your claims honest: you’re no longer “lightning in the ladder,” you’re just **driving the learner along the quotient itself** (`CONTEXT.md#L39998-L40210`).

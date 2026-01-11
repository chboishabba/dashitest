# Stage B2 Acceptance Criteria

This note records the Stage B protocols we must follow when probing the tiny CNN observer (B2) so we stay honest about blocked permutations, entropy guards, and motif attribution. The surrounding research rationale and observer-ladder context are in `CONTEXT.md#L36915-L38950`.

## Definitions

1. Let **K** be the number of classes/bins (e.g., `gate-density-bins=3` ⇒ K=3); **N** is the number of evaluated blocks in Stage B.  
2. Track block-level accuracy (`A_true`) and balanced accuracy (`BA_true`) to guard against frequency drift.  
3. For each permutation `i` (blocked permutation, retrain-from-scratch, block-aggregated evaluation) record `A^{(i)}_π`.  
4. Empirical p-value is  
   ```math
   p = (1 + Σ_{i=1}^P 𝟙[A^{(i)}_π ≥ A_true]) / (1 + P)
   ```  
   (per the conservative “+1” rule and textbook blocked-permutation design in `CONTEXT.md#L36915-L37220`).

## Primary success criterion

B2 “lights up” only if both hold:

- **Permutation significance**: p ≤ 0.01.
- **Effect size**:  
  - Binary (K=2): `BA_true ≥ 0.60`.  
  - Multi-class: `BA_true ≥ 1/K + 0.10` (e.g., K=3 ⇒ BA ≥ 0.433; K=5 ⇒ BA ≥ 0.30).  

These thresholds keep us from “winning by vibes” while staying within the bounded observer philosophy described around `CONTEXT.md#L38215-L38280`.

## Primary failure (Φ-indistinguishable) zone

Declare B2 failed if:

- p ≥ 0.20 **and**
- `BA_true ≤ 1/K + 0.03`.

This is the “no signal left, even with capacity” zone referenced in `CONTEXT.md#L38245-L38267`. Treat that outcome as strong evidence for Φ-indistinguishability at B2.

## Ambiguous zone procedure

When the run is neither a success nor a failure:

1. Increase permutations (e.g., P=500 → 2000) without altering the architecture or data leaks.  
2. Increase the number of blocks / run length under the same observer.  
3. If still inconclusive, consider **B2+** knobs (gradients, Δ/∇ channels, or blockwise temporal pooling) as a *new rung* with the same acceptance/failure thresholds (`CONTEXT.md#L38661-L38691`).  

Follow this sequence faithfully rather than tweaking learning rates or adding channels mid-run.

## Guardrails

- Blocked permutations, retraining from scratch, and block-aggregated evaluation are mandatory (`CONTEXT.md#L37047-L37210`).  
- Skip the run entirely if the label entropy guard says there is insufficient variation.  
- No non-causal leakage (plan metadata, gate density, or cache hits) may influence the observer—only raw frame content.  
- Never evaluate per frame; always aggregate over blocks before scoring.

## Motif attribution (only if B2 lights up)

1. **Block-level occlusion** (preferred): freeze the trained CNN, occlude patches from the block-aggregated input, and measure the drop in the correct-class logit. Average Δs over held-out blocks and produce spatial/per-class heatmaps (`CONTEXT.md#L38691-L38720`).  
2. **Integrated Gradients / Grad-CAM** (optionally): compute attributions on pooled block inputs for held-out blocks only, report averages, and validate with occlusion.  
3. **Scale-separation ablation**: build a Laplacian pyramid of the block input and evaluate coarse-only, residual-only, and full inputs to test hypotheses about p-adic/multiscale structure (`CONTEXT.md#L38736-L38800`).

## Logging checklist

For every Stage B run (B2 or B2+):

- K and evaluated block count N.  
- Class frequencies over blocks.  
- `BA_true` and `A_true`.  
- Permutation null summary (mean, std, max).  
- p-value (per the formula above).  
- Block-level confusion matrix.  
- If B2 lights up: occlusion heatmaps and any scale-separation results.  

These records match the logging expectations described near `CONTEXT.md#L38800-L38950`.

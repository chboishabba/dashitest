# Gray-Scott quotient metrics (plan)

Purpose: define quotient-space evaluation (and optional training loss) for the
Gray-Scott KRR benchmark so the learner is judged on invariants rather than
raw field coordinates.

Source context: see CONTEXT.md lines 5435-5585 for the quotient motivation,
options, and the minimal "learn the quotient" step.

## Quotient choice (decided)
- **Chosen:** V + radial(U).
- **Not chosen:** V-only (too lossy), V + spectrum(U) (too abstract for Phase 2).

## Metrics to log (quotient distance)
- MSE_V: MSE on V.
- MSE_U_radial: MSE on radial_avg(U).
- MSE_U_spec_low: MSE on low-frequency spectral coefficients (optional).
- Mass/mean drift: conserved-quantity deviation over rollout.

## Minimal implementation notes
- Add quotient metrics to rollout CSV and summary logs.
- Keep full-field plots for diagnostics, but do not optimize against them.
- Decide the quotient choice before changing loss/selection logic.

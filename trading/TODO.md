# Trading TODO

- [x] Add plane counters to trading logs (plane index + per-plane activation/rates).
- [ ] Replace confidence thresholds with explicit ternary actions (intent/exec alignment).
- [x] Define and log a "can trade" mask (trade eligibility separate from direction).
- [x] Add MDL score panel to the training dashboard.
- [x] Plot plane rates in the PG dashboard (with optional normalization/log scale).
- [x] Add edge_ema metric + optional cap gate.
- [x] Add bounded thesis memory (thesis_depth) to delay soft-veto exits.
- [x] Add goal/MDL config to CLI (goal cash, deadline, tax rate, MDL params).
- [ ] Add edge_ema panel (or overlay) to the PG dashboard.
- [ ] Plot stress histogram pane in PG dashboard.
- [ ] Implement explicit baseline simulation for regret (sell-all-at-t0 path).
- [ ] Enable plane-aware strategy selection (MDL selector over strategies).

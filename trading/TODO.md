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
- [x] Guard training_dashboard_pg.py against trade-log selection (auto-select per-step log or show clear error).
- [ ] Implement explicit baseline simulation for regret (sell-all-at-t0 path).
- [ ] Add realized-volatility regime feature (continuous + ternary regime flag) and log it.
- [x] Implement minimal thesis-memory state machine (direction/strength/age/cooldown/invalidation) with logging.
- [x] Add benchmark-regret reward option vs constant exposure (optional vol-adjusted penalty).
- [ ] Condition decision simplex on thesis state (split by thesis_d/thesis_s).
- [ ] Split UNKNOWN vs FLAT in belief logging (distinct belief_state).
- [ ] Enable plane-aware strategy selection (MDL selector over strategies).

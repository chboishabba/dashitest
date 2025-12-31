TODO
====

- [x] Implement ternary controller and log ternary state fields.
- [x] Replace binary ban/can_trade with ternary permission.
- [x] Sanity check run_trader with `--max-steps` and `--all`.
- [x] Add `--max-trades` early stop to the trading loop.
- [x] Add thesis memory counter to hold positions across soft vetoes.
- [x] Document edge-gated `run_trader.py --all` sanity test outcomes in README.
- [x] Document buy-and-hold degeneracy and minimal thesis-memory extension.
- [x] Link buy-and-hold doc to related context in `TRADER_CONTEXT.md`.
- [ ] Implement bounded thesis-depth persistence per `docs/buy_hold_degeneracy.md` (state, transitions, exits).
- [ ] Add CLI/config for thesis depth max (default + validation).
- [ ] Log thesis depth per-step and per-trade; update field list in README if new columns are added.
- [ ] Add a focused sanity run note or test case covering soft-veto hold behavior.
- [ ] Update dashboards to display ternary state columns (permission, edge_t, capital_pressure, action_t).

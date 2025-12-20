## Current Trading Stack (Dec 20)

- **Architecture:** `state → TriadicStrategy → Intent → Execution → Log → Dashboard`.
  - `Intent`: `direction∈{-1,0,+1}`, `target_exposure`, `hold`, `actionability` (logged), `urgency`, `ttl_ms`.
  - **Execution:** `BarExecution` respects `hold` (no-op), otherwise moves exposure toward target with slippage/fees and a min-trade threshold. `Runner` marks-to-market equity each bar, routes to `BarExecution` for bars; `HFTExecution` is stubbed for future LOB replay.
  - **Logs:** CSV includes `t, price, pnl, action, hold, exposure, actionability, acceptable, urgency, fee, slippage, equity` (dashboard-friendly).

- **Strategy (TriadicStrategy):**
  - Input state `{−1,0,+1}`.
  - Alignment ramp (slow) with `base_size=0.05`.
  - Confidence hook (`conf_fn` → `[0,1]`) with hysteresis thresholds `tau_on > tau_off` (asserted). Emits `hold=True` when state==0 or confidence below thresholds.
  - Actionability is logged; hysteresis uses internal `_is_holding`.

- **State source:** `run_trader.compute_triadic_state(prices)` (rolling std of returns w=200, EWMA of normalized return, dead-zone) → triadic state array. Used by scripts.

- **Confidence proxy (scripts/run_bars_btc.py):** persistence-based:
  - `run_scale` from consecutive same-direction run-length (tanh), zeroed on flips/zeros.
  - Vol penalty: rolling vol (50-bar std) above 80th percentile zeroes confidence.
  - Passed to strategy via `conf_fn`; script default `tau_on=0.5, tau_off=0.3`.

- **Regime predicate:** `regime.py` with `RegimeSpec` (defaults `min_run_length=3`, `max_flip_rate=None`, `max_vol=None`, `window=50`). `check_regime(states, vols, spec)` returns `acceptable` bool; flip-rate/vol optional. Runner computes vols (50-bar rolling std of returns) and logs `acceptable`.

- **Scripts:**
  - `scripts/run_bars_btc.py`: loads `data/raw/stooq/btc_intraday.csv`, computes state via `compute_triadic_state`, confidence via persistence/vol, runs `run_bars` with hysteresis, writes `logs/trading_log.csv`. Fish-safe: `PYTHONPATH=. python scripts/run_bars_btc.py`.
  - `scripts/sweep_tau_conf.py`: sweeps hysteresis width only (fixed `tau_on`, varying `tau_off` default `{0.30,0.35,0.40,0.45,0.25,0.20,0.15}`), logs `acceptable%`, precision `P(acceptable|ACT)`, recall `P(ACT|acceptable)`, act_bars (engaged bars), HOLD%. Optional `--out` writes PR curve for dashboard sparkline. Stops early if precision < precision_floor (default 0.8).
  - `scripts/sweep_motif_hysteresis.py`: motif CA hysteresis sweep (softmax confidence for constructive state → tau_on/tau_off, optional k_on/k_off). Reports acceptable%, precision/recall, act_bars, HOLD%; optional `--out` CSV.
  - `scripts/plot_engagement_surface.py`: plots heatmaps of engagement surfaces (`tau_off` × actionability bin → engagement) from `*_surface.csv` produced by sweep scripts; supports side-by-side trader vs motif comparison.
  - `scripts/plot_acceptability.py`: heatmap of acceptable density over (time × actionability) from a single trading log.
  - `scripts/sweep_regime_acceptability.py`: sweep RegimeSpec parameters (min_run_length, optional vol cap percentile) to map acceptable%; outputs CSV.
  - `scripts/plot_regime_surface.py`: heatmap of acceptable% over RegimeSpec sweeps (from the CSV above).

- **Recent results (defaults):** BTC intraday bars with persistence confidence & loose RegimeSpec:
  - trades ≈ 138, HOLD ≈ 0.686, acceptable% ≈ 0.831
  - P(acceptable | ACT) = 1.0, P(ACT | acceptable) ≈ 0.0216
  - pnl drift ≈ −0.0026 (bar-level costs dominate; gate now non-degenerate)

- **Stubs / future:** `execution/hft_exec.py` stub for LOB replay (BTC/ETH routed by runner). No state learning/ML yet. MoE/soft gating deferred.

- **User prefs / safety:** Keep fish-safe commands; HOLD must stay epistemic (not flatten). Regime/acceptable is PnL-free; use precision/recall of acceptable vs ACT for diagnostics. Hysteresis maintained.

- **Dashboard:** `training_dashboard.py` shows equity/latent/price; optional `--pr logs/pr_curve.csv` adds a tiny tau_off → (precision, recall) sparkline (from sweep script).

## Latest diagnostics (Dec 21)

- Engagement sweep: acceptable% invariant at 0.831; precision=1.0 across tau_off ∈ {0.45…0.15}; recall rises smoothly (0.305→0.493); engagement surface exported (`surface.png` trader vs motif).
- Acceptable manifold: time×actionability heatmap (`acceptable.png`); RegimeSpec sweep (min_run_length × vol cap × flip cap) with acceptable% (`regime_surface.png`, `logs/accept_surface.csv`); legitimacy margin heatmap (distance to RegimeSpec failure) (`margin.png`).
- Confusion surfaces: false-positive/false-negative density over (actionability × tau_off) (`confusion.png`, `logs/confusion_surface.csv`); FP stays ~0, FN surface shows missed engagement regions.
- Engagement overlay: dashboard plots P(ACT|actionability, acceptable) vs P(ACT|actionability, unacceptable) to catch legitimacy leakage (red curve pinned at ~0 when healthy).

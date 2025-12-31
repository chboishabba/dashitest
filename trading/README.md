# Trading

Minimal triadic trading research sandbox with synthetic/real data runs, dashboards, and analysis scripts.

## Quick start commands

Run from this directory:

- `python run_trader.py` to execute the core loop on cached data (or synthetic fallback).
- `python run_all.py` to run across all cached markets.
- `python training_dashboard.py --log logs/trading_log.csv --refresh 0.5` for a live matplotlib view.
- `python training_dashboard_pg.py --log logs/trading_log.csv --refresh 1.0` for the PyQtGraph dashboard.
- `python ternary_trading_demo.py` for a self-contained demo with synthetic data.

From this directory (or from the parent, add `trading/` prefixes to script/data paths), use `PYTHONPATH=.`:

```bash
PYTHONPATH=. python run_all_two_pointO.py \
  --markets --market-progress-every 500 \
  --csv data/raw/stooq/btc_intraday_1s.csv \
  --live-sweep --run-ca --ca-report-every 1000 \
  --emit-news-windows
```

## Code map (root)

- `__init__.py`: Package marker to allow `import trading` when running from the parent dir.
- `base.py`: Abstract execution backend contract (inputs: intents, outputs: fills + summary).
- `bar_exec.py`: Bar-level execution model with simple slippage/fees and exposure tracking.
- `hft_exec.py`: Stub LOB replay execution adapter for future hftbacktest integration.
- `intent.py`: Immutable `Intent` dataclass emitted by strategies and consumed by executors.
- `regime.py`: Regime acceptance utilities (run-length, flip-rate, vol gates).
- `runner.py`: Strategy/executor glue; builds bar dataframes and logs execution traces.
- `run_trader.py`: Core trading loop, triadic state computation, stress/bad-flag logic, log writer.
  - `python run_trader.py --all` runs all CSVs under `data/raw` sequentially (per-file logs).
- `run_all.py`: Multi-market runner (with optional live dashboard).
- `run_all_two_pointO.py`: Orchestrator for market summaries, tau sweeps, CA tape preview, and news windows.
- `data_downloader.py`: Data ingestion for Stooq/Yahoo/CoinGecko/Binance; writes `data/raw`.
- `ternary_trading_demo.py`: Self-contained demo; encodes ternary signals and compares baseline.
- `training_dashboard.py`: Matplotlib dashboard for `logs/trading_log.csv`.
- `training_dashboard_pg.py`: PyQtGraph dashboard for `logs/trading_log.csv` with progressive-day view.
- `test_trader_real_data.py`: Sanity check that cached Stooq data is used over synthetic fallback.
- `TRADER_CONTEXT.md`: Research and theory notes (large; not required for running code).
- `data/`: Cached datasets and run history (`data/raw`, `data/run_history.csv`).
- `logs/`: Output logs, dashboards, and news windows.
- `scripts/`: Analysis, plotting, and sweep utilities (see catalog below).
- `strategy/triadic_strategy.py`: Deterministic triadic strategy that emits `Intent`.
- `strategy/__init__.py`: Strategy package marker.

## Core runtime flow

1) `run_trader.run_trading_loop` loads prices, computes triadic states, runs the trading loop, and writes logs.
2) `strategy.triadic_strategy.TriadicStrategy` turns state into `Intent`.
3) `bar_exec.BarExecution` (or `hft_exec.LOBReplayExecution`) executes intents and returns fills.
4) `training_dashboard.py` / `training_dashboard_pg.py` visualize `logs/trading_log.csv`.

## Script catalog (analysis + plotting)

Run from this directory with `PYTHONPATH=.` to avoid import issues.

- `scripts/ca_epistemic_tape.py`: CA tape visualization driven by price series.  
  Command: `PYTHONPATH=. python scripts/ca_epistemic_tape.py --csv data/raw/stooq/btc_intraday_1s.csv`
- `scripts/compute_policy_distance.py`: Compare engagement policy geometry between two logs.  
  Command: `PYTHONPATH=. python scripts/compute_policy_distance.py --a logs/trading_log.csv --b logs/trading_log.csv`
- `scripts/contextual_news.py`: Fetch contextual news/stress signals (Reuters RSS, TradingEconomics, yfinance).  
  Command: `PYTHONPATH=. python scripts/contextual_news.py` (prints signals for today; use `contextual_events()` in code for custom dates)
- `scripts/emit_news_windows.py`: Find bad windows and fetch news (GDELT/NewsAPI/RSS).  
  Command: `PYTHONPATH=. python scripts/emit_news_windows.py --log logs/trading_log.csv --out-dir logs/news_events`
- `scripts/news_slice.py`: Fetch news headlines for a time slice (NewsAPI/GDELT/RSS).  
  Command: `PYTHONPATH=. python scripts/news_slice.py --provider gdelt --start 2025-03-01T00:00:00Z --end 2025-03-01T23:59:59Z --out logs/gdelt_slice.csv`
- `scripts/plot_acceptability.py`: Heatmap of acceptable density over time × actionability.  
  Command: `PYTHONPATH=. python scripts/plot_acceptability.py --log logs/trading_log.csv --save logs/acceptable.png`
- `scripts/plot_accept_persistence.py`: Heatmap of acceptable run-length persistence.  
  Command: `PYTHONPATH=. python scripts/plot_accept_persistence.py --log logs/trading_log.csv --save logs/accept_persistence.png`
- `scripts/plot_action_entropy.py`: Action entropy over actionability/margin bins.  
  Command: `PYTHONPATH=. python scripts/plot_action_entropy.py --log logs/trading_log.csv --save logs/action_entropy.png`
- `scripts/plot_confusion_surface.py`: Plot FP/FN surfaces from sweep CSV.  
  Command: `PYTHONPATH=. python scripts/plot_confusion_surface.py --csv logs/confusion_surface.csv --save logs/confusion_surface.png`
- `scripts/plot_direction_legitimacy.py`: Legitimacy by direction/actionability.  
  Command: `PYTHONPATH=. python scripts/plot_direction_legitimacy.py --log logs/trading_log.csv --save logs/direction_legitimacy.png`
- `scripts/plot_engagement_surface.py`: Engagement surface heatmap from sweep outputs.  
  Command: `PYTHONPATH=. python scripts/plot_engagement_surface.py --left logs/engagement_surface.csv --save logs/engagement_surface.png`
- `scripts/plot_first_exit_heatmap.py`: Heatmap of first-exit behavior vs actionability.  
  Command: `PYTHONPATH=. python scripts/plot_first_exit_heatmap.py --log logs/trading_log.csv --save logs/first_exit.png`
- `scripts/plot_fn_anatomy.py`: Diagnostics on false-negative structure vs features.  
  Command: `PYTHONPATH=. python scripts/plot_fn_anatomy.py --log logs/trading_log.csv --save logs/fn_anatomy.png`
- `scripts/plot_hysteresis_phase.py`: Hysteresis phase plot vs actionability/action.  
  Command: `PYTHONPATH=. python scripts/plot_hysteresis_phase.py --log logs/trading_log.csv --save logs/hysteresis_phase.png`
- `scripts/plot_legitimacy_margin.py`: Margin to legitimacy thresholds over time/actionability.  
  Command: `PYTHONPATH=. python scripts/plot_legitimacy_margin.py --log logs/trading_log.csv --save logs/legitimacy_margin.png`
- `scripts/plot_manifold_homology.py`: Compare trader vs CA engagement surfaces.  
  Command: `PYTHONPATH=. python scripts/plot_manifold_homology.py --trader logs/trading_log.csv --ca logs/engagement_surface.csv --save logs/manifold_homology.png`
- `scripts/plot_microstructure_overlay.py`: Overlay price/acceptable with microstructure bands.  
  Command: `PYTHONPATH=. python scripts/plot_microstructure_overlay.py --log logs/trading_log.csv --save logs/microstructure_overlay.png`
- `scripts/plot_policy_curvature.py`: Curvature of engagement policy vs actionability/state.  
  Command: `PYTHONPATH=. python scripts/plot_policy_curvature.py --log logs/trading_log.csv --save logs/policy_curvature.png`
- `scripts/plot_regime_surface.py`: Plot acceptability surface from regime sweep CSV.  
  Command: `PYTHONPATH=. python scripts/plot_regime_surface.py --csv logs/accept_surface.csv --save logs/regime_surface.png`
- `scripts/plot_temporal_homology.py`: Temporal homology diagnostics for acceptable regions.  
  Command: `PYTHONPATH=. python scripts/plot_temporal_homology.py --log logs/trading_log.csv --save logs/temporal_homology.png`
- `scripts/plot_vector_field.py`: Vector-field view of state transitions vs actionability.  
  Command: `PYTHONPATH=. python scripts/plot_vector_field.py --log logs/trading_log.csv --save logs/vector_field.png`
- `scripts/posture_returns.py`: Returns by posture (ACT/HOLD/BAN) with cumulative plots.  
  Command: `PYTHONPATH=. python scripts/posture_returns.py --log logs/trading_log.csv --save logs/posture_returns.png`
- `scripts/rollup_bad_days.py`: Aggregate per-day bad scores for news correlation.  
  Command: `PYTHONPATH=. python scripts/rollup_bad_days.py --log logs/trading_log.csv --out logs/bad_days.csv --top 10`
- `scripts/run_bars_btc.py`: Run bar-level executor on BTC intraday with confidence hysteresis.  
  Command: `PYTHONPATH=. python scripts/run_bars_btc.py`
- `scripts/score_bad_windows.py`: Rank high-severity bad windows; emit CSV.  
  Command: `PYTHONPATH=. python scripts/score_bad_windows.py --log logs/trading_log.csv --out logs/bad_windows.csv`
- `scripts/sweep_confusion_surface.py`: Sweep tau_off and compute FP/FN rates over bins.  
  Command: `PYTHONPATH=. python scripts/sweep_confusion_surface.py --out logs/confusion_surface.csv`
- `scripts/sweep_motif_hysteresis.py`: Sweep hysteresis thresholds on motif CA.  
  Command: `PYTHONPATH=. python scripts/sweep_motif_hysteresis.py`
- `scripts/sweep_regime_acceptability.py`: Sweep regime parameters and export acceptability surface.  
  Command: `PYTHONPATH=. python scripts/sweep_regime_acceptability.py --csv data/raw/stooq/btc_intraday.csv --out logs/accept_surface.csv`
- `scripts/sweep_tau_conf.py`: Sweep tau_off and export precision/recall metrics.  
  Command: `PYTHONPATH=. python scripts/sweep_tau_conf.py --csv data/raw/stooq/btc_intraday.csv --out logs/pr_curve.csv`

## Data and logs

- `data/raw/stooq/`: Stooq CSVs, BTC intraday, and any downloaded market data.
- `data/run_history.csv`: Append-only run summaries from `run_trader`.
- `logs/trading_log.csv`: Primary log for dashboards and analysis scripts.
- `logs/trading_log_trades_*.csv`: Per-trade logs (one row per closed trade).
- `logs/news_events/`: News slices fetched by `emit_news_windows` and `run_all_two_pointO`.

## Notes

- Use `PYTHONPATH=.` when running from the repo root; direct `python <file>.py` works from this directory.
- `data_downloader.py`, `contextual_news.py`, `news_slice.py`, and `emit_news_windows.py` make network calls.
- `training_dashboard_pg.py` requires `pyqtgraph` + Qt bindings; `data_downloader.py` optionally uses `yfinance`.
- `run_trader.py` supports verbosity controls via `--log-level {quiet,info,trades,verbose}` and `--progress-every N`.
- `run_trader.py` supports multi-tape logging with `--all` and `--log-combined` (writes `logs/trading_log_all.csv`).
- `run_trader.py` supports `--max-steps`, `--max-trades`, and `--max-seconds` for bounded runs.
- `run_trader.py` logs edge metrics (`edge_raw`, `edge_ema`); optional cap gate with `--edge-gate --edge-decay --edge-alpha`.
- `run_trader.py` uses a bounded thesis memory counter (`--thesis-depth-max`) to delay soft-veto exits.
- `run_trader.py` exposes goal/MDL tuning (`--goal-cash-x --goal-eps --est-tax-rate --mdl-noise-mult --mdl-switch-penalty --mdl-trade-penalty`).
- `run_trader.py` supports thesis memory (`--thesis-memory`) with per-step logging of thesis state/evidence, position persistence on neutral signals, and optional benchmark-regret reward fields.
- `training_dashboard_pg.py` can render rolling histograms with `--hist --hist-window N --hist-bins M`.
- `training_dashboard_pg.py` expects per-step logs (e.g., `logs/trading_log_*.csv` with a `price` column); trade logs (`logs/trading_log_trades_*.csv`) are not supported.

## Sanity test outcomes

Command (edge-gated, capped runtime; usual sanity runner):

```bash
python run_trader.py --all --log-level trades --progress-every 1000 --inter-run-sleep 0.25 \
  --edge-gate --edge-decay 0.9 --edge-alpha 0.002 --max-trades 1000 --max-seconds 15
```

Run summaries (per CSV):
- `stooq:aapl.us`: steps=6288, trades=616, pnl=99999.5517, elapsed=15.01s, stop=max_seconds
- `stooq:btc.us`: steps=352, trades=2, pnl=100005.7958, elapsed=0.80s
- `stooq:btc_intraday`: steps=6879, trades=16, pnl=99866.8691, elapsed=15.02s, stop=max_seconds
- `stooq:btc_intraday_1s`: steps=6933, trades=54, pnl=99920.4154, elapsed=15.01s, stop=max_seconds
- `stooq:btc_yf`: steps=4121, trades=19, pnl=-126907.0078, elapsed=9.62s
- `stooq:msft.us`: steps=6410, trades=28, pnl=99999.4214, elapsed=15.01s, stop=max_seconds
- `stooq:spy.us`: steps=5242, trades=8, pnl=100065.1214, elapsed=12.40s
- `yahoo:AAPL_1d`: steps=249, trades=0, pnl=100000.0000, elapsed=0.55s
- `yahoo:BTC_USD_1d`: steps=364, trades=2, pnl=90288.2947, elapsed=0.85s
- `yahoo:MSFT_1d`: steps=249, trades=0, pnl=100000.0000, elapsed=0.57s
- `yahoo:SPY_1d`: steps=249, trades=0, pnl=100000.0000, elapsed=0.56s

## Trading logs (fields)

`run_trader.py` now logs both per-step and per-trade fields for efficacy tracking.

Per-step log (see `logs/trading_log*.csv`) includes:
- Execution and movement: `price_exec`, `price_change`, `price_ret`, `fill_units`, `fill_value`
- Position state: `avg_entry_price`, `entry_price`, `entry_step`, `trade_id`, `trade_open`, `price_move_entry`
- PnL detail: `realized_pnl_step`, `realized_pnl_total`, `unrealized_pnl`, `trade_pnl`, `trade_pnl_pct`, `trade_duration`
- Ternary control: `direction`, `edge_t`, `permission`, `capital_pressure`, `risk_budget`, `action_t`
- Thesis memory: `action_signal`, `thesis_depth`, `thesis_hold`
- Thesis memory (FSM): `thesis_d`, `thesis_s`, `thesis_a`, `thesis_c`, `thesis_v`, `thesis_alpha`, `thesis_beta`, `thesis_rho`, `thesis_ds`, `thesis_sum`, `thesis_event`, `thesis_reason`, `thesis_override`
- Belief/decision state: `belief_state`, `belief_plus`, `belief_minus`, `belief_alpha_plus`, `belief_alpha_minus`, `belief_delta_plus`, `belief_delta_minus`, `decision_kind`
- Benchmark-regret reward: `r_step`, `tc_step`, `benchmark_x`, `reward_regret`
- Action persistence: `action_run_length`, `time_since_last_switch`
- Shadow thesis: `shadow_delta_mdl`, `shadow_would_promote`, `shadow_is_tie`, `shadow_reject`
- Plane diagnostics: `plane_abs`, `plane_sign`, `plane_sign_flips_W`, `plane_would_veto`

Per-trade log (see `logs/trading_log_trades_*.csv`) includes:
- `trade_id`, `entry_step`, `exit_step`, `entry_price`, `exit_price`, `trade_duration`
- `trade_pnl`, `trade_pnl_pct`, `price_move`, `price_move_pct`, `close_reason`
- `thesis_depth_exit` (thesis depth at close, when logged)
- `thesis_depth_prev` (thesis depth one step before close, when logged)
- `thesis_depth_peak` (max thesis depth observed during the trade, when logged)

## Belief update (directional gating)

- Directional belief increments are gated by `permission`, `beta`, and `rho`, then apply the signed `alpha` per side.
- See `TRADER_CONTEXT.md:105520` for the formal definition.

## Planned diagnostics (logging-only)

Next pass is observational only (no behavior changes). Add per-step fields:
- `plane_abs = abs(plane_rate)`
- `plane_sign = sign(plane_rate)`
- `plane_sign_flips_W` (rolling sign flips within window `W`)

Then aggregate:
- Promotion rate vs `plane_abs`, `stress`, and `plane_abs × stress`
- PnL vs action run length per plane bucket
- "Would-veto" counts for a prospective plane-stability gate
- Promotion/PnL vs `plane_sign_flips_W` and `plane_would_veto`
- Mean ΔPnL vs `action_run_length` with plane sign persistence and stress quartiles

Geometry plots (no controller changes, see `scripts/plot_decision_geometry.py`):
- Decision heatmaps: `plane_abs × stress → {promotion_rate, action_rate, mean_pnl}`
- Time-series overlay: price + `plane_rate` + `plane_abs` with promote/action markers
- Optional ternary simplex: normalize `p_bad`, `plane_abs`, `stress` to sum to 1
- Use `--simplex-by-thesis` to emit separate simplex plots by `(thesis_d, thesis_s)`
- Use `--simplex-by-thesis-s` to emit separate simplex plots by `thesis_s` only
- Use `--simplex-by-belief` to emit separate simplex plots by belief_state (UNK/FLAT/L1/L2/S1/S2/CONFLICT)

## Planned controller extensions (not implemented)

These are design targets only; log the inputs first and validate geometry before changing behavior.

- Add an orthogonal volatility regime feature from OHLC (realized vol + ternary regime flag).
- Add a minimal thesis-memory state machine (direction/strength/age/cooldown/invalidation).
- Add a benchmark-regret reward option vs a constant-exposure baseline (optionally risk-adjusted by realized vol).
- Plane-aware strategy selection depends on plane-aware belief updates (belief-plane awareness comes first).
- Use fixed quantile bins (q05..q95) with under/overflow bins and suppress bins with < 30 samples
- If no signed `plane_rate` is logged, use `delta_plane` as the signed proxy and `plane_abs = abs(delta_plane)`
- If quantile edges collapse (identical values), fall back to linear bins over finite min/max

`run_trader.py` emits geometry plots by default into `logs/geometry/` with a UTC timestamp
and source label in the filename. The simplex (triangle) plot is enabled by default;
use `--no-geometry-simplex` to skip it, `--no-geometry-overlay` to skip overlays,
or `--no-geometry-plots` to disable all geometry plots.

## Refactor plan (responsibility boundaries first)

When refactoring, split by responsibility boundaries (not file size) and keep the hot loop thin.
Reference notes in `TRADER_CONTEXT.md:106299`.

Proposed minimal module split:

```
trading/
├── io/
│   ├── prices.py        # CSV discovery + load_prices
│   └── logs.py          # CSV append, trade logs
├── signals/
│   ├── triadic.py       # compute_triadic_state
│   ├── stress.py        # compute_structural_stress
│   ├── planes.py        # plane detection, flips, rates
│   └── shadow_mdl.py    # shadow MDL window logic
├── policy/
│   ├── ternary.py       # ternary_controller, permission, sign
│   ├── thesis.py        # thesis FSM + constraints
│   └── belief.py        # belief update + labels
├── execution/
│   ├── sizing.py        # cap, risk parity, edge gate
│   ├── fills.py         # fill logic, slippage, fees
│   └── accounting.py    # PnL, equity, drawdown
├── engine/
│   └── loop.py          # run_trading_loop (thin)
├── cli.py               # argparse + main()
└── run_trader.py        # imports + entrypoint only
```

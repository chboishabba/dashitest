# Repository Guidelines

## Project Structure & Module Organization
- Root contains runnable scripts (e.g., `run_trader.py`, `run_all.py`, `training_dashboard.py`) and core modules (`base.py`, `runner.py`, `intent.py`).
- Trading logic lives under `strategy/` and market state helpers under `signals/`, `features/`, and `regime.py`.
- Execution backends are in `execution/` plus `bar_exec.py` and `hft_exec.py` at the root.
- Data and outputs: `data/` (cached inputs, run history), `logs/` (trading logs, dashboard outputs), `output_*.log` (runtime traces).
- Research and specs are in `docs/` and `policy/`; analysis utilities are in `scripts/`.

## Build, Test, and Development Commands
- `python run_trader.py` runs the core loop on cached data (or synthetic fallback).
- `python run_all.py` runs across all cached markets.
- `python training_dashboard.py --log logs/trading_log.csv --refresh 0.5` launches the Matplotlib dashboard.
- `python training_dashboard_pg.py --log logs/trading_log.csv --refresh 1.0` launches the PyQtGraph dashboard.
- `PYTHONPATH=. python run_all_two_pointO.py --markets ...` runs the full market sweep (see `README.md` for flags).

## Coding Style & Naming Conventions
- Python is the primary language; follow existing style (PEP 8-ish, no enforced formatter).
- Indentation: 4 spaces.
- Naming follows descriptive snake_case for functions/variables and CapWords for classes.
- Scripts generally accept CLI flags; mirror patterns in `scripts/` when adding new tooling.

## Testing Guidelines
- Tests are lightweight and live as runnable scripts (e.g., `scripts/test_ternary_logic.py`, `test_trader_real_data.py`).
- Run with `python <file>.py`; no pytest configuration is present.
- Name new tests with `test_*.py` and keep them self-contained with assertions.

## Commit & Pull Request Guidelines
- Recent history uses short, lowercase messages (often just `auto`). If working in this repo, keep commit subjects concise and consistent with that style unless the maintainer requests otherwise.
- PRs should include: a brief summary, the exact command(s) used to validate, and any log/dashboard artifacts if behavior changes (e.g., updated `logs/trading_log.csv`).

## Security & Configuration Tips
- Some scripts call external data sources (`data_downloader.py`, `scripts/contextual_news.py`, `scripts/emit_news_windows.py`). Confirm API keys and network access before running.
- Use `PYTHONPATH=.` when running from the repo root to avoid import issues.

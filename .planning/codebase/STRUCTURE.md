# Codebase Structure

**Analysis Date:** 2026-01-08

## Directory Layout

```
dashitest/
├── trading/                     # Trading system (signals → policy → execution)
├── dashifine/                   # 4D field slicing & KRR benchmarks
├── dashilearn/                  # Block-sparse MoE training
├── compression/                 # Ternary CA compression & range coding
├── vulkan_compute/              # Vulkan compute shaders & preview
├── vulkan/                      # Vulkan video benchmarks & VAAPI stubs
├── JAX/                        # Reference implementations (not runtime)
├── tests/                       # Root-level tests (pytest)
├── data/                        # Market data CSVs
│   ├── raw/stooq/              # Stooq historical data
│   └── raw/yahoo/              # Yahoo Finance downloads
├── logs/                        # Trading logs, plots, news events
├── outputs/                     # Timestamped benchmark results
├── docs/                        # Design docs and specifications
├── venv/                        # Python virtual environment
├── *_bench.py                   # 30+ benchmark files (root level)
├── CLAUDE.md                    # Master guidance file
├── CONTEXT.md                   # 700KB research notes
├── README.md                    # Project map
└── TODO.md                      # Roadmap
```

## Directory Purposes

**trading/**
- Purpose: Triadic epistemic control system for cryptocurrency markets
- Contains: Signals (triadic state, stress), policy (thesis FSM, regime), strategy (intent generation), execution (bar-level fills), features (quotient extraction)
- Key files: `engine/loop.py`, `run_trader.py`, `run_all_two_pointO.py`, `training_dashboard_pg.py`
- Subdirectories: `signals/`, `policy/`, `strategy/`, `execution/`, `features/`, `trading_io/`, `scripts/` (30+ plotting/analysis scripts)

**dashifine/**
- Purpose: 4D CMYK field slicing with coarse→fine search
- Contains: Core package (`dashifine/dashifine/`), benchmarks (`newtest/`), formal verification (`formal/lean/`)
- Key files: `demo.py`, `newtest/wave_krr.py`, `newtest/grayscott_krr.py`
- Subdirectories: `dashifine/` (palette.py, kernels.py), `newtest/` (40+ specialized runners), `formal/lean/` (Lean 4 proofs), `tests/`

**dashilearn/**
- Purpose: Block-sparse MoE with GPU visualization
- Contains: Training script, compiled int8 kernel, energy exports
- Key files: `bsmoe_train.py`, `vnni_kernel.so`, `sheet_energy.npy`, `run_live_sheet.sh`

**compression/**
- Purpose: Ternary CA + entropy coding
- Contains: CA compression benchmark, range coder, CA generator
- Key files: `compression_bench.py`, `rans.py`, `comp_ca.py`, `video_bench.py`

**vulkan_compute/**
- Purpose: Vulkan compute buffer/image kernels with live preview
- Contains: Compute prototypes, shader compilation, preview window
- Key files: `compute_buffer.py`, `compute_image.py`, `compute_image_preview.py`
- Subdirectories: `shaders/` (GLSL/SPIR-V)

**vulkan/**
- Purpose: Vulkan video benchmarks & VAAPI hardware decode stubs
- Contains: Video benchmark, dmabuf experiments
- Key files: `video_bench_vk.py`, `vaapi_probe.py`
- Subdirectories: `shaders/`

**JAX/**
- Purpose: Reference implementations (not runtime dependency)
- Contains: Codec pipeline, motion search, warps, quadtree
- Key files: `pipeline.py`, `motion_search.py`, `warps.py`, `quadtree.py`

**Root Benchmarks (*.py)**
- Purpose: Standalone micro/macro benchmarks for ternary operations
- Contains: 30+ benchmark files testing SWAR, ternary NN, tree diffusion, Potts-3, etc.
- Examples: `dashitest.py`, `triadic_nn_bench.py`, `tree_diffusion_bench.py`, `swar_test_harness.py`

**data/**
- Purpose: Market data storage
- Location: `data/raw/stooq/` (primary BTC data), `data/raw/yahoo/`
- Key files: `btc_intraday.csv`, `btc_intraday_1s.csv`, `run_history.csv` (append-only)

**logs/**
- Purpose: Trading logs, plots, diagnostics
- Contains: Per-step logs (`trading_log.csv`), per-trade logs (`trading_log_trades_*.csv`), news events (`news_events/`), plots (*.png)

**outputs/**
- Purpose: Timestamped benchmark results
- Contains: Subdirectories with pattern `<benchmark>_<timestamp>/` containing JSON, PNG, GIF

**docs/**
- Purpose: Design documents and specifications
- Contains: `bad_day.md`, `tree_diffusion_benchmark.md`, `grayscott_quotient.md`, `vulkan_jax_parity.md`, `compression_bench.md`

**tests/**
- Purpose: Root-level pytest tests
- Contains: `test_compression_bench.py`, `test_rans.py`, `test_training_dashboard_pg.py`

## Key File Locations

**Entry Points:**
- `trading/run_trader.py` - Core trading loop
- `trading/run_all_two_pointO.py` - Multi-market orchestrator
- `dashifine/demo.py` - 4D field slicing demo
- `dashilearn/bsmoe_train.py` - Block-sparse MoE training
- `compression/compression_bench.py` - CA compression benchmark
- Root `*_bench.py` files - 30+ standalone benchmarks

**Configuration:**
- `CLAUDE.md` - Master guidance file (primary reference)
- `CONTEXT.md` - 700KB research notes
- `dashifine/requirements.txt` - Primary dependency list
- No root `requirements.txt` or `pyproject.toml`

**Core Logic:**
- `trading/engine/loop.py` - Trading loop orchestrator (663+ lines)
- `trading/signals/triadic.py` - Triadic state computation
- `trading/strategy/triadic_strategy.py` - State → Intent conversion
- `trading/execution/bar_exec.py` - Bar-level execution with slippage/fees
- `compression/rans.py` - Range coder implementation
- `dashifine/dashifine/palette.py` - Color/lineage algebra
- `dashilearn/bsmoe_train.py` - Block-sparse MoE core

**Testing:**
- `tests/test_*.py` - Root-level tests
- `dashifine/tests/test_*.py` - Dashifine module tests
- `trading/test_*.py` - Trading module tests (not in tests/)

**Documentation:**
- `README.md` - Project map and file index
- `CLAUDE.md` - Coding patterns, workflows, commands
- `CONTEXT.md` - Detailed research notes (700KB+)
- `TODO.md` - Roadmap and next steps
- `CHANGELOG.md` - Release history

## Naming Conventions

**Files:**
- `run_*.py` - Entry points for orchestrators
- `*_bench.py` - Benchmark files (root level or domain-specific)
- `demo*.py` - Demo/showcase files
- `plot_*.py` - Plotting scripts (always in `trading/scripts/`)
- `sweep_*.py` - Parameter studies
- `test_*.py` - Test files (pytest discovery)
- `*_dashboard*.py` - Visualization dashboards

**Directories:**
- Domain roots: `trading/`, `dashifine/`, `compression/`, etc.
- Snake_case: `vulkan_compute/`, `block_sparse/`
- Plural for collections: `signals/`, `scripts/`, `tests/`

**Special Patterns:**
- `__init__.py` - Module exports (minimal; often empty)
- Output timestamps: `%Y%m%dT%H%M%SZ` format (UTC)
- Benchmark outputs: `outputs/<benchmark>_<timestamp>/`

## Where to Add New Code

**New Trading Feature:**
- Primary code: `trading/<layer>/` (signals/, policy/, execution/)
- Tests: `trading/test_*.py` (co-located with source)
- Plots: `trading/scripts/plot_*.py`

**New Benchmark:**
- Root level for micro/macro: `<name>_bench.py`
- Domain-specific: `<domain>/<name>_bench.py`
- Output: `outputs/<name>_<timestamp>/`

**New Dashifine Runner:**
- Implementation: `dashifine/newtest/<name>_krr.py` or `runner_<name>.py`
- Tests: `dashifine/tests/test_<name>.py`

**New Compression Codec:**
- Implementation: `compression/<name>.py`
- Benchmark: `compression/<name>_bench.py`

**Utilities:**
- Trading utils: `trading/<subsystem>/` (existing structure)
- Shared ternary ops: Root level (e.g., `swar_test_harness.py`)
- Dashifine utils: `dashifine/dashifine/` package

## Special Directories

**venv/**
- Purpose: Python virtual environment
- Source: Created via `python3 -m venv venv`
- Committed: No (in .gitignore)

**outputs/**
- Purpose: Auto-timestamped benchmark results
- Source: Generated by benchmark scripts
- Committed: No (results only)
- Cleanup: Manual (no automated policy)

**logs/**
- Purpose: Trading logs and plots
- Source: Generated by trading runs and plot scripts
- Committed: No
- Cleanup: Manual

**JAX/**
- Purpose: Reference implementations for GPU acceleration research
- Source: Hand-written (not generated)
- Committed: Yes
- Runtime: Not required (JAX is optional)

---

*Structure analysis: 2026-01-08*
*Update when directory structure changes*

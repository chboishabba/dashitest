# Architecture

**Analysis Date:** 2026-01-08

## Pattern Overview

**Overall:** Multi-Domain Triadic Research Platform

**Key Characteristics:**
- Unified triadic logic ({-1, 0, +1}) applied across distinct research domains
- Four primary domains: Trading, Compression, Dashifine (4D field slicing), Dashilearn (block-sparse MoE)
- Philosophy: "Triadic thinking replaces binary collapse" - hold contradictions productively
- Research-oriented with clear separation between production (trading core) and exploratory (benchmarks)
- Local execution (no cloud services), data-driven workflow

## Layers

**Trading System** (`trading/`)

**Signal Layer:**
- Purpose: Pure state computation (no PnL logic)
- Contains: Triadic state {-1,0,+1} from price dynamics, structural stress detection (p_bad)
- Location: `trading/signals/triadic.py`, `trading/signals/stress.py`
- Depends on: NumPy, Pandas for array operations
- Used by: Strategy layer

**Policy Layer:**
- Purpose: Epistemic gates (PnL-free acceptability checks)
- Contains: Regime checking (run-length, flip-rate, vol), thesis memory FSM (belief state machine)
- Location: `trading/policy/thesis.py`, `trading/regime.py`
- Depends on: Signal layer outputs
- Used by: Strategy layer

**Strategy Layer:**
- Purpose: Convert state → Intent (direction, target, hold flag)
- Contains: Hysteresis gating (tau_on > tau_off), actionability checks
- Location: `trading/strategy/triadic_strategy.py`
- Depends on: Signals, policy gates
- Used by: Execution layer

**Execution Layer:**
- Purpose: Fills, slippage, fees, position tracking
- Contains: Bar-level executor, sizing logic, accounting
- Location: `trading/execution/bar_exec.py`, `trading/execution/fills.py`, `trading/execution/sizing.py`, `trading/execution/accounting.py`
- Depends on: Intent from strategy
- Used by: Trading loop

**Dashifine: 4D Field Slicing** (`dashifine/`)
- Purpose: Coarse search → Float refinement → Rotation expansion
- Pipeline: Demo (`demo.py`) orchestrates coarse int8 search over CMYK fields, float32 polish, 10-slice gallery
- Core: `dashifine/dashifine/` (palette.py, kernels.py)
- Benchmarks: `dashifine/newtest/` (wave_krr.py, grayscott_krr.py, primes_krr.py, etc.)

**Dashilearn: Block-Sparse MoE** (`dashilearn/`)
- Purpose: Gate selects active blocks, dense int8 microkernel runs only on active tiles
- Entry: `bsmoe_train.py`
- Exports: `sheet_energy.npy` for Vulkan visualization

**Compression** (`compression/`)
- Purpose: Ternary CA compression, range coding
- Entry: `compression_bench.py` (M4/M7/M9 motifs)
- Core: `rans.py` (range coder), `comp_ca.py` (CA generator)

**GPU Acceleration** (`vulkan_compute/`, `vulkan/`)
- Purpose: Vulkan compute shaders, live preview, dmabuf export stubs
- Entry: `compute_buffer.py`, `compute_image_preview.py`
- Reference: `JAX/` (not runtime dependency)

## Data Flow

**Trading Data Flow:**
```
CSV (Stooq/Yahoo/Binance)
  ↓
data/raw/{stooq,yahoo}/
  ↓
load_prices() [trading_io/prices.py]
  ↓
compute_triadic_state(prices) [signals/triadic.py]
  ↓
compute_structural_stress(prices, returns) [signals/stress.py]
  ↓
compute_quotient_features(log_returns, w1, w2) [features/quotient.py]
  ↓
TriadicStrategy.step(state, conf, regime) → Intent
  ↓
BarExecution.execute(intent) → Fill
  ↓
emit_step_row() → logs/trading_log.csv
  ↓
training_dashboard / plot_* scripts
```

**Benchmark Data Flow:**
```
Entry Point (e.g., tree_diffusion_bench.py)
  ↓
Compute metrics (MSE, quotients, tree diffusion steps)
  ↓
outputs/<benchmark>_<timestamp>/
  ├─ JSON summary
  ├─ PNG rollouts
  └─ GIF animations (if >1 PNG)
```

**State Management:**
- File-based: All state lives in CSV logs, NumPy arrays, JSON outputs
- No persistent in-memory state
- Each run is independent (append-only logs)

## Key Abstractions

**Triadic State Machine:**
- Purpose: Three-state reasoning {-1, 0, +1} (not binary)
- Pattern: 0 is HOLD (epistemic suspension), not absence
- Examples: `trading/signals/triadic.py`, `trading/strategy/triadic_strategy.py`
- Philosophy: Never collapse ±1 together; tensional states (M₆) are productive

**SWAR Ternary Operations:**
- Purpose: Pack 5 trits/byte (99.06% efficiency) for fast computation
- Pattern: Numba-accelerated hot loops with SIMD within a register
- Examples: `dashitest.py`, `swar_test_harness.py`, `triadic_nn_bench*.py`

**Hysteresis & Memory:**
- Purpose: Prevent flip-flop with `tau_on > tau_off` gating
- Pattern: Thesis memory FSM tracks belief states with age/cooldown
- Examples: `trading/policy/thesis.py`, `trading/strategy/triadic_strategy.py`

**Epistemic Separation:**
- Purpose: Separate permission (acceptable) from execution (fills)
- Pattern: Policy checks are PnL-free; execution touches position/PnL
- Examples: `trading/regime.py` (acceptable gates), `trading/execution/bar_exec.py` (fills)

**Cellular Automata Motifs:**
- Purpose: Corridor (M₄), Fatigue (M₇), Shutdown (M₉) detection
- Pattern: Ternary CA with triadic update rules
- Examples: `compression/comp_ca.py`, `motif_ca.py`

**Tree Diffusion & Multiscale Analysis:**
- Purpose: Hierarchical coarse-graining with ultram etric transport
- Pattern: Quotient features (energy, cancellation, spectral dominance)
- Examples: `tree_diffusion_bench.py`, `trading/features/quotient.py`

## Entry Points

**Trading Entry:**
- `python trading/run_trader.py` - Core loop on cached data
- `python trading/run_all.py` - Multi-market runner
- `python trading/run_all_two_pointO.py` - Orchestrator (markets, tau sweeps, CA preview, news)
- `python trading/scripts/run_bars_btc.py` - BTC bar-level execution
- `python trading/training_dashboard.py --log logs/trading_log.csv` - Matplotlib viz
- `python trading/training_dashboard_pg.py --log logs/trading_log.csv` - PyQtGraph viz

**Dashifine Entry:**
- `cd dashifine && python demo.py` - Main 4D field slicing
- `cd dashifine && python newtest/wave_krr.py` - Wave-field KRR benchmark
- `cd dashifine && python newtest/grayscott_krr.py` - Gray-Scott learning

**Dashilearn Entry:**
- `python dashilearn/bsmoe_train.py --epochs 100 --stay-open` - Block-sparse MoE

**Benchmark Entry (root level):**
- `python dashitest.py` - SWAR XOR consumer
- `python triadic_nn_bench.py` - Ternary NN with packed weights
- `python tree_diffusion_bench.py` - Ultrametric transport rollout metrics
- `python compression/compression_bench.py` - CA compression

**Analysis/Plotting (trading/scripts/):**
- 30+ plot_*.py scripts (all auto-timestamp outputs)

## Error Handling

**Strategy:** Minimal error handling (research platform)

**Patterns:**
- Broad `except Exception:` catching in network I/O (`trading/data_downloader.py`)
- Graceful degradation for optional dependencies (yfinance, JAX)
- Validation errors shown before execution (fail fast)

## Cross-Cutting Concerns

**Logging:**
- stdout/stderr for console output
- CSV logs for trading runs (`logs/trading_log*.csv`)
- JSON for benchmark summaries (`outputs/*/summary.json`)

**Validation:**
- Hysteresis assertions: `assert tau_on > tau_off`
- Type hints throughout (using `from __future__ import annotations`)
- No schema validation (file-based I/O)

**Configuration:**
- CLAUDE.md provides master guidance
- Environment variables for API keys
- PYTHONPATH convention for imports

---

*Architecture analysis: 2026-01-08*
*Update when major patterns change*

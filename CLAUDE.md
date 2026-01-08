# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-domain research platform centered on **ternary (triadic) logic** {-1, 0, +1} applied to trading, compression, and GPU-accelerated computation. The unifying philosophy: triadic thinking replaces binary collapse, enabling systems to hold contradictions productively.

**Core domains:**
- **Trading**: Epistemic control system for cryptocurrency markets (primarily BTC)
- **Compression & CA**: Ternary cellular automata and codec research
- **Dashifine**: 4D field slicing and kernel geometry experiments
- **Dashilearn**: Block-sparse MoE training with GPU visualization
- **Vulkan/JAX**: GPU acceleration prototypes (Vulkan compute shaders + JAX references)

## Development Environment

**Python**: 3.12+ (uses `venv` in project root)
**Key dependencies**: NumPy, Pandas, Matplotlib, SciPy, Numba, PyQtGraph (for fast dashboards)

Activate the virtual environment:
```bash
source venv/bin/activate
```

Install dependencies (if needed):
```bash
pip install -r dashifine/requirements.txt
```

## Common Commands

### Trading System

Run the core trading loop on cached/synthetic data:
```bash
python trading/run_trader.py
```

Run across all cached markets:
```bash
python trading/run_all.py
```

Run orchestrator with sweeps, CA preview, and news windows:
```bash
PYTHONPATH=. python trading/run_all_two_pointO.py \
  --markets --market-progress-every 500 \
  --csv data/raw/stooq/btc_intraday_1s.csv \
  --live-sweep --run-ca --ca-report-every 1000
```

Run bar-level executor on BTC with hysteresis:
```bash
PYTHONPATH=. python trading/scripts/run_bars_btc.py
```

Sweep hysteresis parameters (tau_off) for precision/recall:
```bash
PYTHONPATH=. python trading/scripts/sweep_tau_conf.py \
  --csv data/raw/stooq/btc_intraday.csv \
  --out logs/pr_curve.csv
```

### Dashboards

Live matplotlib dashboard:
```bash
python trading/training_dashboard.py --log logs/trading_log.csv --refresh 0.5
```

Fast PyQtGraph dashboard:
```bash
python trading/training_dashboard_pg.py --log logs/trading_log.csv --refresh 1.0
```

### Testing

Run pytest suite (minimal coverage):
```bash
pytest
```

Test compression bench smoke:
```bash
pytest tests/test_compression_bench.py
```

Test PyQtGraph dashboard smoke:
```bash
pytest tests/test_training_dashboard_pg.py
```

### Benchmarks

SWAR ternary operations:
```bash
python dashitest.py
```

Ternary neural network benchmark:
```bash
python triadic_nn_bench.py
```

Tree diffusion benchmark (ultrametric transport):
```bash
python tree_diffusion_bench.py
```

Compression CA benchmark:
```bash
python compression/compression_bench.py
```

Block-sparse MoE training with live visualization:
```bash
python dashilearn/bsmoe_train.py --epochs 100 --stay-open
```

### Dashifine (4D field slicing)

Run the main demo pipeline:
```bash
cd dashifine
python demo.py
```

Run wave-field kernel ridge regression benchmark:
```bash
cd dashifine
python newtest/wave_krr.py
```

Run Gray-Scott operator learning benchmark:
```bash
cd dashifine
PYTHONPATH=. python newtest/grayscott_krr.py \
  --rollout_steps 100 --rollout_gif_steps 20
```

### Vulkan Compute

Run compute buffer prototype:
```bash
cd vulkan_compute
python compute_buffer.py
```

Run live preview with optional recording:
```bash
cd vulkan_compute
python compute_image_preview.py --sheet --sheet-data ../dashilearn/sheet_energy.npy
```

## Architecture Overview

### Trading System (`/trading/`)

**Signal Flow:**
```
Prices (CSV)
  → compute_triadic_state() [rolling vol + EWMA + dead-zone]
  → compute_structural_stress() [vol z-score, flip rate, jump magnitude]
  → compute_quotient_features() [multiscale band analysis]
  → TriadicStrategy.step() [generates Intent with hysteresis]
  → BarExecution.execute() [fills with slippage/fees]
  → emit_step_row() [logs to CSV]
  → training_dashboard [visualization]
```

**Key Components:**
- **`engine/loop.py`**: Core `run_trading_loop()` orchestrator
- **`signals/`**: Triadic state computation, stress metrics
- **`policy/`**: Thesis memory FSM (belief tracking, direction/strength/age/cooldown)
- **`strategy/triadic_strategy.py`**: Intent generation with hysteresis gates
- **`execution/bar_exec.py`**: Bar-level executor with slippage/fees
- **`trading_io/`**: CSV discovery, loading, logging
- **`features/quotient.py`**: Multiscale quotient feature extraction

**Decision Logic:**
```python
if stress > threshold:  # M₉ veto
    UNWIND
elif regime_acceptable:
    if confidence > tau_on:
        ACT (emit intent with direction)
    elif confidence > tau_off:
        HOLD (no-op)
    else:
        emit intent with hold=True
else:
    OBSERVE (intent direction=0)
```

**Important Patterns:**
1. **Epistemic separation**: Permission/acceptability (PnL-free) vs execution
2. **Hysteresis**: `tau_on > tau_off` prevents flip-flop
3. **Triadic state**: {-1, 0, +1} not binary; 0 is HOLD (epistemic), not flatten
4. **27-state backbone**: (S, M, N) ∈ {-1, 0, +1}³ for Self/Mirror/Norm lenses
5. **Structural stress**: `p_bad` classifier detects market dysfunction independently of direction

### Benchmarks & SWAR (`/*.py` root level)

Standalone micro/macro benchmarks for ternary operations:
- **SWAR packing**: 5 trits/byte (99.06% efficiency)
- **Iterative XOR/threshold/dot**: Numba-accelerated hot loops
- **Potts-3 lattice updates**: Mod-3 center+neighbors
- **Snapshot benchmarks**: Hot P/N compute with optional 5-trit snapshots

### Dashifine (`/dashifine/`)

4D CMYK field slicing with adaptive coarse→fine search:
1. **Coarse int8 search**: Fast grid over origin/slopes
2. **Float32 refinement**: Polish best candidate
3. **Rotation expansion**: Generate fan of views around best slice

**Key files:**
- `demo.py`: Full pipeline showcase
- `dashifine/kernels.py`: PSD kernels for KRR/GP experiments
- `newtest/wave_krr.py`: Wave-field completion benchmark (dashifine vs RBF)
- `newtest/grayscott_krr.py`: Gray-Scott operator learning (one-step + rollouts)
- `newtest/primes_krr.py`: p-adic divisibility/valuation tasks

### Dashilearn (`/dashilearn/`)

Block-sparse MoE with GPU visualization:
- `bsmoe_train.py`: Main trainer (gate masks, tile-plan reuse, energy export)
- `run_live_sheet.sh`: Orchestration for learner + Vulkan preview
- `vnni_kernel.so`: Compiled int8 VNNI kernel (tile-level operations)

### Compression (`/compression/`)

Ternary CA + entropy coding:
- `compression_bench.py`: Triadic CA compression benchmark (M4/M7/M9 motifs)
- `rans.py`: Range coder with rANS-like API
- `comp_ca.py`: CA generator/visualizer

### Vulkan/JAX (`/vulkan_compute/`, `/vulkan/`, `/JAX/`)

GPU acceleration prototypes:
- **Vulkan compute**: Buffer/image kernels, live preview with optional recording
- **JAX references**: Motion search, codec pipelines (not runtime dependencies)
- **Parity map**: See `docs/vulkan_jax_parity.md` for kernel correspondence

## Important Coding Patterns

### 1. Triadic Control (Never Collapse to Binary)

```python
# GOOD: Triadic state preserved
state = {-1, 0, +1}  # negative, neutral, positive
if state == 1:
    action = ACT
elif state == -1:
    action = BAN
else:  # state == 0
    action = HOLD  # Epistemic suspension, not forced flatten
```

```python
# BAD: Binary collapse loses information
if state != 0:  # Collapses ±1 together
    action = ACT
```

### 2. Hysteresis Gates (Prevent Flip-Flop)

Always enforce `tau_on > tau_off`:
```python
# GOOD: Hysteresis with threshold assertion
assert tau_on > tau_off, "Hysteresis requires tau_on > tau_off"
if not is_holding and confidence > tau_on:
    is_holding = False  # Enter ACT
elif is_holding and confidence < tau_off:
    is_holding = True   # Enter HOLD
```

### 3. Separation of Concerns (Trading)

**Signals** (pure computation):
```python
# No side effects, no PnL logic
def compute_triadic_state(prices: np.ndarray) -> np.ndarray:
    """Returns {-1, 0, +1} array."""
```

**Policy** (epistemic gates):
```python
# PnL-free regime checks
def check_regime(states, vols, spec: RegimeSpec) -> bool:
    """Returns acceptable bool based on flip-rate, run-length, vol."""
```

**Execution** (fills & accounting):
```python
# Only here do we touch position/PnL
def execute(intent: Intent, price: float) -> FillInfo:
    """Applies slippage, fees, updates exposure."""
```

### 4. M₆ vs M₉ Distinction (Crucial)

- **M₆**: Unresolved tension between two stances (productive)
- **M₉**: Veto/closure operator (use sparingly)

```python
# GOOD: M₆ held as valid state
if (S, M, N) has 2 distinct values:
    return CAUTION  # Dialectical tension

# GOOD: M₉ as rare veto
if structural_stress > CRITICAL_THRESHOLD:
    return BAN  # Market is untradeable
```

**Never auto-escalate M₆ → M₉.** M₆ is where insight forms.

### 5. Logging Best Practices

All plot scripts auto-timestamp `--save` outputs to avoid overwriting:
```python
# GOOD: Auto-timestamped output
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_path = f"logs/quotient_{timestamp}.png"
```

When dumping many PNGs (rollouts, planes), roll them into a GIF and delete PNGs:
```python
# After generating frames
subprocess.run(["convert", "-delay", "10", "-loop", "0",
                "frame_*.png", "rollout.gif"])
os.system("rm frame_*.png")
```

### 6. PYTHONPATH Convention

When running scripts from repo root, always use `PYTHONPATH=.`:
```bash
PYTHONPATH=. python trading/scripts/run_bars_btc.py
```

When running from within a subdirectory (e.g., `dashifine/`), you can omit it:
```bash
cd dashifine
python demo.py
```

## Data Organization

### Trading Data
- **`data/raw/stooq/`**: Stooq CSVs (historical OHLCV)
- **`data/raw/yahoo/`**: Yahoo Finance downloads
- **`data/run_history.csv`**: Append-only run summaries
- **`logs/trading_log.csv`**: Per-step logs for dashboards
- **`logs/trading_log_trades_*.csv`**: Per-trade logs
- **`logs/news_events/`**: News slices from bad-day windows

### Benchmark Outputs
- **`outputs/`**: Timestamped benchmark results (JSON, PNG, GIF)
- **`dashilearn/sheet_energy.npy`**: Band energy exports for Vulkan preview
- **`/mnt/data`**: Dashifine demo outputs (coarse density map, slices, summary.json)

## Key Files to Reference

### Documentation
- **`README.md`**: Project map, file index
- **`CONTEXT.md`**: Long-form research notes (700KB, detailed context)
- **`CHANGELOG.md`**: Release history
- **`TODO.md`**: Roadmap and next steps
- **`trading/README.md`**: Trading system detailed guide
- **`dashifine/README.md`**: Dashifine pipeline and benchmarks

### Core Implementation
- **`trading/engine/loop.py`**: Main trading loop
- **`trading/strategy/triadic_strategy.py`**: Intent generation
- **`trading/signals/triadic.py`**: Triadic state computation
- **`trading/signals/stress.py`**: Structural stress (p_bad)
- **`trading/policy/thesis.py`**: Thesis memory FSM
- **`swar_test_harness.py`**: SWAR ternary operations harness
- **`compression/rans.py`**: Range coder implementation

### Specifications
- **`docs/bad_day.md`**: Structural stress concept
- **`docs/tree_diffusion_benchmark.md`**: Tree diffusion spec
- **`docs/vulkan_jax_parity.md`**: Vulkan↔JAX kernel correspondence
- **`docs/grayscott_quotient.md`**: Gray-Scott quotient metrics
- **`trading/docs/decision_alignment_check.md`**: Closest-profitable alignment spec
- **`trading/docs/quotient_integration.md`**: Quotient-learner integration plan

## Triadic Philosophy (Core Principles)

1. **Binary logic collapses contradictions. Triadic logic holds them.**
   - Don't force {-1, +1} into a single "active" state
   - Zero (0) is HOLD: epistemic suspension, not absence

2. **Permission ≠ Prediction**
   - Acceptability gates are PnL-free (flip-rate, run-length, vol)
   - Execution is separate from epistemic gating

3. **Hysteresis prevents flip-flop**
   - Always `tau_on > tau_off`
   - State transitions have memory

4. **27-state backbone is minimal, not maximal**
   - (S, M, N) ∈ {-1, 0, +1}³ = Self/Mirror/Norm lenses
   - Full virtual tensor is 3⁹ = 19,683 states projected onto 27

5. **Structural stress is sovereign**
   - When `p_bad` > threshold (M₉ veto), flatten and refuse exposure
   - This is "world is a bad game" detection, not directional signal

6. **M₆ (unresolved tension) ≠ M₉ (closure/veto)**
   - M₆ is productive; don't auto-escalate to M₉
   - M₉ is rare and existential

## Testing Strategy

Tests are minimal but critical smoke tests:
- `tests/test_compression_bench.py`: CA compression roundtrip
- `tests/test_rans.py`: rANS encoder/decoder
- `tests/test_training_dashboard_pg.py`: PyQtGraph dashboard smoke
- `dashifine/tests/`: Palette math, runner primitives, integration

**Run full suite:**
```bash
pytest
```

**Run specific test:**
```bash
pytest tests/test_rans.py -v
```

## Common Pitfalls

1. **Don't commit files unless necessary**
   - Prefer editing existing files to creating new ones
   - Never create markdown docs proactively unless explicitly requested

2. **Don't collapse triadic states to binary**
   - Keep {-1, 0, +1} distinct; 0 is not "no signal"

3. **Don't skip hysteresis checks**
   - Always assert `tau_on > tau_off` in strategy code

4. **Don't mix epistemic and execution logic**
   - Signals compute state (no PnL)
   - Policy gates (no PnL)
   - Execution touches position/PnL

5. **Don't use relative paths in commands**
   - From repo root: `PYTHONPATH=. python trading/...`
   - From subdir: `cd trading && python ...`

6. **Don't overwrite prior benchmark outputs**
   - Always auto-timestamp save files
   - Roll multiple PNGs into GIFs for storage efficiency

7. **Don't assume JAX is available at runtime**
   - JAX is reference-only on this machine
   - Vulkan is the target for GPU acceleration

## Project-Specific Terminology

- **Triadic/Ternary**: {-1, 0, +1} states, not binary
- **HOLD**: Epistemic suspension (state=0), not flatten
- **ACT**: State ∈ {-1, +1}, permission granted
- **BAN**: Forced flat (M₉ veto), structural stress override
- **M₄/M₇/M₉**: CA motifs (corridor, fatigue, shutdown)
- **SWAR**: SIMD Within A Register (bit-packing ternary ops)
- **p_bad**: Structural stress score ∈ [0,1], "bad day" classifier
- **Acceptable**: Regime predicate (PnL-free), based on flip-rate/run-length
- **Hysteresis**: `tau_on > tau_off` gate with memory
- **27-state backbone**: (S,M,N) ∈ {-1,0,+1}³ minimal observable
- **Quotient features**: Multiscale band analysis via tree diffusion
- **Thesis memory**: FSM tracking belief (direction, strength, age, cooldown)
- **Intent**: Immutable dataclass from strategy (direction, target, hold, actionability)
- **Fill**: Execution result (exposure, fee, slippage)

## Workflow Reminders

1. **Always read files before editing**
   - Never propose changes to code you haven't read
   - Use Read tool first, then Edit

2. **Use Explore agent for codebase questions**
   - Don't run grep/glob directly for broad questions
   - Use Task tool with subagent_type=Explore

3. **Prefer parallel tool calls when independent**
   - Read multiple files in one message if no dependencies
   - Run multiple Bash commands in parallel when possible

4. **Keep solutions simple**
   - Don't add features beyond what was asked
   - No premature abstractions or over-engineering
   - Don't add comments/docstrings to unchanged code

5. **Follow naming conventions**
   - Trading logs: `logs/trading_log*.csv`
   - Plots: Auto-timestamp with `%Y%m%dT%H%M%SZ`
   - Benchmark outputs: `outputs/<benchmark>_<timestamp>/`

## Contact & Support

For questions or issues, see:
- `/help`: Claude Code help
- GitHub issues: https://github.com/anthropics/claude-code/issues (for Claude Code itself)
- `CONTEXT.md`: Extensive research notes and design decisions

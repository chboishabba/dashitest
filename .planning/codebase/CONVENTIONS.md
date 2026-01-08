# Coding Conventions

**Analysis Date:** 2026-01-08

## Naming Patterns

**Files:**
- snake_case for all Python files (`triadic_strategy.py`, `bar_exec.py`)
- `test_*.py` for test files
- `run_*.py` for entry points
- `*_bench.py` for benchmarks
- `plot_*.py` for plotting scripts

**Functions:**
- snake_case for all functions (`compute_triadic_state`, `sign_run_lengths`)
- Private functions: underscore prefix (`_legacy_hsv`, `_neighbor_counts_ternary`)
- No special prefix for async functions

**Variables:**
- snake_case for variables (`price`, `volume`, `ts`, `state`)
- Boolean with `is_` prefix: `is_holding`, `acceptable`
- UPPER_SNAKE_CASE for constants (`MAX_TOTAL`, `MAGIC`, `LANE_SHIFTS`)

**Types:**
- PascalCase for classes (`FreqTable`, `RangeEncoder`, `RegimeSpec`, `TriadicStrategy`)
- PascalCase for dataclasses (`Intent`)
- PascalCase for enums
- No `I` prefix for interfaces

## Code Style

**Formatting:**
- 4 spaces per indentation level (standard Python)
- Line length: Generally <100 characters (some exceptions for NumPy operations)
- Single blank lines between functions, double blank lines between classes
- No automated formatter (Prettier, Black) detected

**Linting:**
- No configured linters (no `.flake8`, `.pylintrc`, `setup.cfg`, `pyproject.toml`)
- Style enforced by CLAUDE.md guidance and manual review

**Quoting:**
- Mix of single and double quotes (no consistent preference)
- F-strings used for string formatting

## Import Organization

**Order:**
1. `from __future__ import annotations` (in newer files)
2. Standard library imports (`sys`, `pathlib`, `time`, `dataclasses`, `typing`)
3. Third-party imports (`numpy`, `pandas`, `matplotlib`)
4. Local/relative imports (with try/except for portability)

**Grouping:**
- Blank line between groups
- No alphabetical sorting enforced

**Path Handling:**
Portable import pattern with try/except:
```python
try:
    from trading.bar_exec import BarExecution
except ModuleNotFoundError:
    from bar_exec import BarExecution
```

**Path Aliases:**
- None configured
- Use PYTHONPATH convention: `PYTHONPATH=. python trading/...`

## Error Handling

**Patterns:**
- Broad `except Exception:` catching common (especially in network I/O)
- Graceful degradation for optional dependencies (yfinance, JAX)
- Minimal logging in exception handlers
- No custom exception hierarchy

**Error Types:**
- Standard library exceptions used (`ValueError`, `FileNotFoundError`)
- No project-specific exception classes

**Async:**
- Limited async usage (mostly synchronous code)

## Logging

**Framework:**
- Standard print() to stdout/stderr
- No structured logging framework (pino, winston, etc.)

**Patterns:**
- Print statements for progress/debugging
- CSV logs for trading runs (`logs/trading_log.csv`)
- JSON for benchmark summaries (`outputs/*/summary.json`)

## Comments

**When to Comment:**
- Explain WHY, not WHAT (per CLAUDE.md guidance)
- Document business rules and triadic logic
- Explain non-obvious algorithms
- Avoid obvious comments

**Module Docstrings:**
- Present at file top with triple quotes
- Examples: `compression/rans.py`, `trading/regime.py`

**Function Docstrings:**
- Present for public functions
- Format: Basic description with Args/Returns in some files
- Not uniformly applied across all modules

**Inline Comments:**
- Sparse but descriptive
- Dashed section separators: `# --- Small triadic CA generator ---`

**TODO Comments:**
- Format: `# TODO: description` (no username)
- Extensive in-code TODOs (see CONCERNS.md)

## Function Design

**Size:**
- No hard limit enforced
- Many functions >100 lines (e.g., `trading/engine/loop.py`)
- Complex functions not aggressively extracted

**Parameters:**
- No max parameter limit
- Mix of individual params and option dicts
- Type hints used extensively

**Return Values:**
- Explicit return statements preferred
- Type hints for return types: `-> np.ndarray`, `-> bool`, `-> Tuple[float, float]`
- Early returns for guard clauses

## Module Design

**Exports:**
- Minimal __init__.py files (often empty)
- Direct imports from modules (not via package __init__)
- No barrel files pattern

**Circular Dependencies:**
- Avoided via try/except import patterns
- Module boundaries generally clean

## Type Hints

**Usage:**
- Extensive type hints throughout
- Modern syntax: `from __future__ import annotations`
- Function signatures include return types
- Dataclass field types specified

**Patterns:**
- `np.ndarray` for NumPy arrays
- `Tuple`, `List`, `Dict` from typing module
- Generic types: `List[int]`, `Dict[str, Any]`

## Documentation Style

**Primary Reference:**
- CLAUDE.md is master documentation (not scattered READMEs)
- Project-specific terminology documented in CLAUDE.md
- 700KB CONTEXT.md for detailed design decisions

**Per-module READMEs:**
- `trading/README.md` - Trading system guide
- `dashifine/README.md` - Dashifine guide
- `vulkan_compute/README.md` - GPU stack details

## Project-Specific Patterns

**Triadic State Preservation:**
- NEVER collapse {-1, 0, +1} to binary
- 0 is HOLD (epistemic suspension), not absence
- Always enforce `tau_on > tau_off` for hysteresis

**Epistemic Separation:**
- Signals: pure computation, no PnL logic
- Policy: gates without PnL
- Execution: touches position/PnL only

**Auto-Timestamped Outputs:**
- Pattern: `%Y%m%dT%H%M%SZ` (UTC)
- Roll PNGs into GIFs then delete for storage efficiency

**PYTHONPATH Convention:**
- From root: `PYTHONPATH=. python trading/...`
- From subdir: `cd dashifine && python ...`

---

*Convention analysis: 2026-01-08*
*Update when patterns change*

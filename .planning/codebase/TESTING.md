# Testing Patterns

**Analysis Date:** 2026-01-08

## Test Framework

**Runner:**
- pytest (discovery pattern)
- No config file (`pytest.ini`, `conftest.py`) detected in root

**Assertion Library:**
- pytest built-in expect
- `np.testing.assert_allclose()` for numerical tolerance
- Custom assertion helpers in some tests

**Run Commands:**
```bash
pytest                            # Run all tests
pytest tests/                     # Root-level tests only
pytest dashifine/tests/           # Dashifine tests
pytest -v                         # Verbose output
```

## Test File Organization

**Location:**
- `tests/` - Root-level tests (3 files)
  - `test_compression_bench.py` (8 lines) - Smoke test
  - `test_rans.py` (23 lines) - rANS roundtrip
  - `test_training_dashboard_pg.py` (56 lines) - Dashboard tests

- `dashifine/tests/` - Dashifine module tests (10+ files)
  - `test_primitives.py` - Core primitives
  - `test_lineage_palette.py` - Palette math
  - `test_chsh_harness.py` - Quantum/CHSH tests
  - `test_integration.py` - End-to-end pipeline

- `trading/` - Trading tests (2 files, NOT in tests/)
  - `test_trader_real_data.py` (44 lines) - Real data sanity
  - `test_thesis_memory.py` (177 lines) - FSM state machine

**Naming:**
- `test_*.py` for all test files (pytest discovery)
- Functions: `test_<name>()`

**Structure:**
```
src/
  lib/
    utils.py
    (no test - trading module handles own tests)
  dashifine/tests/
    test_lineage_palette.py
    test_primitives.py
trading/
  test_trader_real_data.py (co-located)
  test_thesis_memory.py (co-located)
tests/
  test_compression_bench.py
  test_rans.py
```

## Test Structure

**Suite Organization:**
```python
# Standard pytest pattern
def test_function_name():
    # arrange
    input_data = create_test_data()

    # act
    result = function_under_test(input_data)

    # assert
    assert result == expected
```

**Path Setup Pattern:**
```python
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

**Patterns:**
- No `beforeEach`/`afterEach` patterns detected
- Fixtures: pytest `tmp_path` fixture used (`test_integration.py`)
- Custom assertions: `assert_equal()` helper in `test_thesis_memory.py`

## Mocking

**Framework:**
- No mocking framework detected (no `unittest.mock`, `pytest-mock`)
- Tests use real data or synthetic data generation

**Patterns:**
- No mocking patterns found
- Tests are integration-style rather than unit-style

**What to Mock:**
- Not applicable (tests use real implementations)

**What NOT to Mock:**
- Not applicable

## Fixtures and Factories

**Test Data:**
```python
# Factory pattern (inline)
def create_test_user():
    return {"id": "test", "name": "Test User"}

# Synthetic data generation
prices = np.random.randn(1000).cumsum()
```

**Location:**
- Factory functions: inline in test files
- No shared fixtures directory

## Coverage

**Requirements:**
- No enforced coverage target
- Minimal coverage philosophy (per CLAUDE.md)
- Focus on correctness validation, not coverage metrics

**Configuration:**
- No coverage tools configured

**View Coverage:**
- Not tracked

## Test Types

**Smoke Tests:**
- Test that major benchmarks run without error
- Examples: `test_compression_benchmark_smoke()`

**Unit Tests:**
- Focused on single functions
- Examples: `test_gelu_is_odd()`, `test_rans_roundtrip_small()`

**Integration Tests:**
- Component interaction
- Examples: `test_main_creates_outputs()`, `test_trader_real_data()`

**FSM/State Machine Tests:**
- Complex logic validation
- Examples: `test_thesis_memory.py` (177 lines testing FSM transitions)

**Mathematical/Numerical Tests:**
- Validate mathematical properties
- Examples: Orthonormalization tests, tolerance checks with `np.testing.assert_allclose()`

## Common Patterns

**Async Testing:**
- Not applicable (synchronous codebase)

**Error Testing:**
```python
# Standard pattern
def test_should_raise_on_invalid():
    with pytest.raises(ValueError):
        function_call(invalid_input)
```

**Numerical Tolerance:**
```python
import numpy as np

def test_numerical_result():
    result = compute()
    expected = 1.23456
    np.testing.assert_allclose(result, expected, rtol=1e-5)
```

**Snapshot Testing:**
- Not used

## Manual Testing Approach

**Extensive manual testing** via benchmark scripts:
- `dashitest.py` - SWAR kernel validation
- `triadic_nn_bench.py` - Ternary NN validation
- `compression_bench.py` - Codec roundtrip verification
- `tree_diffusion_bench.py` - Metric validation
- 30+ other benchmark files serve as integration tests

**Visual verification:**
- Dashboards: `training_dashboard.py`, `training_dashboard_pg.py`
- Plot scripts: `trading/scripts/plot_*.py` (30+ scripts)
- Outputs inspected manually

## Testing Philosophy

From CLAUDE.md:
- **Minimal but critical coverage**: Focus on smoke tests, correctness, FSM validation
- **No coverage metrics**: Coverage for awareness only, not enforcement
- **Manual verification**: Extensive benchmarks + visual inspection
- **Research platform**: Tests validate behavior, not comprehensive coverage

## Known Gaps

- Trading loop logic (`trading/engine/loop.py`, 663+ lines) has no unit tests
- Data downloader (`trading/data_downloader.py`, 600+ lines) has no error case tests
- Block-sparse execution (`dashilearn/bsmoe_train.py`) not tested
- Most benchmarks lack automated tests (rely on manual inspection)

---

*Testing analysis: 2026-01-08*
*Update when test patterns change*

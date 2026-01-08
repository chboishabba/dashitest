# Technology Stack

**Analysis Date:** 2026-01-08

## Languages

**Primary:**
- Python 3.12+ - All application code (`CLAUDE.md`, `dashifine/requirements.txt`)

**Secondary:**
- Lean 4 - Formal verification (`dashifine/formal/lean/`)
- GLSL/SPIR-V - GPU shaders (`vulkan_compute/shaders/`, `vulkan/shaders/`)
- Bash - Orchestration scripts (`dashilearn/run_live_sheet.sh`)

## Runtime

**Environment:**
- Python 3.12+ (venv at project root)
- No browser runtime (local CPU/GPU only)

**Package Manager:**
- pip via venv
- Primary requirements: `dashifine/requirements.txt`
- No `requirements.txt` in project root or `pyproject.toml`

**Activation:**
```bash
source venv/bin/activate
```

## Frameworks

**Core:**
- **Triadic/Ternary Computing Framework** (proprietary)
  - SWAR ternary packing (5 trits/byte, 99.06% efficiency) - `swar_test_harness.py`, `dashitest.py`
  - Potts-3 Cellular Automata - `compression/comp_ca.py`, `motif_ca.py`
  - Triadic Logic {-1, 0, +1} - Core to all domains

**Scientific Computing:**
- NumPy 2.4.0 - Array operations (`dashifine/requirements.txt`)
- SciPy - Signal processing, optimization
- Pandas - Time series, market data

**Visualization:**
- Matplotlib 3.10.8 - Primary plotting (`dashifine/requirements.txt`)
- PyQtGraph - Fast dashboards (`trading/training_dashboard_pg.py`)
- Pillow 12.1.0 - Image processing

**Performance:**
- Numba - JIT compilation for hot loops
- CFFI 2.0.0 - C FFI for Vulkan bindings

**GPU Acceleration:**
- Vulkan 1.3.275.1 - Compute shaders (`vulkan_compute/compute_buffer.py`)
- GLFW 2.10.0 - Window/input management

**Testing:**
- pytest - Unit tests (`dashifine/requirements.txt`)
- ImageIO - GIF/video frame handling

**HTTP:**
- requests - Data fetching (`trading/data_downloader.py`)

**Optional/Reference:**
- JAX - Reference implementations only (not runtime dependency) (`JAX/`)
- yfinance - Optional market data (gracefully degraded) (`trading/data_downloader.py`)

## Key Dependencies

**Critical:**
- **NumPy** 2.4.0 - Core array operations across all modules
- **Matplotlib** 3.10.8 - Plotting and visualization
- **PyQtGraph** - Real-time trading dashboards
- **Vulkan** 1.3.275.1 - GPU compute acceleration
- **Numba** - Performance-critical ternary operations

**Infrastructure:**
- **SciPy** - Signal processing for trading/dashifine
- **Pandas** - Market data handling, time series
- **Pillow** 12.1.0 - Image operations for compression/video
- **CFFI** 2.0.0 - Vulkan FFI bindings
- **requests** - Market data APIs (Binance, Stooq, NewsAPI, GDELT)

## Configuration

**Environment:**
- `NEWSAPI_KEY` - NewsAPI authentication (optional)
- `VK_ICD_FILENAMES` - Vulkan ICD selection (`/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`)
- `YF_CACHE_DIR` - Yahoo Finance cache directory
- `YF_NO_CACHE` - Disable yfinance caching (set to "1")
- No `.env` files detected

**PYTHONPATH Convention:**
- From repo root: `PYTHONPATH=. python trading/run_trader.py`
- From subdirectory: `cd dashifine && python demo.py`

**Build:**
- No build tools detected (pure Python)
- Vulkan shaders pre-compiled

## Platform Requirements

**Development:**
- Linux (tested on Linux 6.18.3-1-cachyos-bore)
- AMD GPU (RX 580/gfx803) for Vulkan compute
- Python 3.12+ with venv

**Production:**
- Same as development (local execution)
- No deployment to cloud services
- Data sources: Public APIs only (Binance, Stooq, Yahoo, NewsAPI, GDELT)

---

*Stack analysis: 2026-01-08*
*Update after major dependency changes*

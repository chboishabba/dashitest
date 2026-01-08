# Codebase Concerns

**Analysis Date:** 2026-01-08

## Tech Debt

**LOB Replay Execution Stub:**
- Issue: `trading/hft_exec.py:32` - LOBReplayExecution returns empty fills, non-functional
- Why: Waiting for Binance BTC/ETH book+trade data integration
- Impact: Cannot test limit order execution, queue simulation, or LOB dynamics
- Fix approach: Implement hftbacktest integration once L2/trades data is available

**Runner Integration Incomplete:**
- Issue: `trading/runner.py:15-19` - Multiple documented TODOs in header
- Why: Transition from direct fills to Intent-based architecture in progress
- Impact: Runner module not yet wired to new strategy/execution separation
- Fix approach: Wire strategy to emit Intents, implement BarExecution using current run_trader logic

**Repeated Retry Logic:**
- Issue: `trading/data_downloader.py` - Same exponential backoff pattern in 6+ functions (lines 79-94, 114-128, 153-187, 214-226, 273-284)
- Why: Copy-paste during rapid development
- Impact: Code duplication, inconsistent retry behavior
- Fix approach: Extract shared retry utility with configurable backoff/timeout

**Hardcoded Configuration:**
- Issue: `trading/engine/loop.py:38-81` - 40+ magic numbers for thresholds (PBAD_CAUTION=0.4, VEL_EXIT=3.0, etc.)
- Why: Exploratory tuning during research phase
- Impact: Difficult to sweep parameters or A/B test
- Fix approach: Externalize to config file or command-line args

## Known Bugs

**None explicitly documented** - Research platform with iterative development

## Security Considerations

**Bare Star Imports:**
- Risk: `from vulkan import *` in 6+ files (`vulkan_compute/compute_buffer.py`, `vulkan/video_bench_vk.py`, etc.)
- Files: `vulkan_compute/compute_buffer.py:10`, `vulkan_compute/compute_image.py:10`, `vulkan_compute/compute_image_preview.py:23`, `vulkan/vaapi_probe.py:10`, `vulkan/symbol_stream_stub.py:9`, `vulkan/vaapi_dmabuf_stub.py:11`
- Current mitigation: None (local execution only, no user input)
- Recommendations: Explicit imports to avoid namespace pollution, improve IDE support

**API Keys in Environment:**
- Risk: `NEWSAPI_KEY` stored in environment variables
- Files: `trading/scripts/news_slice.py:27`
- Current mitigation: Optional feature, local execution only
- Recommendations: Document proper env var handling, consider `.env` file support

## Performance Bottlenecks

**GIF Export Runtime:**
- Problem: Gray-Scott KRR GIF export takes ~25m on CPU
- File: Gray-Scott rollout generation
- Measurement: ~25 minutes per benchmark run
- Cause: Sequential frame generation + ImageIO encoding
- Improvement path: Add caching or `--no-viz` flag to skip GIF export

**File Row Counting:**
- Problem: Opens and scans entire file just to count rows
- Files: `trading/data_downloader.py:140, 202, 256`
- Measurement: O(n) scan for validation
- Cause: `sum(1 for _ in open(out_path))` without closing file handle
- Improvement path: Track row count during write, or sample-based validation

**Block-Sparse Tile Set Conversion:**
- Problem: Jaccard similarity computed via set conversion for large tile counts
- File: `dashilearn/bsmoe_train.py:122-131`
- Measurement: O(n) space for millions of tiles
- Cause: Converts entire NumPy arrays to Python sets
- Improvement path: Sparse approximation or bit-vector operations

## Fragile Areas

**Trading Loop State Management:**
- File: `trading/engine/loop.py`
- Why fragile: 663+ lines with complex FSM (thesis memory, stress gates, sizing)
- Common failures: State transitions not validated, accumulator overflow risk
- Safe modification: Add unit tests before changing state logic
- Test coverage: Only smoke test in `trading/test_trader_real_data.py`

**Data Downloader Error Handling:**
- Files: `trading/data_downloader.py` (17+ broad `except Exception:` blocks)
- Why fragile: Network errors silently caught, difficult to debug DNS/timeout failures
- Common failures: Silent failures on malformed responses, no retry logging
- Safe modification: Add specific exception types, log all caught exceptions
- Test coverage: No tests for error cases

**Tree Diffusion Benchmark:**
- Files: `tree_diffusion_bench.py`, `docs/tree_diffusion_benchmark.md`
- Why fragile: Design choices not locked (TODO.md:62-74)
- Common failures: Additional sweeps added mid-implementation
- Safe modification: Lock design choices before adding features
- Test coverage: Manual validation via benchmark outputs

## Scaling Limits

**Output File Accumulation:**
- Current capacity: Unbounded growth in `outputs/` and `logs/`
- Limit: Disk space
- Symptoms at limit: Out of disk space errors
- Scaling path: Implement cleanup policy or archive old runs

**CSV Log Size:**
- Current capacity: `logs/trading_log.csv` grows linearly with steps
- Limit: Memory for dashboard loading
- Symptoms at limit: Dashboard slowdown on large logs
- Scaling path: Implement log rotation or binary format

## Dependencies at Risk

**JAX Reference Dependency:**
- Risk: Optional reference-only, but import errors possible if used without installation
- Files: `JAX/` modules import `jax` without runtime guards
- Impact: Import errors in reference code
- Migration plan: Add `try/except` imports or document as optional

**yfinance Optional Dependency:**
- Risk: Inconsistent checks for None after try/except import
- Files: `trading/data_downloader.py:20-23`
- Impact: Some functions silently return None, others may fail
- Migration plan: Consistent None-checking pattern across all usages

## Missing Critical Features

**From TODO.md - Documented Deferred Work:**

- **Block-sparse int8/VNNI path** (`dashilearn/bsmoe_train.py`) - Active tiles from gate masks, TilePlan caching not implemented

- **Vulkan/JAX parity mapping** - Entry point correspondence not documented (see `docs/vulkan_jax_parity.md`)

- **Vulkan sheet visual shader** - Live learner data feed not integrated (`vulkan_compute/compute_image_preview.py`)

- **Tree diffusion adversarial operator** - Nonlinear band coupling marked `[wip]`, design choices not finalized

- **Gray-Scott quotient metrics** - V+radial(U) metrics per `docs/grayscott_quotient.md` not implemented

- **Motif CA visualizer** - True vs learned CA motifs not wired to dashboard

- **Hysteresis PR sweep** - `trading/scripts/sweep_tau_conf.py` incomplete, needs P/Q/N cull benchmark integration

## Test Coverage Gaps

**Trading Loop Logic:**
- What's not tested: `trading/engine/loop.py` (663+ lines) - Complex FSM, stress computation, sizing
- Risk: State transitions could break unnoticed
- Priority: High (core trading logic)
- Difficulty to test: Requires synthetic market data fixtures

**Data Downloader Reliability:**
- What's not tested: `trading/data_downloader.py` (600+ lines) - Network error cases
- Risk: Silent failures in production data fetch
- Priority: Medium
- Difficulty to test: Requires mock HTTP responses, DNS failures

**Compression Codec Edge Cases:**
- What's not tested: `compression/rans.py` - Empty data, single symbol, max alphabet
- Risk: Codec failures on edge cases
- Priority: Low (research code)
- Difficulty to test: Easy (unit tests with synthetic data)

**Block-Sparse Execution:**
- What's not tested: `dashilearn/bsmoe_train.py` - Tile plan generation, energy computation
- Risk: Compiled `.so` kernel makes validation difficult
- Priority: Medium
- Difficulty to test: Requires reference implementation or known-good outputs

## Documentation Gaps

**Undocumented Algorithms:**
- Files: `dashifine/demo.py:62-96` (field slicing), `compression/compression_bench.py:36-90` (CA step function)
- What's missing: Inline explanation of mathematical models, motif logic (M4/M7/M9)
- Impact: Difficult for new contributors to understand
- Fix: Add detailed comments or link to `docs/` specifications

**Data Schema Evolution:**
- Files: `trading/runner.py:88-124` vs `trading/engine/loop.py`
- What's missing: Log DataFrame schema validation, migration handling
- Impact: Subtle schema differences cause downstream failures
- Fix: Define explicit schema, add validation

## Infinite Loops / Long-Running Processes

**Dashboard Event Loops:**
- Files: `trading/training_dashboard.py:261` - `while True:` with no timeout
- Why: Relies on matplotlib window close for exit
- Impact: Difficult to integrate into CI/CD or batch pipelines
- Fix: Add timeout parameter or explicit exit condition

**News API Retry Loop:**
- Files: `trading/scripts/news_slice.py:43` - `while True:` with retry logic
- Why: Handles transient API failures
- Impact: Could loop forever on persistent failures
- Fix: Add max retry count or timeout

## Duplicate Code Patterns

**Retry Wrapper:**
- Files: `trading/data_downloader.py` (6+ repeated patterns)
- Duplicate: Exponential backoff in `_binance_get`, `download_stooq`, `download_btc_*` functions
- Impact: Inconsistent retry behavior, difficult to maintain
- Fix: Extract to shared `retry_with_backoff()` utility

**Hysteresis Logic:**
- Files: `trading/strategy/triadic_strategy.py:50-78`, `trading/policy/thesis.py`
- Duplicate: Confidence gating logic repeated across layers
- Impact: Logic drift between modules
- Fix: Extract shared hysteresis abstraction

---

*Concerns audit: 2026-01-08*
*Update as issues are fixed or new ones discovered*

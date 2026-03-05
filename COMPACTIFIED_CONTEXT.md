# dashitest — Compactified Context

## Scope
- This file is a compact, durable snapshot for the dashitest repo (root).
- It summarizes current intent, implemented state, and the highest-value next steps.

## Intent (current)
- Keep trading stack epistemic gating PnL-free; evaluation = precision/recall on acceptable vs ACT.
- Keep CA/benchmark work as research-lab outputs, not trading inputs.
- Maintain reproducibility: timestamped outputs and documented run artifacts.

## Implemented (high-signal)
- Trading stack: `state → TriadicStrategy → Intent → Execution → Log → Dashboard`.
- Hysteresis gating (`tau_on > tau_off`) + `RegimeSpec` acceptable gate (PnL-free).
- `run_all.py` runs cached markets + optional live dashboard; logs `p_bad`/`bad_flag` for structural stress.
- Dashifine / tree diffusion / compression benchmarks have docs and scripts with timestamped outputs.
- Phase-3 quotient training in `dashilearn/bsmoe_train.py` logs JSON + plots per run.

## Key Docs
- `README.md` (project map + doc index)
- `docs/tree_diffusion_benchmark.md`
- `docs/phase3_quotient_learning.md`
- `docs/b2_acceptance.md`

## Recent Chat Sync (canonical archive)
- Trading diagnostics: ES/NQ proposals are flat; monitor logic is correct; next sprint = amplitude diagnostics.
- Formalizing kernel: capital kernel + Meta-Witness refusal rules; Phase-9 wiring before actions.
- dashiCORE: create Function Coverage Map + benchmarking harness with efficiency surfaces.

## Next Steps (short list)
- Implement proposal amplitude diagnostics (size/dir/score quantiles + correlations).
- Decide whether to wire Phase-9 capital kernel + Meta-Witness into stream daemon.
- Extend function coverage map + benchmark harness for dashiCORE.

## Assumptions
- Python 3.11+, NumPy + PyTest are available.
- No GPU dependency required for core correctness; Vulkan/JAX are reference/optional.

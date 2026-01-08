# Tree Diffusion Benchmark (Ultrametric Transport)

This benchmark targets dynamics on a tree/ultrametric space observed through a
Euclidean projection. It is designed to validate quotient-level learning under
projection artifacts. Context: `CONTEXT.md#L1055`, `CONTEXT.md#L7941`, and
the band/telescope identity in `CONTEXT.md#L11154`–`CONTEXT.md#L11216`.

## Goal

Demonstrate that a geometry-aware kernel can recover tree-structured transport
when Euclidean kernels see only a scrambled projection of the leaves.

## Setup

- Latent state: values on leaves of a balanced `p`-ary tree of depth `D`.
- Dynamics: diffusion along the tree via multiscale subtree averaging.
- Observation: leaf values permuted into a Euclidean vector (projection).
- Tree model: RBF in quotient feature space using `quotient_vector` (Option A in
  `CONTEXT.md#L7947`–`CONTEXT.md#L7961`).
- Optional init: restrict energy to a single detail band with
  `--init-band` (0=root, `D`=leaves) and `--init-band-scale` to stress
  depth-killing behavior.

## Evaluation

- One-step MSE (raw observed vectors).
- Rollout MSE (raw observed vectors).
- Quotient MSE (tree-level averages across depths).
- Tree-band quotient MSE (detail-band energies across depths).

## Adversarial init (depth-killing)

To force separation between Euclidean RBF and tree-quotient models, initialize
energy in a single detail band (tree-Haar sheet). This stress test directly
measures band kill rates and leakage. Context: `CONTEXT.md#L13434`–`CONTEXT.md#L13499`.

Flags:
- `--adv-band <int>`: band index (0=root, `D`=leaves).
- `--adv-style {haar,randphase,sparse,mix}`: band construction strategy.
- `--adv-sparse-m <int>`: number of active parent blocks (sparse only).
- `--adv-mix-band <int>` and `--adv-mix-eps <float>`: add a second band (mix only).
- `--adv-seed <int>`: RNG seed for adversarial construction.

`randphase` permutes the within-parent weights per block (zero-mean preserved).
`sparse` activates only `m` parent blocks (others zero). `mix` adds a smaller
band at a second depth to probe leakage (mix uses randphase for both bands).

## Adversarial operator (nonlinear band coupling)

Adversarial init isolates depth-killing, but true separation needs a dynamics
variant that couples bands nonlinearly so leakage and band-kill diagnostics are
meaningful. The recommended target is a diffusion-ish base operator with an
explicit band-coupling term:

```
x_{t+1} = A x_t + g(B(x_t))
```

where `B(x_t)` extracts band coefficients, `g` mixes them nonlinearly, and the
result is pushed back to leaves. Context: `CONTEXT.md#L17472`–`CONTEXT.md#L17512`.

## Next-phase design commitments (lock before coding)

These choices must be pinned down before implementing the operator toggle or
bridge-task evaluation. Context: `CONTEXT.md#L17930`–`CONTEXT.md#L18197`.

- Coupling scope: local/adjacent band coupling by default (global mixing is an
  explicit extension). Context: `CONTEXT.md#L17957`–`CONTEXT.md#L17966`.
- Adversary type: static (fixed operator across runs) before any adaptive
  variants.
- Bridge direction: coarse→fine vs fine→coarse must be explicitly chosen and
  scored. Context: `CONTEXT.md#L18085`–`CONTEXT.md#L18096`.
- Pass/fail: define relative thresholds (improvement vs baseline) rather than
  absolute error cutoffs. Context: `CONTEXT.md#L18186`–`CONTEXT.md#L18197`.

## Definitions: nonlinear band-coupled adversarial operator

This section pins down the operator and leakage metrics so the adversary and
diagnostics are fully specifiable. Context: `CONTEXT.md#L17934`–`CONTEXT.md#L18079`.

### Multiband state space

Define the multiband state as a list of band tensors:

```
x = [x^(0), x^(1), ..., x^(J)]
x^(j) in R^{Omega_j x C_j}
```

Assume fixed inter-band resize maps:

```
D_{j->k}: Omega_j x C_j -> Omega_k x C_j  (k < j)
U_{j->k}: Omega_j x C_j -> Omega_k x C_j  (k > j)
```

Declare the resize method (nearest/bilinear/area) to avoid ambiguity.

### Band coupling graph (local adjacency default)

Let bands be nodes in a directed graph with edges:

```
E = {(j, j-1), (j, j), (j, j+1)} where valid
N(j) = {k | (j,k) in E}
```

### Feature maps used for coupling

Align band `k` to band `j` with:

```
R_{k->j} = U_{k->j} if k < j
         = Id     if k = j
         = D_{k->j} if k > j
```

Optionally apply a per-site channel adapter `A_{k->j}` (e.g., 1x1 conv). Then:

```
m_{k->j}(x) = A_{k->j}(R_{k->j}(x^(k)))
M_j(x) = concat(m_{k->j}(x) for k in N(j))
```

### Nonlinear band-coupled operator

Define per-band gated residual updates:

```
Delta^(j)(x) = sigma(Phi^(j)(M_j(x))) * Psi^(j)(M_j(x))
A_theta(x)^(j) = x^(j) + eps_j * Delta^(j)(x)
```

where `Phi` produces gate logits, `Psi` produces update proposals, and `sigma`
is bounded (e.g., sigmoid). This is the explicit nonlinear coupling term that
augments diffusion. Context: `CONTEXT.md#L17996`–`CONTEXT.md#L18017`.

### Concrete instantiation (no mystery networks)

Use fixed-degree polynomials with local convolution:

```
Psi^(j)(M_j) = K^(j) * M_j
Phi^(j)(M_j) = P^(j)(K^(j) * M_j)
P^(j)(z) = a^(j) z + b^(j) z^2 + c^(j)
```

Declare kernel supports and polynomial degree. Context: `CONTEXT.md#L18021`–`CONTEXT.md#L18037`.

### Leakage metrics (band influence)

For `y = A_theta(x)`, define ablation influence:

```
I_{j<-k}(x) = ||y^(j) - y_{\\k}^(j)||_j / (||y^(j) - x^(j)||_j + delta)
```

with `x_{\\k}` zeroing or noise-replacing band `k`. Then:

```
Leak(j) = sum_{k != j} E[I_{j<-k}(X)]
NonLocalLeak = sum_j sum_{|k-j|>1} E[I_{j<-k}(X)]
```

Context: `CONTEXT.md#L18040`–`CONTEXT.md#L18079`.

## Acceptance tests: bridge task (two-sided inference)

The bridge task evaluates coarse→fine and fine→coarse inference under the
band-coupled adversary, with leakage-aware acceptance thresholds. Context:
`CONTEXT.md#L18083`–`CONTEXT.md#L18201`.

### Task definition

- Coarse→Fine: predict `x^(0)` from `x^(J)` (optionally with mid-bands).
- Fine→Coarse: predict `x^(J)` from `x^(0)`.

### Pseudocode (acceptance harness)

```python
def bridge_acceptance_test(
    dataset, adversary, learner_cf, learner_fc,
    metric_fine, metric_coarse, leakage_fn, thresholds
):
    stats = {"cf": [], "fc": [], "leak": [], "nonlocal": []}
    for x in dataset:
        x_adv = adversary(x)
        fine_pred = learner_cf.predict(x_adv[-1])
        coarse_pred = learner_fc.predict(x_adv[0])
        stats["cf"].append(metric_fine(fine_pred, x_adv[0]))
        stats["fc"].append(metric_coarse(coarse_pred, x_adv[-1]))
        leak = leakage_fn(x, adversary)
        stats["leak"].append(leak.get("LeakTotal", leak))
        stats["nonlocal"].append(leak.get("NonLocalLeak", 0.0))
    mean_cf = mean(stats["cf"])
    mean_fc = mean(stats["fc"])
    asym_gap = abs(mean_cf - mean_fc)
    corr_cf = corr(stats["cf"], stats["leak"])
    corr_fc = corr(stats["fc"], stats["leak"])
    accept = True
    accept &= mean_cf <= thresholds["max_mean_cf"]
    accept &= mean_fc <= thresholds["max_mean_fc"]
    accept &= asym_gap <= thresholds["max_asym_gap"]
    if "max_nonlocal_leak" in thresholds:
        accept &= mean(stats["nonlocal"]) <= thresholds["max_nonlocal_leak"]
    accept &= max(abs(corr_cf), abs(corr_fc)) >= thresholds["min_abs_corr_leak"]
    return {
        "accept": accept,
        "mean_cf": mean_cf,
        "mean_fc": mean_fc,
        "asym_gap": asym_gap,
        "corr_cf_leak": corr_cf,
        "corr_fc_leak": corr_fc,
        "mean_nonlocal_leak": mean(stats["nonlocal"]),
    }
```

### Thresholds and baselines

Set thresholds relative to a baseline:

- `max_mean_cf`: coarse→fine error <= baseline * (1 - improvement)
- `max_mean_fc`: fine→coarse error <= baseline * (1 - improvement)
- `max_asym_gap`: constrain directional asymmetry
- `min_abs_corr_leak`: leakage must correlate with error

Baselines:

- Coarse→Fine: upsample coarse + channel projection.
- Fine→Coarse: downsample fine.

Context: `CONTEXT.md#L18182`–`CONTEXT.md#L18200`.

## Required outputs (core dashboard)

Principle: every visual must answer exactly one learner question; if it does not
falsify a learning claim, it is optional. Context: `CONTEXT.md#L18329`–`CONTEXT.md#L18506`.

These plots make coupling, leakage, and bridge asymmetry explicit. Context:
`CONTEXT.md#L18207`–`CONTEXT.md#L18318`.

- Band-quotient curves over rollout time for each model/operator setting.
- Cross-band influence matrix heatmap (adversary on/off, optional diff).
- Leakage summaries: `LeakTotal`, `NonLocalLeak`, plus per-band `Leak(j)`.
- Bridge task error distributions for coarse→fine vs fine→coarse + asymmetry gap.
- Leakage vs bridge-error scatter with correlation coefficient.

Suggested filenames:

- `rollout_band_energy_{model}_{adv}.png`
- `influence_matrix_{adv}.png`
- `leakage_summary_{adv}.png`
- `bridge_errors_{adv}.png`
- `leakage_vs_bridge_error_{adv}.png`
- `killrate_{model}_{adv}.png`

When emitting large PNG grids (rollouts, band planes, or influence sweeps),
merge them into a GIF or WebM and remove the individual PNGs to keep outputs
manageable; this is a delivery/storage optimization, not a diagnostic. See
`README.md#L3`.

## Minimal learner-dashboard checklist

A run is learner-diagnosable if it outputs:

- Band-energy / band-quotient rollout curves.
- Cross-band influence matrix.
- Leakage summary (scalar + per-band).
- Bridge task errors (both directions).
- Leakage vs error correlation plot.

Everything else is secondary. Context: `CONTEXT.md#L18496`–`CONTEXT.md#L18506`.

## Strongly recommended (proof-like visuals)

These outputs turn the dashboard into a sharper diagnostic suite. Context:
`CONTEXT.md#L18275`–`CONTEXT.md#L18293`.

- Kill-rate / collapse-step plot per band (time-to-threshold or count of bands
  crossing a threshold per step).
- Operator sanity panel: sweep operator strength (`eps`) and report influence
  matrix sparsity, nonlocal leak, bridge error, and band-quotient curves.

## Nice-to-have (intuition/paper polish)

Use a small sample set (3–5 examples) to visualize band images or band energies
before/after the adversary, plus bridge reconstructions in both directions.
Context: `CONTEXT.md#L18296`–`CONTEXT.md#L18305`.

## Benchmark closure criteria

This benchmark is closed when the conditions in §1–§4 are satisfied. No
additional visuals, operators, or metrics may be added without opening a new
benchmark version. Context: `CONTEXT.md#L18525`–`CONTEXT.md#L18663`.

### §1. Design freeze (structural commitments)

The following choices must be fixed and documented (with flags defaulting as
stated):

1. Adversarial operator
   - Coupling scope: local / adjacent bands only.
   - Adversary type: static.
   - Nonlinearity: fixed polynomial degree and kernel support.
   - Per-band strength schedule (eps_j): fixed or explicitly parameterized.
2. Bridge task
   - Directions: coarse→fine and fine→coarse.
   - Inputs/targets: fixed band indices and representations.
   - Baselines: declared (upsample+proj; downsample).
3. Metrics
   - Band norm definition (e.g., L2).
   - Leakage ablation strategy (zero or noise).
   - Stability constant (delta).

Closure rule: once these are fixed, no further operator variants or metric
definitions are permitted within this benchmark.

### §2. Required learner-facing outputs

A benchmark run is valid iff it emits all of the following artifacts (filenames
may vary, content may not):

1. Band-energy / band-quotient rollout curves.
2. Cross-band influence matrix (E[I_{j<-k}]).
3. Leakage summaries: LeakTotal, NonLocalLeak, and per-band leakage bars.
4. Bridge-task results: error distributions for both directions + asymmetry gap.
5. Leakage↔error correlation plot.

Optional outputs (kill-rate plots, sweeps, sample visuals) must not be used for
acceptance decisions.

Closure rule: if any required output is missing, the run is invalid and cannot
be cited.

### §3. Acceptance logic (pass/fail)

Acceptance is defined relative to a declared baseline, not absolute scores. A
learner passes if all conditions hold:

1. Rollout structure
   - Band-quotient ordering preserved better than baseline.
   - No spurious cross-band energy inflation.
2. Attribution
   - Influence matrix significantly more local than baseline.
   - NonLocalLeak <= declared threshold (or <= baseline × factor).
3. Bridge task
   - Mean error in both directions <= baseline × (1 - margin).
   - Asymmetry gap <= declared maximum (unless asymmetry is the hypothesis).
4. Causal coherence
   - Bridge-task error correlates with measured leakage.
   - Correlation magnitude is non-trivial; sign documented.

Closure rule: once thresholds are declared and one learner passes and one fails
under identical conditions, the benchmark is decisive.

### §4. Reproducibility & stopping conditions

The benchmark is complete when:

- Results are reproducible across >=2 random seeds.
- Tree vs RBF (or declared baselines) show stable qualitative separation.
- No new failure modes appear when rerunning the same configuration.

At this point: no additional plots, ablations, or interpretation text are
required. Further exploration must occur in a new benchmark or extension, not
by modifying this one.

### §5. Non-goals (explicit exclusions)

The following are not criteria for closure:

- Visual aesthetics
- Runtime performance
- Hyperparameter optimality
- Absolute prediction accuracy
- Downstream task success

This benchmark evaluates epistemic structure learning, not task utility.

### One-line summary

The tree diffusion benchmark is closed once learner behavior under adversarial
band coupling is fully explained by band-local influence, quantified leakage,
and directional bridge-task performance, with clear separation from baselines.

## Lemma: gauge-equivalence collapse (quotient + band)

If both learners only depend on the quotient of tree-Haar detail bands, then
their predictions are identical in that observable (up to numerical
conditioning). This explains why band-quotient metrics match when the benchmark
is fully gauge-fixed. Full statement: `CONTEXT.md#L13516`–`CONTEXT.md#L13540`.

## Quotient metric

Define subtree averages at each depth `l` (root `l=0`, leaves `l=D`):

```
avg_l = mean of each subtree at depth l
Q(x) = concat(avg_0, avg_1, ..., avg_D)
```

Quotient error is `MSE(Q(x_hat), Q(x_true))`.

## Tree-intrinsic quotient (depth energy)

To avoid Euclidean leakage, also score a tree-intrinsic metric based on
depth-wise energy:

```
E_l(x) = mean over nodes at depth l of (node_value^2)
Q_tree(x) = [E_0, E_1, ..., E_D]
```

This captures transport along the hierarchy without depending on leaf ordering.

## Detail band energies (tree-Haar)

The depth-energy metric above uses nested averages, so it is a cumulative view
of scale. For a band/decomposition view (orthogonal residuals), define detail
bands using the projection identity from `CONTEXT.md#L11154`–`CONTEXT.md#L11182`
and the residual-plane mapping in `CONTEXT.md#L9499`–`CONTEXT.md#L9502`.

Let `s_l` be the lifted level-`l` average (expanded to leaf space). Then:

```
s_l = s_{l+1} + d_l
```

where `d_l` is the detail band (residual sheet) between levels `l` and `l+1`.
Band energy is `mean(d_l ** 2)` per level, which avoids double counting and
makes leakage checks meaningful.

Implementation detail (matches `tree_diffusion_bench.py` ordering where
`level=0` is the root and `level=D` is the leaves): define
`band_0 = avg_0` and `band_l = avg_l - repeat(avg_{l-1}, p)` for `l >= 1`.
Energies are reported as `mean(band_l ** 2)` for `l=0..D`.

The benchmark now reports both cumulative depth energies and band energies so
the existing plots remain intact while adding the diagnostic view.

## Sheet visualization mapping (codec vs tree)

The sheet identity in `CONTEXT.md#L11567`–`CONTEXT.md#L11621` makes explicit that
each prior sheet is the difference from the next coarser sheet. In codec terms,
the balanced-ternary plane dumps (`compression/video_bench.py --dump-planes`)
are direct visualizations of these detail bands. The original tree benchmark
plots were cumulative depth energies; the new tree-band energies match the
codec sheet interpretation without double counting.

To visualize tree sheets directly, run with `--dump-band-planes` to emit one
PNG per band per rollout step (codec-style). Default output includes:
- `norm`: symmetric per-band normalization (codec-style gauge),
- `energy`: normalized planes with height scaled by band energy,
- `ternary`: thresholded ternary planes (balanced-ternary view).

## Expected outcome

- Euclidean RBF: competitive one-step MSE in observed space, weaker quotient
  fidelity under rollout.
- Tree kernel: better quotient consistency; raw-space drift is acceptable.

## Null-separation note

If rollout curves and quotient metrics are identical between RBF and tree
models, treat this as a control case: the dynamics commute with the projection,
so ultrametric geometry is not activated. This confirms the benchmark is not
biased, but it is not a stress test. Introduce symmetry-breaking (depth-varying
diffusion, non-commuting observation map, or valuation-conditioned updates) to
force separation.

The CLI now exposes `--obs-map-mode` (choose between `permute_depth` and
`mix_depth`) together with `--obs-map-seed` (falls back to `--seed`) to build a
single deterministic depth-aware observation map. This map is reused during
dataset construction, rollout scoring, and the bridge task so the
`observe_depth ∘ F ≠ F ∘ observe_depth` condition described in
`CONTEXT.md#L23435-L23509` stays intact run after run, and we can credibly
attribute separation to non-commuting projections instead of stochastic noise.

## Interpretation (control case)

Identical one-step and rollout errors indicate the diffusion operator is
radially symmetric under the observation permutation, so Euclidean kernels can
represent the dynamics without access to tree geometry. This is a sanity
check, not a failure case.

## Discriminator checks (required)

Before interpreting results, verify the benchmark can actually distinguish
geometries:

- Log `rel_diff = ||K_rbf - K_tree|| / ||K_rbf||` to ensure kernels are not
  numerically identical.
- Report correlation between pairwise distance matrices (observed vs tree).
- Initialize with sparse impulses (localized leaves) to prevent immediate
  collapse onto a low-rank smooth manifold.

## Bridge task (two-sided inference)

The bridge task now runs when `--bridge-task` is enabled and `--bridge-task-T`
controls the horizon that determines `x_{T/2}`. Sliding windows of `(x0, x_T)`
are passed through the same deterministic observation map, and both RBF and tree
kernels are trained on the concatenated endpoints so they can infer the mid-step
state. We report raw, quotient, and tree-band quotient mean-squared errors;
the RBF tree-band error serves as the leakage proxy. These metrics are emitted
as `rbf_bridge_*` and `tree_bridge_*` in the JSON dump and capture the global
inference axis described in `CONTEXT.md#L23880-L23970`. Once the bridge task
shows Tree ≫ RBF keep the benchmark closed—no more adversaries, sweeps, or
visual additions beyond this context (`CONTEXT.md#L23924-L24020`).

To connect the benchmark to two-sided inference (fill-in-the-middle/smoothing),
add a bridge task: sample `x0`, roll to `xT`, hide intermediate steps, then ask
the learner to infer `x_{T/2}` (or band energies / band signs) from `(x0, xT)`.
Score in band-quotient space and report leakage. Context:
`CONTEXT.md#L17562`–`CONTEXT.md#L17600`.

This bridge task is the tree-diffusion analogue of smoothing vs filtering in
LLM inference, where training uses two-sided information but generation is
causal. Context: `CONTEXT.md#L17550`–`CONTEXT.md#L17558`.

## Permutation invariance caveat

If the observation map is a pure coordinate permutation and the “tree” model
still uses a Euclidean RBF kernel, then RBF on observed space and RBF on latent
space are equivalent up to reindexing. This yields identical one-step and
rollout errors. See `CONTEXT.md#L7905` through `CONTEXT.md#L7922`.

To break this equivalence, the tree model uses a quotient-aware metric (RBF on
`quotient_vector`; see `CONTEXT.md#L7947`–`CONTEXT.md#L7961`). Weighted depth-wise
features are still a future variant.

## Script

`tree_diffusion_bench.py` generates trajectories, trains KRR baselines (with
quotient-space features for the tree kernel), and emits one-step + rollout
metrics (raw and quotient) plus optional rollout plots (`--plots`):

- `outputs/tree_diffusion_rollout_mse_<timestamp>.png`
- `outputs/tree_diffusion_rollout_quotient_<timestamp>.png`
- `outputs/tree_diffusion_rollout_tree_band_quotient_<timestamp>.png`
- `outputs/tree_diffusion_band_planes/{label}_tree_bands/norm/png/band{band}_t{step}.png`
- `outputs/tree_diffusion_band_planes/{label}_tree_bands/energy/png/band{band}_t{step}.png`
- `outputs/tree_diffusion_band_planes/{label}_tree_bands/ternary/png/band{band}_t{step}.png`

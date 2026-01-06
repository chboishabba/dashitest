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

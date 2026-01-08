import argparse
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def build_weights(depth: int, decay: float) -> np.ndarray:
    levels = np.arange(depth + 1)
    weights = decay ** (depth - levels)
    weights /= weights.sum()
    return weights


def subtree_averages(x: np.ndarray, p: int, depth: int) -> list[np.ndarray]:
    if x.size != p ** depth:
        raise ValueError("x length must equal p**depth")
    averages = []
    for level in range(depth + 1):
        group = p ** (depth - level)
        reshaped = x.reshape(p ** level, group)
        averages.append(reshaped.mean(axis=1))
    return averages


def expand_level(avg: np.ndarray, p: int, depth: int, level: int) -> np.ndarray:
    group = p ** (depth - level)
    return np.repeat(avg, group)


def tree_diffusion_step(x: np.ndarray, p: int, depth: int, alpha: float, decay: float) -> np.ndarray:
    averages = subtree_averages(x, p, depth)
    weights = build_weights(depth, decay)
    mix = np.zeros_like(x)
    for level, avg in enumerate(averages):
        mix += weights[level] * expand_level(avg, p, depth, level)
    return (1.0 - alpha) * x + alpha * mix


def get_nl_fn(nl_str: str, threshold: float = 0.0):


    if nl_str == "tanh":


        return np.tanh


    elif nl_str == "sigmoid":


        return lambda z: 1 / (1 + np.exp(-z))


    elif nl_str == "sign":


        return np.sign


    elif nl_str == "threshold":


        return lambda z: np.where(np.abs(z) > threshold, z, 0.0)


    else:


        raise ValueError(f"unknown nonlinearity {nl_str}")





def tree_diffusion_step_adversarial(





    x: np.ndarray,





    p: int,





    depth: int,





    alpha: float,





    decay: float,





    adv_op_strength: float,





    adv_op_nl: str,





    adv_op_nl_threshold: float,





) -> np.ndarray:





    """Diffusion with non-linear, state-dependent, depth-varying weights AND explicit band coupling."""











    nl_fn = get_nl_fn(adv_op_nl, adv_op_nl_threshold)











    # 1. Compute modified weights for diffusion part (depth-dependent and state-modulated)





    band_energies = tree_band_energy_vector(x, p, depth)  # coarse to fine





    original_weights = build_weights(depth, decay)





    modified_weights = np.copy(original_weights)





    for level in range(depth + 1):


        modulation_factor = nl_fn(adv_op_strength * band_energies[level])


        # Add 1.0 to ensure positive weight and keep original if modulation is 0.


        modified_weights[level] *= np.clip(1.0 + modulation_factor, 0.0, None)





    # Compute adversarial mix using modified weights


    averages = subtree_averages(x, p, depth)


    modified_mix = np.zeros_like(x)


    for level, avg in enumerate(averages):


        modified_mix += modified_weights[level] * expand_level(avg, p, depth, level)


    


    # The base diffusion term with modified weights


    x_diffusion_modified = (1.0 - alpha) * x + alpha * modified_mix





    # 2. Compute non-linear band-coupling term (additive)


    bands = tree_detail_bands(x, p, depth)


    coupling_bands = [np.zeros_like(b) for b in bands]


    for l in range(1, depth + 1):


        parent_band_interaction = nl_fn(bands[l - 1])  # Using adv_op_nl


        parent_band_interaction_expanded = np.repeat(parent_band_interaction, p)


        coupling_bands[l] = parent_band_interaction_expanded * bands[l]


    


    coupling_term = leaf_from_bands(coupling_bands, p)


    


    # The final combined operator: modified diffusion + explicit coupling term


    return x_diffusion_modified + adv_op_strength * coupling_term








def rollout(








    x0: np.ndarray,








    steps: int,








    p: int,








    depth: int,








    alpha: float,








    decay: float,








    adv_op: bool,








    adv_op_strength: float,








    adv_op_nl: str,








    adv_op_nl_threshold: float,








    dump_live_sheet_path: Path = None,








) -> list[np.ndarray]:








    traj = [x0]








    x = x0








    if dump_live_sheet_path:








        _save_live_sheet(x, p, depth, dump_live_sheet_path)  # Save initial state








    for _ in range(steps):








        if adv_op:








            x = tree_diffusion_step_adversarial(








                x, p, depth, alpha, decay, adv_op_strength, adv_op_nl, adv_op_nl_threshold








            )








        else:








            x = tree_diffusion_step(x, p, depth, alpha, decay)








        traj.append(x)








        if dump_live_sheet_path:








            _save_live_sheet(x, p, depth, dump_live_sheet_path)








    return traj





def quotient_vector(x: np.ndarray, p: int, depth: int) -> np.ndarray:
    avgs = subtree_averages(x, p, depth)
    return np.concatenate(avgs, axis=0)


def tree_energy_vector(x: np.ndarray, p: int, depth: int) -> np.ndarray:
    avgs = subtree_averages(x, p, depth)
    return np.array([np.mean(level ** 2) for level in avgs], dtype=float)


def tree_detail_bands(x: np.ndarray, p: int, depth: int) -> list[np.ndarray]:
    """Tree-Haar style detail bands aligned to subtree_averages order."""
    avgs = subtree_averages(x, p, depth)  # level 0..depth, root->leaves
    bands = [avgs[0]]
    for level in range(1, depth + 1):
        parent = avgs[level - 1]
        expanded = np.repeat(parent, p)
        bands.append(avgs[level] - expanded)
    return bands


def tree_band_energy_vector(x: np.ndarray, p: int, depth: int) -> np.ndarray:
    """Per-level band energies (detail sheets), not cumulative energies."""
    bands = tree_detail_bands(x, p, depth)
    return np.array([np.mean(level ** 2) for level in bands], dtype=float)


def leaf_from_bands(bands: list[np.ndarray], p: int) -> np.ndarray:
    """Reconstruct leaf signal from tree-detail bands."""
    avgs = [bands[0]]
    for level in range(1, len(bands)):
        avgs.append(bands[level] + np.repeat(avgs[level - 1], p))
    return avgs[-1]


def _band_weights(p: int) -> np.ndarray:
    if p < 2:
        raise ValueError("p must be >= 2 for adversarial band construction")
    weights = np.zeros(p, dtype=float)
    weights[0] = 1.0
    weights[1] = -1.0
    norm = float(np.linalg.norm(weights))
    return weights if norm == 0.0 else weights / norm


def _build_band_detail(
    p: int,
    band: int,
    rng: np.random.Generator,
    style: str,
    sparse_m: int,
) -> np.ndarray:
    if band == 0:
        return rng.normal(size=1)
    parents = p ** (band - 1)
    weights = _band_weights(p)
    detail = np.zeros(p ** band, dtype=float)
    if style == "sparse":
        sparse_m = min(max(sparse_m, 0), parents)
        active = set(rng.choice(parents, size=sparse_m, replace=False).tolist())
    else:
        active = None
    for parent in range(parents):
        if active is not None and parent not in active:
            continue
        w = weights
        if style in ("randphase", "sparse"):
            w = rng.permutation(weights)
        start = parent * p
        detail[start : start + p] = w
    return detail


def make_adv_init(
    p: int,
    depth: int,
    band: int,
    style: str,
    sparse_m: int,
    mix_band: int,
    mix_eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if band < 0 or band > depth:
        raise ValueError("adv_band must be in [0, depth]")
    if style not in {"haar", "randphase", "sparse", "mix"}:
        raise ValueError("adv_style must be one of: haar, randphase, sparse, mix")
    base_style = "randphase" if style == "mix" else style
    bands = [np.zeros(p ** level, dtype=float) for level in range(depth + 1)]
    bands[band] = _build_band_detail(p, band, rng, base_style, sparse_m)
    if style == "mix":
        if mix_band < 0 or mix_band > depth:
            raise ValueError("adv_mix_band must be in [0, depth] for mix style")
        bands[mix_band] += mix_eps * _build_band_detail(p, mix_band, rng, "randphase", sparse_m)
    return leaf_from_bands(bands, p)

import math
def _save_live_sheet(x: np.ndarray, p: int, depth: int, path: Path) -> None:
    n_leaves = p ** depth
    side = int(math.sqrt(n_leaves))
    if side * side != n_leaves:
        # Fallback to a 1D representation or pad if not a perfect square
        # For now, let's assume it's a perfect square for simplicity or warn
        print(f"warning: n_leaves {n_leaves} is not a perfect square, can't reshape to 2D for live sheet.", file=sys.stderr)
        return

    sheet_data = x.reshape((side, side))
    np.save(path, sheet_data.astype(np.float32))

class ObservationMap:
    def __init__(self, mode: str, p: int, depth: int, seed: int | None):
        self.mode = mode
        self.p = p
        self.depth = depth
        self.n_leaves = p ** depth
        self.seed = seed
        self._permute_levels: dict[int, np.ndarray] = {}
        self._mix_shifts: dict[int, int] = {}
        if mode == "permute_depth":
            self._build_permutations()
        elif mode == "mix_depth":
            self._build_mix_shifts()
        elif mode != "none":
            raise ValueError(f"unknown observation mode {mode}")

    def _build_permutations(self) -> None:
        rng = np.random.default_rng(self.seed)
        for level in range(self.depth + 1):
            block_size = self.p ** (self.depth - level)
            if block_size <= 1:
                continue
            self._permute_levels[level] = rng.permutation(block_size)

    def _build_mix_shifts(self) -> None:
        rng = np.random.default_rng(self.seed)
        for level in range(self.depth + 1):
            block_size = self.p ** (self.depth - level)
            if block_size <= 1:
                continue
            self._mix_shifts[level] = int(rng.integers(1, block_size))

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return x
        if x.size != self.n_leaves:
            raise ValueError("observation map applied to unexpected shape")
        if self.mode == "permute_depth":
            return self._apply_permute(x)
        if self.mode == "mix_depth":
            return self._apply_mix(x)
        raise ValueError(f"unsupported observation mode {self.mode}")

    def _apply_permute(self, x: np.ndarray) -> np.ndarray:
        result = x.copy()
        for level, perm in self._permute_levels.items():
            block_size = perm.size
            for start in range(0, self.n_leaves, block_size):
                block = result[start : start + block_size]
                result[start : start + block_size] = block[perm]
        return result

    def _apply_mix(self, x: np.ndarray) -> np.ndarray:
        result = x.copy()
        for level, shift in self._mix_shifts.items():
            block_size = self.p ** (self.depth - level)
            if shift == 0:
                continue
            for start in range(0, self.n_leaves, block_size):
                block = result[start : start + block_size]
                result[start : start + block_size] = 0.5 * (block + np.roll(block, shift))
        return result

def _save_gray_png(path: Path, img_u8: np.ndarray) -> None:
    plt.imsave(path, img_u8, cmap="gray", vmin=0, vmax=255)


def dump_tree_band_images(
    out_dir: Path,
    label: str,
    rollout_lat: np.ndarray,
    p: int,
    depth: int,
    vis_mode: str,
    ternary_thresh: float,
    energy_alpha: float,
    energy_beta: float,
    energy_max_height: int,
) -> None:
    band_dir = out_dir / f"{label}_tree_bands"
    bands_per_step = [tree_detail_bands(x, p, depth) for x in rollout_lat]

    level_max_abs = []
    for level in range(depth + 1):
        values = np.concatenate([bands[level].ravel() for bands in bands_per_step])
        level_max_abs.append(float(np.max(np.abs(values))))

    modes = ["norm", "energy", "ternary"] if vis_mode == "all" else [vis_mode]
    png_dirs = {mode: band_dir / mode / "png" for mode in modes}
    for mode_dir in png_dirs.values():
        mode_dir.mkdir(parents=True, exist_ok=True)

    for t, bands in enumerate(bands_per_step):
        for level, band in enumerate(bands):
            max_abs = level_max_abs[level]
            if max_abs == 0.0:
                norm = np.full((1, band.size), 0.5, dtype=float)
            else:
                norm = (0.5 + 0.5 * (band / max_abs)).reshape(1, -1)
            norm_u8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)

            if "norm" in png_dirs:
                out_path = png_dirs["norm"] / f"band{level:02d}_t{t:04d}.png"
                _save_gray_png(out_path, norm_u8)

            if "energy" in png_dirs:
                energy = float(np.mean(band ** 2))
                height = int(np.clip(np.log10(energy + 1e-12) * energy_alpha + energy_beta, 1, energy_max_height))
                energy_img = np.repeat(norm_u8, height, axis=0)
                out_path = png_dirs["energy"] / f"band{level:02d}_t{t:04d}.png"
                _save_gray_png(out_path, energy_img)

            if "ternary" in png_dirs:
                if max_abs == 0.0:
                    ternary = np.zeros_like(band, dtype=np.int8)
                else:
                    ternary = (np.abs(band) > (ternary_thresh * max_abs)).astype(np.int8) * np.sign(band).astype(np.int8)
                lut = np.array([0, 127, 255], dtype=np.uint8)
                ternary_u8 = lut[ternary + 1].reshape(1, -1)
                out_path = png_dirs["ternary"] / f"band{level:02d}_t{t:04d}.png"
                _save_gray_png(out_path, ternary_u8)



def rbf_kernel(X: np.ndarray, Y: np.ndarray, lengthscale: float) -> np.ndarray:
    x2 = np.sum(X ** 2, axis=1)[:, None]
    y2 = np.sum(Y ** 2, axis=1)[None, :]
    d2 = x2 + y2 - 2.0 * (X @ Y.T)
    return np.exp(-d2 / (2.0 * lengthscale ** 2))


def krr_fit(X: np.ndarray, Y: np.ndarray, lengthscale: float, reg: float) -> dict:
    K = rbf_kernel(X, X, lengthscale)
    n = K.shape[0]
    alpha = np.linalg.solve(K + reg * np.eye(n), Y)
    return {"X": X, "alpha": alpha, "lengthscale": lengthscale}


def krr_predict(model: dict, X: np.ndarray) -> np.ndarray:
    K = rbf_kernel(X, model["X"], model["lengthscale"])
    return K @ model["alpha"]


def dataset_from_traj(
    traj: list[np.ndarray],
    perm: np.ndarray,
    p: int,
    depth: int,
    obs_map: ObservationMap,
) -> tuple[np.ndarray, np.ndarray]:


    X = []


    Y = []


    for t in range(len(traj) - 1):


        x_observed = obs_map.apply(traj[t])


        y_observed = obs_map.apply(traj[t + 1])


        X.append(x_observed[perm])


        Y.append(y_observed[perm])


    return np.asarray(X), np.asarray(Y)


def build_bridge_dataset(
    traj: list[np.ndarray],
    perm: np.ndarray,
    inv_perm: np.ndarray,
    p: int,
    depth: int,
    obs_map: ObservationMap,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if horizon < 2:
        raise ValueError("bridge-task horizon must be >= 2")
    if len(traj) <= horizon:
        raise ValueError("trajectory too short for the requested bridge horizon")
    step = horizon // 2
    if step < 1:
        raise ValueError("bridge-task horizon produces empty midpoint")
    windows = len(traj) - horizon

    obs_features = []
    tree_features = []
    target_obs = []
    target_lat = []
    for start in range(windows):
        x0_obs = obs_map.apply(traj[start])[perm]
        xT_obs = obs_map.apply(traj[start + horizon])[perm]
        x_mid_obs = obs_map.apply(traj[start + step])[perm]
        x0_lat = x0_obs[inv_perm]
        xT_lat = xT_obs[inv_perm]
        obs_features.append(np.concatenate([x0_obs, xT_obs]))
        tree_features.append(
            np.concatenate(
                [quotient_vector(x0_lat, p, depth), quotient_vector(xT_lat, p, depth)],
                axis=0,
            )
        )
        target_obs.append(x_mid_obs)
        target_lat.append(x_mid_obs[inv_perm])

    return (
        np.stack(obs_features),
        np.stack(tree_features),
        np.stack(target_obs),
        np.stack(target_lat),
    )





def score_rollout(
    model,
    x0: np.ndarray,
    steps: int,
    perm: np.ndarray,
    inv_perm: np.ndarray,
    p: int,
    depth: int,
    obs_map: ObservationMap,
    model_space: str,
):
    preds = []
    x_state = x0
    for _ in range(steps):
        x_pred_raw = krr_predict(model, x_state[None, :])[0]
        preds.append(x_pred_raw) # Store raw predictions for rollout
        x_state = x_pred_raw # Next state is the raw prediction
    preds = np.asarray(preds)

    # Apply observation map to the entire rollout AFTER generation for scoring
    if obs_map.mode != "none":
        preds_observed = np.array([obs_map.apply(x_val) for x_val in preds])
    else:
        preds_observed = preds
    
    if model_space == "latent":
        preds_latent = preds_observed
        preds_obs = preds_observed[:, perm]
    else:
        preds_obs = preds_observed
        preds_latent = preds_observed[:, inv_perm]
    return preds_obs, preds_latent


def score_rollout_quotient(
    model,
    x0_lat: np.ndarray,
    steps: int,
    p: int,
    depth: int,
    perm: np.ndarray,
    inv_perm: np.ndarray,
    obs_map: ObservationMap,
):
    preds_lat = []
    x_state = x0_lat
    for _ in range(steps):
        q_state = quotient_vector(x_state, p, depth)
        x_pred_lat_raw = krr_predict(model, q_state[None, :])[0]
        preds_lat.append(x_pred_lat_raw)
        x_state = x_pred_lat_raw # Next state is the raw prediction in latent space
    preds_lat = np.asarray(preds_lat)

    # Apply observation map to the entire rollout AFTER generation for scoring
    if obs_map.mode != "none":
        preds_lat_observed = np.array([obs_map.apply(x_val) for x_val in preds_lat])
    else:
        preds_lat_observed = preds_lat
    
    preds_obs = preds_lat_observed[:, perm]
    preds_lat = preds_lat_observed
    return preds_obs, preds_lat


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def run_benchmark(args):
    rng = np.random.default_rng(args.seed)
    adv_rng = np.random.default_rng(args.seed if args.adv_seed is None else args.adv_seed)
    obs_map_seed = args.obs_map_seed if args.obs_map_seed is not None else args.seed
    obs_map = ObservationMap(args.obs_map_mode, args.p, args.depth, obs_map_seed)
    n = args.p ** args.depth
    perm = rng.permutation(n)
    inv_perm = np.argsort(perm)

    if args.adv_band >= 0:
        x0 = make_adv_init(
            args.p,
            args.depth,
            args.adv_band,
            args.adv_style,
            args.adv_sparse_m,
            args.adv_mix_band,
            args.adv_mix_eps,
            adv_rng,
        )
        std = float(np.std(x0))
        if std > 0.0:
            x0 = x0 / std
        x0 = x0 * args.init_band_scale
    elif args.init_band >= 0:
        if args.init_band > args.depth:
            raise ValueError("init_band must be in [0, depth]")
        bands = [np.zeros(args.p ** level, dtype=float) for level in range(args.depth + 1)]
        bands[args.init_band] = rng.normal(size=args.p ** args.init_band)
        x0 = leaf_from_bands(bands, args.p)
        std = float(np.std(x0))
        if std > 0.0:
            x0 = x0 / std
        x0 = x0 * args.init_band_scale
    else:
        x0 = rng.normal(size=n)
    traj = rollout(
        x0,
        args.steps,
        args.p,
        args.depth,
        args.alpha,
        args.decay,
        args.adv_op,
        args.adv_op_strength,
        args.adv_op_nl,
        args.adv_op_nl_threshold,
        dump_live_sheet_path=Path(args.dump_live_sheet) if args.dump_live_sheet else None,
    )

    X_obs, Y_obs = dataset_from_traj(
        traj, perm, args.p, args.depth, obs_map
    )
    split = min(args.train, X_obs.shape[0])
    X_train, Y_train = X_obs[:split], Y_obs[:split]
    X_test, Y_test = X_obs[split:], Y_obs[split:]

    # Baseline: RBF on observed vectors
    rbf_model = krr_fit(X_train, Y_train, args.rbf_ls, args.reg)
    rbf_pred = krr_predict(rbf_model, X_test)

    # Tree kernel: use quotient features for geometry-aware learning
    X_train_lat = X_train[:, inv_perm]
    Y_train_lat = Y_train[:, inv_perm]
    X_test_lat = X_test[:, inv_perm]
    Y_test_lat = Y_test[:, inv_perm]
    Q_train = np.stack([quotient_vector(x, args.p, args.depth) for x in X_train_lat])
    Q_test = np.stack([quotient_vector(x, args.p, args.depth) for x in X_test_lat])
    tree_model = krr_fit(Q_train, Y_train_lat, args.tree_ls, args.reg)
    tree_pred_lat = krr_predict(tree_model, Q_test)
    tree_pred_obs = tree_pred_lat[:, perm]

    # One-step metrics
    rbf_mse = mse(rbf_pred, Y_test)
    tree_mse = mse(tree_pred_obs, Y_test)

    # Quotient metrics (one-step)
    rbf_q = mse(
        np.stack([quotient_vector(y[inv_perm], args.p, args.depth) for y in rbf_pred]),
        np.stack([quotient_vector(y[inv_perm], args.p, args.depth) for y in Y_test]),
    )
    tree_q = mse(
        np.stack([quotient_vector(y, args.p, args.depth) for y in tree_pred_lat]),
        np.stack([quotient_vector(y, args.p, args.depth) for y in Y_test_lat]),
    )
    rbf_tree_q = mse(
        np.stack([tree_energy_vector(y[inv_perm], args.p, args.depth) for y in rbf_pred]),
        np.stack([tree_energy_vector(y[inv_perm], args.p, args.depth) for y in Y_test]),
    )
    tree_tree_q = mse(
        np.stack([tree_energy_vector(y, args.p, args.depth) for y in tree_pred_lat]),
        np.stack([tree_energy_vector(y, args.p, args.depth) for y in Y_test_lat]),
    )
    rbf_tree_band_q = mse(
        np.stack([tree_band_energy_vector(y[inv_perm], args.p, args.depth) for y in rbf_pred]),
        np.stack([tree_band_energy_vector(y[inv_perm], args.p, args.depth) for y in Y_test]),
    )
    tree_tree_band_q = mse(
        np.stack([tree_band_energy_vector(y, args.p, args.depth) for y in tree_pred_lat]),
        np.stack([tree_band_energy_vector(y, args.p, args.depth) for y in Y_test_lat]),
    )

    # Rollout from first test sample
    if X_test.shape[0] > 0:
        x0_obs = X_test[0]
        rbf_roll_obs, rbf_roll_lat = score_rollout(
            rbf_model, x0_obs, args.rollout_steps, perm, inv_perm,
            args.p, args.depth, obs_map, "obs"
        )
        tree_roll_obs, tree_roll_lat = score_rollout_quotient(
            tree_model, x0_obs[inv_perm], args.rollout_steps, args.p, args.depth, perm, inv_perm,
            obs_map
        )
        true_roll = rollout(
            x0_obs[inv_perm],
            args.rollout_steps,
            args.p,
            args.depth,
            args.alpha,
            args.decay,
            args.adv_op,
            args.adv_op_strength,
            args.adv_op_nl,
            args.adv_op_nl_threshold,
            dump_live_sheet_path=None, # Don't dump true rollout to same file
        )
        true_roll_lat = np.asarray(true_roll[1:])
        true_roll_obs = true_roll_lat[:, perm]

        rbf_roll_mse = mse(rbf_roll_obs, true_roll_obs)
        tree_roll_mse = mse(tree_roll_obs, true_roll_obs)
        rbf_roll_q = mse(
            np.stack([quotient_vector(x, args.p, args.depth) for x in rbf_roll_lat]),
            np.stack([quotient_vector(x, args.p, args.depth) for x in true_roll_lat]),
        )
        tree_roll_q = mse(
            np.stack([quotient_vector(x, args.p, args.depth) for x in tree_roll_lat]),
            np.stack([quotient_vector(x, args.p, args.depth) for x in true_roll_lat]),
        )
        rbf_roll_tree_q = mse(
            np.stack([tree_energy_vector(x, args.p, args.depth) for x in rbf_roll_lat]),
            np.stack([tree_energy_vector(x, args.p, args.depth) for x in true_roll_lat]),
        )
        tree_roll_tree_q = mse(
            np.stack([tree_energy_vector(x, args.p, args.depth) for x in tree_roll_lat]),
            np.stack([tree_energy_vector(x, args.p, args.depth) for x in true_roll_lat]),
        )
        rbf_roll_tree_band_q = mse(
            np.stack([tree_band_energy_vector(x, args.p, args.depth) for x in rbf_roll_lat]),
            np.stack([tree_band_energy_vector(x, args.p, args.depth) for x in true_roll_lat]),
        )
        tree_roll_tree_band_q = mse(
            np.stack([tree_band_energy_vector(x, args.p, args.depth) for x in tree_roll_lat]),
            np.stack([tree_band_energy_vector(x, args.p, args.depth) for x in true_roll_lat]),
        )
        rbf_roll_curve = np.mean((rbf_roll_obs - true_roll_obs) ** 2, axis=1)
        tree_roll_curve = np.mean((tree_roll_obs - true_roll_obs) ** 2, axis=1)
        rbf_roll_q_curve = np.mean(
            (np.stack([quotient_vector(x, args.p, args.depth) for x in rbf_roll_lat])
             - np.stack([quotient_vector(x, args.p, args.depth) for x in true_roll_lat])) ** 2,
            axis=1,
        )
        tree_roll_q_curve = np.mean(
            (np.stack([quotient_vector(x, args.p, args.depth) for x in tree_roll_lat])
             - np.stack([quotient_vector(x, args.p, args.depth) for x in true_roll_lat])) ** 2,
            axis=1,
        )
        rbf_roll_tree_q_curve = np.mean(
            (np.stack([tree_energy_vector(x, args.p, args.depth) for x in rbf_roll_lat])
             - np.stack([tree_energy_vector(x, args.p, args.depth) for x in true_roll_lat])) ** 2,
            axis=1,
        )
        tree_roll_tree_q_curve = np.mean(
            (np.stack([tree_energy_vector(x, args.p, args.depth) for x in tree_roll_lat])
             - np.stack([tree_energy_vector(x, args.p, args.depth) for x in true_roll_lat])) ** 2,
            axis=1,
        )
        rbf_roll_tree_band_q_curve = np.mean(
            (np.stack([tree_band_energy_vector(x, args.p, args.depth) for x in rbf_roll_lat])
             - np.stack([tree_band_energy_vector(x, args.p, args.depth) for x in true_roll_lat])) ** 2,
            axis=1,
        )
        tree_roll_tree_band_q_curve = np.mean(
            (np.stack([tree_band_energy_vector(x, args.p, args.depth) for x in tree_roll_lat])
             - np.stack([tree_band_energy_vector(x, args.p, args.depth) for x in true_roll_lat])) ** 2,
            axis=1,
        )
        if args.dump_band_planes is not None:
            out_dir = Path(args.dump_band_planes)
            out_dir.mkdir(parents=True, exist_ok=True)
            dump_tree_band_images(
                out_dir,
                "true",
                true_roll_lat,
                args.p,
                args.depth,
                args.band_vis,
                args.band_ternary_threshold,
                args.band_energy_alpha,
                args.band_energy_beta,
                args.band_energy_max_height,
            )
            dump_tree_band_images(
                out_dir,
                "rbf",
                rbf_roll_lat,
                args.p,
                args.depth,
                args.band_vis,
                args.band_ternary_threshold,
                args.band_energy_alpha,
                args.band_energy_beta,
                args.band_energy_max_height,
            )
            dump_tree_band_images(
                out_dir,
                "tree",
                tree_roll_lat,
                args.p,
                args.depth,
                args.band_vis,
                args.band_ternary_threshold,
                args.band_energy_alpha,
                args.band_energy_beta,
                args.band_energy_max_height,
            )
            print(f"Saved tree band plane PNGs under {out_dir}")
    else:
        rbf_roll_mse = np.nan
        tree_roll_mse = np.nan
        rbf_roll_q = np.nan
        tree_roll_q = np.nan
        rbf_roll_tree_q = np.nan
        tree_roll_tree_q = np.nan
        rbf_roll_tree_band_q = np.nan
        tree_roll_tree_band_q = np.nan
        rbf_roll_curve = None
        tree_roll_curve = None
        rbf_roll_q_curve = None
        tree_roll_q_curve = None
        rbf_roll_tree_q_curve = None
        tree_roll_tree_q_curve = None
        rbf_roll_tree_band_q_curve = None
        tree_roll_tree_band_q_curve = None

    bridge_metrics = {
        "rbf_bridge_mse": np.nan,
        "tree_bridge_mse": np.nan,
        "rbf_bridge_q_mse": np.nan,
        "tree_bridge_q_mse": np.nan,
        "rbf_bridge_tree_band_q_mse": np.nan,
        "tree_bridge_tree_band_q_mse": np.nan,
        "bridge_windows": 0,
        "bridge_train_samples": 0,
        "bridge_test_samples": 0,
    }
    if args.bridge_task:
        (
            X_bridge_obs,
            X_bridge_tree,
            Y_bridge_obs,
            Y_bridge_lat,
        ) = build_bridge_dataset(
            traj, perm, inv_perm, args.p, args.depth, obs_map, args.bridge_task_T
        )
        bridge_windows = X_bridge_obs.shape[0]
        if bridge_windows < 2:
            raise ValueError("bridge task needs at least two windows for train/test split")
        bridge_train = max(1, min(args.train, bridge_windows - 1))
        X_bridge_obs_train = X_bridge_obs[:bridge_train]
        Y_bridge_obs_train = Y_bridge_obs[:bridge_train]
        X_bridge_obs_test = X_bridge_obs[bridge_train:]
        Y_bridge_obs_test = Y_bridge_obs[bridge_train:]
        X_bridge_tree_train = X_bridge_tree[:bridge_train]
        X_bridge_tree_test = X_bridge_tree[bridge_train:]
        Y_bridge_lat_train = Y_bridge_lat[:bridge_train]
        Y_bridge_lat_test = Y_bridge_lat[bridge_train:]
        rbf_bridge_model = krr_fit(X_bridge_obs_train, Y_bridge_obs_train, args.rbf_ls, args.reg)
        tree_bridge_model = krr_fit(X_bridge_tree_train, Y_bridge_lat_train, args.tree_ls, args.reg)
        rbf_bridge_pred = krr_predict(rbf_bridge_model, X_bridge_obs_test)
        tree_bridge_pred_lat = krr_predict(tree_bridge_model, X_bridge_tree_test)
        tree_bridge_pred_obs = tree_bridge_pred_lat[:, perm]
        rbf_bridge_pred_lat = rbf_bridge_pred[:, inv_perm]
        true_bridge_lat = Y_bridge_lat_test

        rbf_bridge_mse = mse(rbf_bridge_pred, Y_bridge_obs_test)
        tree_bridge_mse = mse(tree_bridge_pred_obs, Y_bridge_obs_test)
        rbf_bridge_q = mse(
            np.stack([quotient_vector(pred, args.p, args.depth) for pred in rbf_bridge_pred_lat]),
            np.stack([quotient_vector(target, args.p, args.depth) for target in true_bridge_lat]),
        )
        tree_bridge_q = mse(
            np.stack([quotient_vector(pred, args.p, args.depth) for pred in tree_bridge_pred_lat]),
            np.stack([quotient_vector(target, args.p, args.depth) for target in true_bridge_lat]),
        )
        rbf_bridge_tree_band_q = mse(
            np.stack([tree_band_energy_vector(pred, args.p, args.depth) for pred in rbf_bridge_pred_lat]),
            np.stack([tree_band_energy_vector(target, args.p, args.depth) for target in true_bridge_lat]),
        )
        tree_bridge_tree_band_q = mse(
            np.stack([tree_band_energy_vector(pred, args.p, args.depth) for pred in tree_bridge_pred_lat]),
            np.stack([tree_band_energy_vector(target, args.p, args.depth) for target in true_bridge_lat]),
        )
        bridge_metrics = {
            "rbf_bridge_mse": rbf_bridge_mse,
            "tree_bridge_mse": tree_bridge_mse,
            "rbf_bridge_q_mse": rbf_bridge_q,
            "tree_bridge_q_mse": tree_bridge_q,
            "rbf_bridge_tree_band_q_mse": rbf_bridge_tree_band_q,
            "tree_bridge_tree_band_q_mse": tree_bridge_tree_band_q,
            "bridge_windows": bridge_windows,
            "bridge_train_samples": bridge_train,
            "bridge_test_samples": bridge_windows - bridge_train,
        }

    metrics = {
        "rbf_one_step_mse": rbf_mse,
        "tree_one_step_mse": tree_mse,
        "rbf_one_step_q_mse": rbf_q,
        "tree_one_step_q_mse": tree_q,
        "rbf_one_step_tree_q_mse": rbf_tree_q,
        "tree_one_step_tree_q_mse": tree_tree_q,
        "rbf_one_step_tree_band_q_mse": rbf_tree_band_q,
        "tree_one_step_tree_band_q_mse": tree_tree_band_q,
        "rbf_rollout_mse": rbf_roll_mse,
        "tree_rollout_mse": tree_roll_mse,
        "rbf_rollout_q_mse": rbf_roll_q,
        "tree_rollout_q_mse": tree_roll_q,
        "rbf_rollout_tree_q_mse": rbf_roll_tree_q,
        "tree_rollout_tree_q_mse": tree_roll_tree_q,
        "rbf_rollout_tree_band_q_mse": rbf_roll_tree_band_q,
        "tree_rollout_tree_band_q_mse": tree_roll_tree_band_q,
        "n_leaves": n,
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
    }
    metrics.update(bridge_metrics)
    curves = {
        "rbf_roll_curve": rbf_roll_curve,
        "tree_roll_curve": tree_roll_curve,
        "rbf_roll_q_curve": rbf_roll_q_curve,
        "tree_roll_q_curve": tree_roll_q_curve,
        "rbf_roll_tree_q_curve": rbf_roll_tree_q_curve,
        "tree_roll_tree_q_curve": tree_roll_tree_q_curve,
        "rbf_roll_tree_band_q_curve": rbf_roll_tree_band_q_curve,
        "tree_roll_tree_band_q_curve": tree_roll_tree_band_q_curve,
    }
    return metrics, curves


def plot_rollout_curves(curves: dict, out_prefix: Path) -> None:
    if curves["rbf_roll_curve"] is None:
        return
    steps = np.arange(1, curves["rbf_roll_curve"].shape[0] + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, curves["rbf_roll_curve"], label="RBF")
    plt.plot(steps, curves["tree_roll_curve"], label="Tree")
    plt.xlabel("rollout step")
    plt.ylabel("MSE (raw)")
    plt.title("Rollout MSE (observed)")
    plt.legend()
    plt.tight_layout()
    out_path = out_prefix.with_name(f"{out_prefix.name}_rollout_mse.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")

    plt.figure(figsize=(7, 4))
    plt.plot(steps, curves["rbf_roll_q_curve"], label="RBF")
    plt.plot(steps, curves["tree_roll_q_curve"], label="Tree")
    plt.xlabel("rollout step")
    plt.ylabel("MSE (quotient)")
    plt.title("Rollout MSE (quotient)")
    plt.legend()
    plt.tight_layout()
    out_path = out_prefix.with_name(f"{out_prefix.name}_rollout_quotient.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")

    plt.figure(figsize=(7, 4))
    plt.plot(steps, curves["rbf_roll_tree_q_curve"], label="RBF")
    plt.plot(steps, curves["tree_roll_tree_q_curve"], label="Tree")
    plt.xlabel("rollout step")
    plt.ylabel("MSE (tree quotient)")
    plt.title("Rollout MSE (tree quotient)")
    plt.legend()
    plt.tight_layout()
    out_path = out_prefix.with_name(f"{out_prefix.name}_rollout_tree_quotient.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")

    plt.figure(figsize=(7, 4))
    plt.plot(steps, curves["rbf_roll_tree_band_q_curve"], label="RBF")
    plt.plot(steps, curves["tree_roll_tree_band_q_curve"], label="Tree")
    plt.xlabel("rollout step")
    plt.ylabel("MSE (tree band quotient)")
    plt.title("Rollout MSE (tree band quotient)")
    plt.legend()
    plt.tight_layout()
    out_path = out_prefix.with_name(f"{out_prefix.name}_rollout_tree_band_quotient.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=3, help="branching factor")
    ap.add_argument("--depth", type=int, default=6, help="tree depth (leaves = p^depth)")
    ap.add_argument("--steps", type=int, default=300, help="trajectory steps")
    ap.add_argument("--train", type=int, default=200, help="train samples")
    ap.add_argument("--alpha", type=float, default=0.4, help="diffusion mixing")
    ap.add_argument("--decay", type=float, default=0.6, help="weight decay across levels")
    ap.add_argument("--rbf-ls", type=float, default=1.0, help="RBF lengthscale (observed)")
    ap.add_argument("--tree-ls", type=float, default=1.0, help="tree kernel lengthscale")
    ap.add_argument("--reg", type=float, default=1e-3, help="ridge regularization")
    ap.add_argument("--rollout-steps", type=int, default=50, help="rollout length")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    ap.add_argument("--out", type=str, default="outputs/tree_diffusion_metrics.json")
    ap.add_argument("--plots", action="store_true", help="write rollout plots")
    ap.add_argument(
        "--init-band",
        type=int,
        default=-1,
        help="Initialize with energy only in this band (0=root, depth=leaves).",
    )
    ap.add_argument(
        "--init-band-scale",
        type=float,
        default=1.0,
        help="Scale for init-band signals after unit-variance normalization.",
    )
    ap.add_argument(
        "--adv-op",
        action="store_true",
        help="Enable adversarial nonlinear band-coupling operator in dynamics.",
    )
    ap.add_argument(
        "--adv-op-strength",
        type=float,
        default=0.1,
        help="Strength of the adversarial operator (non-linear modulation of weights).",
    )
    ap.add_argument(
        "--adv-op-nl",
        choices=["tanh", "sigmoid", "sign", "threshold"],
        default="tanh",
        help="Nonlinearity for the adversarial operator.",
    )
    ap.add_argument(
        "--adv-op-nl-threshold",
        type=float,
        default=0.1,
        help="Threshold value for 'threshold' nonlinearity.",
    )
    ap.add_argument(
        "--adv-band",
        type=int,
        default=-1,
        help="Adversarial init band (overrides --init-band when set).",
    )
    ap.add_argument(
        "--adv-style",
        choices=["haar", "randphase", "sparse", "mix"],
        default="randphase",
        help="Adversarial band construction style.",
    )
    ap.add_argument(
        "--adv-sparse-m",
        type=int,
        default=64,
        help="Active parent blocks for adv sparse style.",
    )
    ap.add_argument(
        "--adv-mix-band",
        type=int,
        default=-1,
        help="Secondary band index for adv mix style.",
    )
    ap.add_argument(
        "--adv-mix-eps",
        type=float,
        default=0.05,
        help="Secondary band scale for adv mix style.",
    )
    ap.add_argument(
        "--adv-seed",
        type=int,
        default=None,
        help="RNG seed override for adversarial init (defaults to --seed).",
    )
    ap.add_argument(
        "--dump-live-sheet",
        type=str,
        default=None,
        help="Path to dump a 2D representation of the tree state for live preview.",
    )
    ap.add_argument(
        "--dump-band-planes",
        type=str,
        help="Output dir for tree-band plane PNGs (one per band per rollout step).",
    )
    ap.add_argument(
        "--band-vis",
        choices=["norm", "energy", "ternary", "all"],
        default="all",
        help="Band visualization mode for --dump-band-planes.",
    )
    ap.add_argument(
        "--band-ternary-threshold",
        type=float,
        default=0.05,
        help="Relative threshold (fraction of band max) for ternary band planes.",
    )
    ap.add_argument(
        "--band-energy-alpha",
        type=float,
        default=6.0,
        help="Energy-height scaling alpha for band-plane visualization.",
    )
    ap.add_argument(
        "--band-energy-beta",
        type=float,
        default=6.0,
        help="Energy-height scaling beta for band-plane visualization.",
    )
    ap.add_argument(
        "--band-energy-max-height",
        type=int,
        default=16,
        help="Max row height for energy-scaled band planes.",
    )
    ap.add_argument(
        "--bridge-task",
        action="store_true",
        help="Enable bridge-task evaluation (infer x_T/2 from x0, x_T).",
    )
    ap.add_argument(
        "--bridge-task-T",
        type=int,
        default=50,
        help="Total steps T for bridge task (x0 to x_T).",
    )
    ap.add_argument(
        "--obs-map-mode",
        choices=["none", "permute_depth", "mix_depth"],
        default="none",
        help="Mode for non-commuting depth-dependent observation map.",
    )
    ap.add_argument(
        "--obs-map-seed",
        type=int,
        default=None,
        help="RNG seed for observation map (defaults to --seed).",
    )

    args = ap.parse_args()

    metrics, curves = run_benchmark(args)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")
    out_path.write_text(
        "{\n" + ",\n".join([f'  "{k}": {metrics[k]!r}' for k in sorted(metrics)]) + "\n}\n"
    )
    print(f"Saved {out_path}")
    if args.plots:
        plot_rollout_curves(curves, out_path.with_suffix(""))


if __name__ == "__main__":
    main()

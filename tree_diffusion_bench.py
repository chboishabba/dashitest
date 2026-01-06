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


def rollout(x0: np.ndarray, steps: int, p: int, depth: int, alpha: float, decay: float) -> list[np.ndarray]:
    traj = [x0]
    x = x0
    for _ in range(steps):
        x = tree_diffusion_step(x, p, depth, alpha, decay)
        traj.append(x)
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


def dataset_from_traj(traj: list[np.ndarray], perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = []
    Y = []
    for t in range(len(traj) - 1):
        X.append(traj[t][perm])
        Y.append(traj[t + 1][perm])
    return np.asarray(X), np.asarray(Y)


def score_rollout(model, x0, steps, perm, inv_perm, model_space: str):
    preds = []
    x_state = x0
    for _ in range(steps):
        x_pred = krr_predict(model, x_state[None, :])[0]
        preds.append(x_pred)
        x_state = x_pred
    preds = np.asarray(preds)
    if model_space == "latent":
        preds_latent = preds
        preds_obs = preds[:, perm]
    else:
        preds_obs = preds
        preds_latent = preds[:, inv_perm]
    return preds_obs, preds_latent


def score_rollout_quotient(model, x0_lat, steps, p, depth, perm):
    preds_lat = []
    x_state = x0_lat
    for _ in range(steps):
        q_state = quotient_vector(x_state, p, depth)
        x_pred = krr_predict(model, q_state[None, :])[0]
        preds_lat.append(x_pred)
        x_state = x_pred
    preds_lat = np.asarray(preds_lat)
    preds_obs = preds_lat[:, perm]
    return preds_obs, preds_lat


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def run_benchmark(args):
    rng = np.random.default_rng(args.seed)
    adv_rng = np.random.default_rng(args.seed if args.adv_seed is None else args.adv_seed)
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
    traj = rollout(x0, args.steps, args.p, args.depth, args.alpha, args.decay)

    X_obs, Y_obs = dataset_from_traj(traj, perm)
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
            rbf_model, x0_obs, args.rollout_steps, perm, inv_perm, "obs"
        )
        tree_roll_obs, tree_roll_lat = score_rollout_quotient(
            tree_model, x0_obs[inv_perm], args.rollout_steps, args.p, args.depth, perm
        )
        true_roll = rollout(x0_obs[inv_perm], args.rollout_steps, args.p, args.depth, args.alpha, args.decay)
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

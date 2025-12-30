"""
compute_policy_distance.py
--------------------------
Compare engagement geometry and dynamics between two logs.
Inputs: two CSVs each with columns actionability, acceptable, action (action!=0 -> ACT).
State space: (acceptable ∈ {0,1}) × actionability bins.
Metrics:
  - Occupancy-weighted JS and L1/L2 distances between engagement policies π(x)=P(ACT|x)
  - Boundary-weighted L1 using legitimacy margin (optional)
  - Occupancy-weighted L2 distance between transition kernels K(x→x')

Usage:
  PYTHONPATH=. python trading/scripts/compute_policy_distance.py --a logs/trading_log.csv --b logs/trading_log.csv
"""

import argparse
import numpy as np
import pandas as pd
from trading.regime import RegimeSpec


def load_states(path, bins):
    df = pd.read_csv(path)
    for col in ("actionability", "acceptable", "action"):
        if col not in df.columns:
            raise SystemExit(f"{path} missing required column '{col}'")
    a = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)
    acc = df["acceptable"].astype(bool).to_numpy()
    act_flag = (df["action"] != 0).to_numpy(dtype=bool)

    mask = np.isfinite(a)
    a = a[mask]
    acc = acc[mask]
    act_flag = act_flag[mask]

    edges = np.linspace(0.0, 1.0, bins + 1)
    a_bin = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
    state_idx = acc.astype(int) * bins + a_bin  # 0..2*bins-1

    return state_idx, act_flag, 2 * bins


def legitimacy_margin(df: pd.DataFrame, spec: RegimeSpec):
    # simple margin: distance to min_run_length; optional flip/vol not used unless provided
    states = pd.to_numeric(df["state"], errors="coerce").fillna(0).to_numpy(dtype=int)
    prices = pd.to_numeric(df["price"], errors="coerce").fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
    runs = np.zeros(len(states), dtype=int)
    run = 0
    for i, s in enumerate(states):
        if s == 0:
            run = 0
        else:
            if i > 0 and s == states[i - 1]:
                run += 1
            else:
                run = 1
        runs[i] = run
    margin = runs - (spec.min_run_length if spec.min_run_length is not None else 0)
    # ignore flip/vol for simplicity here
    return margin


def policy_occupancy(state_idx, act_flag, n_states):
    counts = np.bincount(state_idx, minlength=n_states)
    act_counts = np.bincount(state_idx, weights=act_flag.astype(int), minlength=n_states)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi = np.where(counts > 0, act_counts / counts, 0.0)
    rho = counts / counts.sum() if counts.sum() > 0 else counts
    return pi, rho, counts


def js_bernoulli(p, q):
    m = 0.5 * (p + q)
    def kl(x, y):
        x = np.clip(x, 1e-12, 1 - 1e-12)
        y = np.clip(y, 1e-12, 1 - 1e-12)
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
    return 0.5 * (kl(p, m) + kl(q, m))


def transition_kernel(state_idx, n_states):
    # transitions s_t -> s_{t+1}
    s0 = state_idx[:-1]
    s1 = state_idx[1:]
    K = np.zeros((n_states, n_states), dtype=float)
    for i, j in zip(s0, s1):
        K[i, j] += 1
    row_sums = K.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        K = np.where(row_sums > 0, K / row_sums, 0.0)
    occ = np.bincount(s0, minlength=n_states)
    rho = occ / occ.sum() if occ.sum() > 0 else occ
    return K, rho


def compute_boundary_weight(df: pd.DataFrame, bins: int, spec: RegimeSpec):
    if not {"state", "price"}.issubset(df.columns):
        return None
    margin = legitimacy_margin(df, spec)
    a = pd.to_numeric(df["actionability"], errors="coerce").to_numpy(dtype=float)
    acc = df["acceptable"].astype(bool).to_numpy() if "acceptable" in df else np.ones(len(a), dtype=bool)
    mask = np.isfinite(a) & np.isfinite(margin)
    a = a[mask]
    acc = acc[mask]
    margin = margin[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    a_bin = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
    state_idx = acc.astype(int) * bins + a_bin
    weights = np.clip(np.exp(-np.abs(margin)), 0.0, 1.0)
    w = np.zeros(2 * bins)
    for idx, wgt in zip(state_idx, weights):
        w[idx] += wgt
    if w.sum() > 0:
        w = w / w.sum()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=str, required=True, help="First log CSV")
    ap.add_argument("--b", type=str, required=True, help="Second log CSV")
    ap.add_argument("--bins", type=int, default=20, help="Actionability bins")
    ap.add_argument("--min_run_length", type=int, default=3, help="RegimeSpec min_run_length for boundary weight")
    args = ap.parse_args()

    sa, act_a, n_states = load_states(args.a, args.bins)
    sb, act_b, n_states_b = load_states(args.b, args.bins)
    if n_states != n_states_b:
        raise SystemExit("State space mismatch.")

    pi_a, rho_a, _ = policy_occupancy(sa, act_a, n_states)
    pi_b, rho_b, _ = policy_occupancy(sb, act_b, n_states)
    w = 0.5 * (rho_a + rho_b)
    js = np.sum(w * js_bernoulli(pi_a, pi_b))
    l2 = np.sqrt(np.sum(w * (pi_a - pi_b) ** 2))
    l1 = np.sum(w * np.abs(pi_a - pi_b))

    # boundary-weighted L1 using margin if available
    spec = RegimeSpec(min_run_length=args.min_run_length)
    w_a = compute_boundary_weight(pd.read_csv(args.a), args.bins, spec)
    w_b = compute_boundary_weight(pd.read_csv(args.b), args.bins, spec)
    w_boundary = None
    if w_a is not None and w_b is not None:
        w_boundary = 0.5 * (w_a + w_b)
        l1_boundary = np.sum(w_boundary * np.abs(pi_a - pi_b))
    else:
        l1_boundary = np.nan

    Ka, rhoKa = transition_kernel(sa, n_states)
    Kb, rhoKb = transition_kernel(sb, n_states)
    wK = 0.5 * (rhoKa[:, None] + rhoKb[:, None])  # weight per row
    frob = np.sqrt(np.sum(wK * (Ka - Kb) ** 2))

    print(f"Policy JS distance (occupancy-weighted): {js:.6f}")
    print(f"Policy L1 distance (occupancy-weighted): {l1:.6f}")
    print(f"Policy L2 distance (occupancy-weighted): {l2:.6f}")
    print(f"Policy L1 distance (boundary-weighted): {l1_boundary:.6f}")
    print(f"Transition kernel Frobenius (occupancy-weighted): {frob:.6f}")

    out = pd.DataFrame(
        [
            {"metric": "policy_js", "value": js},
            {"metric": "policy_l1", "value": l1},
            {"metric": "policy_l2", "value": l2},
            {"metric": "policy_l1_boundary", "value": l1_boundary},
            {"metric": "kernel_frobenius", "value": frob},
        ]
    )
    out_path = "policy_distance.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

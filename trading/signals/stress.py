from __future__ import annotations

import pandas as pd


def compute_structural_stress(prices, states, window=100, vol_z_thr=1.5, flip_thr=0.4):
    """
    Derive a crude structural stress score from price/triadic state history:
    - rolling vol z-score (median/MAD)
    - rolling jump z-score (abs return / rolling vol)
    - rolling flip rate of the triadic state

    Returns p_bad in [0,1] and a bad_flag bool (p_bad > 0.7).
    """
    prices = pd.Series(prices, dtype=float)
    states = pd.Series(states, dtype=float)
    rets = prices.pct_change().fillna(0.0)
    vol = rets.rolling(window).std().bfill().fillna(0.0)
    med_vol = vol.median()
    mad_vol = (vol - med_vol).abs().median() + 1e-9
    vol_z = (vol - med_vol) / mad_vol
    jump_z = (rets.abs() / (vol + 1e-9)).fillna(0.0)
    flips = (states != states.shift(1)).astype(float)
    flip_rate = flips.rolling(window).mean().fillna(0.0)
    score = (
        (vol_z / vol_z_thr).clip(lower=0)
        + (jump_z).clip(lower=0)
        + (flip_rate / flip_thr).clip(lower=0)
    )
    # squash to [0,1] with a smooth cap
    p_bad = (score / (1.0 + score)).clip(0.0, 1.0)
    bad_flag = p_bad > 0.7
    return p_bad.to_numpy(), bad_flag.to_numpy()

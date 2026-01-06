import numpy as np


def mad(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    median = np.median(x)
    return float(np.median(np.abs(x - median)))


def _window_features(log_returns: list[float], window: int, eps: float) -> tuple[float, float, float]:
    if len(log_returns) < window:
        return (np.nan, np.nan, np.nan)
    window_slice = np.asarray(log_returns[-window:], dtype=float)
    sigma = mad(window_slice)
    denom = sigma if sigma > eps else eps
    normalized = window_slice / denom
    energy = float(np.mean(normalized ** 2))
    abs_sum = float(np.sum(np.abs(normalized)))
    if abs_sum <= eps:
        cancel = 0.0
    else:
        cancel = float(np.abs(np.sum(normalized)) / abs_sum)
    spectrum = np.fft.rfft(normalized)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    if total_power <= eps:
        spectral = 0.0
    else:
        spectral = float(np.max(power) / total_power)
    return energy, cancel, spectral


def compute_quotient_features(
    log_returns: list[float],
    w1: int = 64,
    w2: int = 256,
    eps: float = 1e-9,
) -> dict[str, float]:
    e1, c1, s1 = _window_features(log_returns, w1, eps)
    e2, c2, s2 = _window_features(log_returns, w2, eps)

    if np.isnan(e1) or np.isnan(e2):
        delta_e = np.nan
    else:
        delta_e = abs(e1 - e2)
    if np.isnan(c1) or np.isnan(c2):
        delta_c = np.nan
    else:
        delta_c = abs(c1 - c2)
    if np.isnan(s1) or np.isnan(s2):
        delta_s = np.nan
    else:
        delta_s = abs(s1 - s2)

    return {
        "e64": e1,
        "c64": c1,
        "s64": s1,
        "delta_e": delta_e,
        "delta_c": delta_c,
        "delta_s": delta_s,
    }

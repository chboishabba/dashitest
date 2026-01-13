import numpy as np
import math


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


def compute_qfeat(
    prices: np.ndarray,
    *,
    w1: int = 64,
    w2: int = 256,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Deterministic float32 quotient feature vector (len=6).

    Features (fixed order):
      0 vol_ratio     : std(r) / (max(r) - min(r) + eps)
      1 curvature     : log1p(std(second_diff(r)))
      2 drawdown      : max peak-to-trough of cum log-price, normalized by total move
      3 burstiness    : L1 / (L2 + eps)
      4 acorr_1       : lag-1 autocorr of returns, clipped to [-1,1]
      5 var_ratio     : std(last w1) / (std(last w2) + eps)

    All arithmetic is float32 with explicit loop order for GPU parity.
    """
    p = np.asarray(prices, dtype=np.float32)
    Wp = int(p.size)
    if Wp < 3:
        return np.zeros(6, dtype=np.float32)

    eps32 = np.float32(eps)

    # returns r[i] = log(p[i+1]) - log(p[i])
    W = Wp - 1
    r = np.empty((W,), dtype=np.float32)
    for i in range(W):
        a = p[i]
        b = p[i + 1]
        if a <= eps32:
            a = eps32
        if b <= eps32:
            b = eps32
        r[i] = np.float32(math.log(float(b)) - math.log(float(a)))

    if W < 2:
        return np.zeros(6, dtype=np.float32)

    # mean, min, max
    sum_r = np.float32(0.0)
    rmin = np.float32(r[0])
    rmax = np.float32(r[0])
    for i in range(W):
        v = r[i]
        sum_r = np.float32(sum_r + v)
        if v < rmin:
            rmin = v
        if v > rmax:
            rmax = v
    mean_r = np.float32(sum_r / np.float32(W))

    # var
    ss = np.float32(0.0)
    for i in range(W):
        d = np.float32(r[i] - mean_r)
        ss = np.float32(ss + d * d)
    var = np.float32(ss / np.float32(W))
    sigma = np.float32(math.sqrt(float(var + eps32)))

    # 0) vol_ratio
    r_range = np.float32((rmax - rmin) + eps32)
    vol_ratio = np.float32(sigma / r_range)

    # 1) curvature
    curvature = np.float32(0.0)
    if W >= 3:
        n = W - 2
        d2_sum = np.float32(0.0)
        for i in range(1, W - 1):
            d2 = np.float32(r[i + 1] - np.float32(2.0) * r[i] + r[i - 1])
            d2_sum = np.float32(d2_sum + d2)
        d2_mean = np.float32(d2_sum / np.float32(n))

        d2_ss = np.float32(0.0)
        for i in range(1, W - 1):
            d2 = np.float32(r[i + 1] - np.float32(2.0) * r[i] + r[i - 1])
            d = np.float32(d2 - d2_mean)
            d2_ss = np.float32(d2_ss + d * d)

        d2_var = np.float32(d2_ss / np.float32(n))
        d2_std = np.float32(math.sqrt(float(d2_var + eps32)))
        curvature = np.float32(math.log1p(float(d2_std)))

    # 2) drawdown
    s0 = np.float32(r[0])
    s = np.float32(s0)
    peak = np.float32(s0)
    dd = np.float32(0.0)
    for i in range(1, W):
        s = np.float32(s + r[i])
        if s > peak:
            peak = s
        gap = np.float32(peak - s)
        if gap > dd:
            dd = gap
    send = s
    norm = np.float32(abs(float(send - s0)) + float(eps32))
    drawdown = np.float32(dd / norm)

    # 3) burstiness
    l1 = np.float32(0.0)
    l2s = np.float32(0.0)
    for i in range(W):
        v = r[i]
        l1 = np.float32(l1 + (-v if v < 0 else v))
        l2s = np.float32(l2s + v * v)
    l2 = np.float32(math.sqrt(float(l2s)) + float(eps32))
    burstiness = np.float32(l1 / l2)

    # 4) acorr_1
    acorr_1 = np.float32(0.0)
    if W >= 2:
        s00 = np.float32(0.0)
        s11 = np.float32(0.0)
        s01 = np.float32(0.0)
        for i in range(W - 1):
            a = np.float32(r[i] - mean_r)
            b = np.float32(r[i + 1] - mean_r)
            s00 = np.float32(s00 + a * a)
            s11 = np.float32(s11 + b * b)
            s01 = np.float32(s01 + a * b)
        denom = np.float32(math.sqrt(float(s00 * s11)))
        if denom > 0:
            ac = np.float32(s01 / denom)
            if ac > 1:
                ac = np.float32(1.0)
            elif ac < -1:
                ac = np.float32(-1.0)
            acorr_1 = ac

    # 5) var_ratio
    nf = w1 if W >= w1 else W
    ns = w2 if W >= w2 else W

    sum_f = np.float32(0.0)
    for i in range(W - nf, W):
        sum_f = np.float32(sum_f + r[i])
    mean_f = np.float32(sum_f / np.float32(nf))
    ss_f = np.float32(0.0)
    for i in range(W - nf, W):
        d = np.float32(r[i] - mean_f)
        ss_f = np.float32(ss_f + d * d)
    var_f = np.float32(ss_f / np.float32(nf))
    std_f = np.float32(math.sqrt(float(var_f + eps32)))

    sum_s = np.float32(0.0)
    for i in range(W - ns, W):
        sum_s = np.float32(sum_s + r[i])
    mean_s = np.float32(sum_s / np.float32(ns))
    ss_s = np.float32(0.0)
    for i in range(W - ns, W):
        d = np.float32(r[i] - mean_s)
        ss_s = np.float32(ss_s + d * d)
    var_s = np.float32(ss_s / np.float32(ns))
    std_s = np.float32(math.sqrt(float(var_s + eps32)))

    var_ratio = np.float32(std_f / (std_s + eps32))

    q = np.array(
        [vol_ratio, curvature, drawdown, burstiness, acorr_1, var_ratio],
        dtype=np.float32,
    )
    for i in range(6):
        v = float(q[i])
        if math.isnan(v) or math.isinf(v):
            q[i] = np.float32(0.0)
    return q

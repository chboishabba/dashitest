from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import math
import numpy as np
import pandas as pd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black_scholes(
    price: float,
    strike: float,
    sigma: float,
    tau: float,
    opt_type: str,
) -> float:
    if price <= 0.0 or strike <= 0.0 or tau <= 0.0 or sigma <= 0.0:
        intrinsic = max(price - strike, 0.0) if opt_type == "call" else max(strike - price, 0.0)
        return intrinsic
    vol_sqrt = sigma * math.sqrt(tau)
    if vol_sqrt <= 0:
        intrinsic = max(price - strike, 0.0) if opt_type == "call" else max(strike - price, 0.0)
        return intrinsic
    log_m = math.log(price / strike) if price > 0 and strike > 0 else 0.0
    d1 = (log_m + 0.5 * sigma * sigma * tau) / vol_sqrt
    d2 = d1 - vol_sqrt
    phi_d1 = _norm_cdf(d1)
    phi_d2 = _norm_cdf(d2)
    if opt_type == "call":
        return price * phi_d1 - strike * phi_d2
    return strike * (1.0 - phi_d2) - price * (1.0 - phi_d1)


@dataclass(frozen=True)
class OptionCandidate:
    t_idx: int
    opt_type: str
    expiry_days: int
    strike: float
    tenor_bin: str
    mny_bin: str
    mid: Optional[float]
    iv: Optional[float]
    source: str
    spot: float
    filled: bool


@dataclass(frozen=True)
class SyntheticOptionsConfig:
    sigma0: float = 0.60
    sigma_min: float = 0.20
    sigma_max: float = 1.50
    k_v: float = 0.50
    k_b: float = 0.10
    tenor_bins: tuple[str, ...] = ("e_1_3", "e_4_7", "e_8_21", "e_22_60", "e_61_180")
    tenor_days: tuple[int, ...] = (2, 5, 14, 41, 120)
    mny_bins: tuple[str, ...] = ("m_deep_itm", "m_itm", "m_atm", "m_otm", "m_deep_otm")
    mny_deltas: tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20)


class SyntheticOptionChain:
    def __init__(self, config: SyntheticOptionsConfig) -> None:
        if len(config.tenor_bins) != len(config.tenor_days):
            raise ValueError("tenor_bins and tenor_days length mismatch")
        if len(config.mny_bins) != len(config.mny_deltas):
            raise ValueError("mny_bins and mny_deltas length mismatch")
        self.config = config

    def generate(
        self,
        bar_idx: int,
        spot: float,
        qrow: Iterable[float],
        ts: int,
        *,
        source: str = "synthetic",
        filled: bool = False,
    ) -> List[OptionCandidate]:
        vol_ratio = float(qrow[0])
        burstiness = max(float(qrow[3]), 0.0)
        sigma = (
            self.config.sigma0
            + self.config.k_v * max(vol_ratio, 0.0)
            + self.config.k_b * math.log1p(burstiness)
        )
        sigma = max(self.config.sigma_min, min(sigma, self.config.sigma_max))
        safe_spot = max(spot, 1e-6)
        candidates: List[OptionCandidate] = []
        for tenor_bin, days in zip(self.config.tenor_bins, self.config.tenor_days):
            tau = max(days / 365.0, 1e-6)
            for mny_bin, delta in zip(self.config.mny_bins, self.config.mny_deltas):
                strike = max(safe_spot * math.exp(delta), 1e-6)
                for opt_type in ("call", "put"):
                    mid = _black_scholes(spot, strike, sigma, tau, opt_type)
                    candidates.append(
                    OptionCandidate(
                        t_idx=bar_idx,
                        opt_type=opt_type,
                        expiry_days=days,
                        strike=strike,
                        tenor_bin=tenor_bin,
                        mny_bin=mny_bin,
                        mid=mid,
                        iv=sigma,
                        source=source,
                        spot=spot,
                        filled=filled,
                    )
                    )
        return candidates


def _check_ts_unit(value: int) -> str:
    if value <= 0:
        return "invalid"
    if value < 1e12:
        return "seconds"
    if value > 1e15:
        return "nanoseconds"
    return "milliseconds"


class RealOptionChain:
    def __init__(
        self,
        path: Path,
        *,
        stale_ms: float = 1800_000.0,
        tenor_bins: Sequence[str],
        tenor_days: Sequence[int],
        mny_bins: Sequence[str],
        mny_cutoffs: Sequence[float],
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Options database missing: {path}")
        df = pd.read_parquet(path)
        if "ts" not in df.columns:
            raise ValueError("Options dataset must include ts column")
        if df.empty:
            raise ValueError("Options dataset is empty")
        ts_sample = df["ts"].astype(np.int64)
        unit = _check_ts_unit(int(ts_sample.iloc[0]))
        if unit != "milliseconds":
            raise ValueError(f"Options surface ts must be milliseconds ({unit} found)")
        df["ts"] = ts_sample
        df = df.sort_values("ts")
        self._groups: dict[int, pd.DataFrame] = {}
        for ts, group in df.groupby("ts"):
            self._groups[int(ts)] = group
        self._ts_keys = np.array(sorted(self._groups.keys()), dtype=np.int64)
        self._stale_ms = int(max(stale_ms, 0.0))
        self._tenor_bins = tuple(tenor_bins)
        self._tenor_days = tuple(tenor_days)
        self._mny_bins = tuple(mny_bins)
        self._mny_cutoffs = tuple(mny_cutoffs)

    def _canonical_bins(self) -> Sequence[tuple[str, str, str]]:
        combos = []
        for tenor in self._tenor_bins:
            for mny in self._mny_bins:
                for opt in ("call", "put"):
                    combos.append((tenor, mny, opt))
        return combos

    def generate(
        self,
        bar_idx: int,
        spot: float,
        qrow: Iterable[float],
        ts: int,
    ) -> List[OptionCandidate]:
        if self._ts_keys.size == 0:
            return []
        idx = np.searchsorted(self._ts_keys, ts, side="right") - 1
        if idx < 0:
            return []
        ts_key = int(self._ts_keys[idx])
        if ts - ts_key > self._stale_ms:
            return []
        group = self._groups.get(ts_key)
        if group is None:
            return []
        candidates: List[OptionCandidate] = []
        for _, row in group.iterrows():
            try:
                opt_type = str(row.get("opt_type", "")).lower()
                if opt_type not in {"call", "put"}:
                    opt_type = "call"
                tenor_bin = str(row.get("tenor_bin", self._tenor_bins[0]))
                mny_bin = str(row.get("mny_bin", self._mny_bins[2]))
                expiry_days = int(row.get("expiry_days", 0))
                strike = float(row.get("strike", spot))
                mid = row.get("mid")
                iv = row.get("iv")
            except Exception:
                continue
            candidates.append(
                OptionCandidate(
                    t_idx=bar_idx,
                    opt_type=opt_type,
                    expiry_days=expiry_days,
                    strike=strike,
                    tenor_bin=tenor_bin,
                    mny_bin=mny_bin,
                    mid=float(mid) if pd.notna(mid) else None,
                    iv=float(iv) if pd.notna(iv) else None,
                    source="real",
                    spot=spot,
                    filled=False,
                )
            )
        return candidates


class OptionChain:
    def __init__(
        self,
        *,
        synthetic: Optional[SyntheticOptionChain] = None,
        real: Optional[RealOptionChain] = None,
        fallback_to_synthetic: bool = False,
    ) -> None:
        self.synthetic = synthetic
        self.real = real
        self.fallback_to_synthetic = fallback_to_synthetic

    def generate(
        self,
        bar_idx: int,
        spot: float,
        qrow: Iterable[float],
        ts: int,
    ) -> List[OptionCandidate]:
        if self.real is not None:
            candidates = self.real.generate(bar_idx, spot, qrow, ts)
            if candidates:
                return candidates
            if self.fallback_to_synthetic and self.synthetic is not None:
                return self.synthetic.generate(
                    bar_idx,
                    spot,
                    qrow,
                    ts,
                    source="fallback",
                    filled=True,
                )
            return []
        if self.synthetic is not None:
            return self.synthetic.generate(bar_idx, spot, qrow, ts)
        return []

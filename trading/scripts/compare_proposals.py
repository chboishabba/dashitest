"""
compare_proposals.py
--------------------
Compare two proposal logs for ACT/HOLD equivalence and safety invariants.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def _act_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.startswith("ACT")


def _rate(mask: pd.Series) -> float:
    if mask.size == 0:
        return float("nan")
    return float(mask.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare proposal logs for Phase-2 checks.")
    ap.add_argument("--base", type=Path, required=True, help="Baseline proposal CSV.")
    ap.add_argument("--ont", type=Path, required=True, help="Ontology proposal CSV.")
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    ont = pd.read_csv(args.ont)

    if len(base) != len(ont):
        raise SystemExit(f"Row count mismatch: base={len(base)} ont={len(ont)}")

    base_act = _act_mask(base["would_act"])
    ont_act = _act_mask(ont["would_act"])
    identical = bool((base_act == ont_act).all())

    base_rate = _rate(base_act)
    ont_rate = _rate(ont_act)
    rel_change = float("nan")
    if math.isfinite(base_rate) and base_rate > 0:
        rel_change = (ont_rate - base_rate) / base_rate

    base_veto_act = int(((base["veto"] == 1) & base_act).sum())
    ont_veto_act = int(((ont["veto"] == 1) & ont_act).sum())
    base_hazard_act = int(((base["veto_reason"] == "hazard") & base_act).sum())
    ont_hazard_act = int(((ont["veto_reason"] == "hazard") & ont_act).sum())

    print("ACT/HOLD equivalence:", "PASS" if identical else "FAIL")
    print(f"ACT rate base={base_rate:.6f} ont={ont_rate:.6f} rel_change={rel_change:.6%}")
    print(f"vetoed && ACT base={base_veto_act} ont={ont_veto_act}")
    print(f"hazard_veto && ACT base={base_hazard_act} ont={ont_hazard_act}")


if __name__ == "__main__":
    main()

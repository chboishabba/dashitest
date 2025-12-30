"""
sweep_confusion_surface.py
--------------------------
Sweep tau_off and compute false-positive / false-negative densities over actionability bins.
False-positive: ACT & not acceptable
False-negative: HOLD & acceptable

Outputs a CSV with columns: tau_off, bin_center, fp_rate, fn_rate

Usage:
  PYTHONPATH=. python trading/scripts/sweep_confusion_surface.py --out logs/confusion_surface.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from runner import run_bars
from trading.run_trader import load_prices, compute_triadic_state
from trading.scripts.run_bars_btc import confidence_from_persistence


def bin_rates(df: pd.DataFrame, bins: int = 20):
    if df is None or df.empty:
        return []
    if not {"acceptable", "actionability", "action"}.issubset(df.columns):
        return []
    acceptable = df["acceptable"].astype(bool)
    act = df["action"] != 0
    actionability = pd.to_numeric(df["actionability"], errors="coerce")
    mask = np.isfinite(actionability)
    acceptable = acceptable[mask]
    act = act[mask]
    actionability = actionability[mask]
    if actionability.empty:
        return []
    fp = act & (~acceptable)
    fn = (~act) & acceptable
    edges = np.linspace(0.0, 1.0, max(5, bins) + 1)
    labels = 0.5 * (edges[:-1] + edges[1:])
    cat = pd.cut(actionability, edges, include_lowest=True, labels=labels)
    counts = cat.value_counts().sort_index()
    fp_counts = fp.groupby(cat).sum().reindex(counts.index, fill_value=0)
    fn_counts = fn.groupby(cat).sum().reindex(counts.index, fill_value=0)
    rates = []
    for center, cnt in counts.items():
        if cnt <= 0:
            continue
        center_val = float(center)
        rates.append(
            {
                "bin_center": center_val,
                "fp_rate": float(fp_counts.loc[center] / cnt),
                "fn_rate": float(fn_counts.loc[center] / cnt),
            }
        )
    return rates


def run_once(csv: Path, tau_on: float, tau_off: float):
    price, _ = load_prices(csv)
    ts = np.arange(len(price))
    state = compute_triadic_state(price)
    bars = pd.DataFrame({"ts": ts, "close": price, "state": state})
    rets = np.diff(price, prepend=price[0])
    vol = pd.Series(rets).rolling(50).std().to_numpy()
    conf_seq = confidence_from_persistence(
        state, run_scale=30, ret_vol=vol, vol_thresh=np.nanpercentile(vol, 80)
    )

    def conf_fn(t, s):
        idx = int(t)
        if idx < 0 or idx >= len(conf_seq):
            return 1.0
        return conf_seq[idx]

    df = run_bars(
        bars,
        symbol="BTCUSDT",
        mode="bar",
        log_path=None,
        confidence_fn=conf_fn,
        tau_conf_enter=tau_on,
        tau_conf_exit=tau_off,
    )
    return bin_rates(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/stooq/btc_intraday.csv"),
        help="Price CSV",
    )
    ap.add_argument("--tau_on", type=float, default=0.5, help="Entry threshold")
    ap.add_argument(
        "--tau_off",
        type=float,
        nargs="+",
        default=[0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
        help="Exit thresholds to sweep",
    )
    ap.add_argument("--bins", type=int, default=20, help="Actionability bins")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = ap.parse_args()

    rows = []
    for tau_off in args.tau_off:
        rates = run_once(args.csv, args.tau_on, tau_off)
        for r in rates:
            r["tau_off"] = tau_off
            rows.append(r)
        print(f"tau_off={tau_off:.2f} bins={len(rates)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote confusion surface to {args.out}")


if __name__ == "__main__":
    main()

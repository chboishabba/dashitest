"""
plot_sheet_visualizer.py
------------------------
Diagnostic "sheet" visualizer that consumes qfeat tapes and proposal logs.

The sheet mapping is fixed: pick any two qfeat coordinates as axes and
overlay proposal/veto metadata. This script is observer-only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required for plotting") from exc

try:
    from trading.vk_qfeat import QFeatTape
except ModuleNotFoundError:
    from vk_qfeat import QFeatTape


QFEAT_COLUMNS = {
    "vol_ratio": 0,
    "curvature": 1,
    "drawdown": 2,
    "burstiness": 3,
    "acorr_1": 4,
    "var_ratio": 5,
    "ell": 6,
}


def _parse_axis(value: str) -> tuple[str, int]:
    if value.isdigit():
        idx = int(value)
        if idx < 0 or idx > 7:
            raise SystemExit(f"axis index out of range: {value}")
        return f"qfeat_{idx}", idx
    key = value.strip().lower()
    if key not in QFEAT_COLUMNS:
        raise SystemExit(f"unknown axis '{value}', expected one of {sorted(QFEAT_COLUMNS)}")
    return key, QFEAT_COLUMNS[key]


def _downsample(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_points, replace=False)
    return df.iloc[idx]


def _plot_continuous(ax, x, y, c, title: str) -> None:
    sc = ax.scatter(x, y, c=c, s=6, alpha=0.6, cmap="viridis", linewidths=0)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def _plot_categorical(ax, x, y, labels: Iterable[str], title: str) -> None:
    labels = np.asarray(labels, dtype=object)
    uniq = pd.unique(labels)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(uniq))))
    for idx, val in enumerate(uniq):
        mask = labels == val
        ax.scatter(x[mask], y[mask], s=6, alpha=0.6, color=colors[idx], label=str(val), linewidths=0)
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8, loc="best", frameon=False)


def _resolve_out(path: Path, name: str) -> Path:
    if "{name}" in str(path):
        return Path(str(path).format(name=name))
    if path.suffix:
        return path.with_name(f"{path.stem}_{name}{path.suffix}")
    return path.with_name(f"{path.name}_{name}.png")


def _safe_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return df[name].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize qfeat sheet coordinates with proposal overlays.")
    ap.add_argument("--tape", type=Path, required=True, help="QFeat tape (.memmap).")
    ap.add_argument("--proposal-log", type=Path, default=None, help="Proposal log CSV (optional).")
    ap.add_argument("--series", type=int, default=0, help="Series index in tape.")
    ap.add_argument("--rows", type=int, default=None, help="Row hint for flat memmap tapes.")
    ap.add_argument("--x-col", type=str, default="curvature", help="Sheet x-axis (name or index).")
    ap.add_argument("--y-col", type=str, default="burstiness", help="Sheet y-axis (name or index).")
    ap.add_argument("--tau-on", type=float, default=0.5, help="Gate threshold for ell overlays.")
    ap.add_argument(
        "--plots",
        type=str,
        default="all,gate,veto,instrument,opt_tenor,opt_mny",
        help="Comma list.",
    )
    ap.add_argument("--max-points", type=int, default=50000, help="Downsample for plotting.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for downsampling.")
    ap.add_argument("--drop-zero", action="store_true", help="Drop rows with near-zero qfeat.")
    ap.add_argument("--out", type=Path, default=Path("logs/plots/sheet_{name}.png"))
    args = ap.parse_args()

    x_name, x_idx = _parse_axis(args.x_col)
    y_name, y_idx = _parse_axis(args.y_col)

    rows_hint = args.rows
    if rows_hint is None and args.proposal_log is not None and args.proposal_log.exists():
        try:
            rows_hint = int(pd.read_csv(args.proposal_log, usecols=["i"]).shape[0])
        except Exception:
            rows_hint = None
    try:
        tape = QFeatTape.from_existing(str(args.tape), rows=rows_hint)
    except ValueError:
        tape = QFeatTape.from_existing(str(args.tape))
    if args.series < 0 or args.series >= tape.num_series:
        raise SystemExit(f"series index {args.series} out of range (S={tape.num_series})")
    qfeat = tape.mm[args.series]
    df = pd.DataFrame(
        {
            "i": np.arange(tape.T, dtype=int),
            "x": qfeat[:, x_idx].astype(float, copy=False),
            "y": qfeat[:, y_idx].astype(float, copy=False),
            "ell": qfeat[:, 6].astype(float, copy=False),
        }
    )

    if args.proposal_log is not None and args.proposal_log.exists():
        log_cols = [
            "i",
            "ell",
            "veto",
            "veto_reason",
            "would_act",
            "instrument_pred",
            "opt_tenor_pred",
            "opt_mny_pred",
            "hazard",
            "score_margin",
            "ontology_k",
            "p_ont_t",
            "p_ont_r",
            "p_ont_h",
        ]
        log_df = pd.read_csv(args.proposal_log, usecols=lambda c: c in log_cols)
        df = df.merge(log_df, on="i", how="left", suffixes=("", "_log"))
        if "ell_log" in df.columns:
            df["ell"] = df["ell_log"].where(np.isfinite(df["ell_log"]), df["ell"])
            df = df.drop(columns=["ell_log"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    if args.drop_zero:
        zero_mask = (df["x"].abs() + df["y"].abs()) < 1e-9
        df = df[~zero_mask]
    df = _downsample(df, args.max_points, args.seed)

    plots = [p.strip() for p in args.plots.split(",") if p.strip()]
    out_base = args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)

    for name in plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        if name == "all":
            _plot_continuous(ax, x, y, df["ell"].to_numpy(), f"all (ell) [{x_name} vs {y_name}]")
        elif name == "gate":
            mask = df["ell"].to_numpy() >= float(args.tau_on)
            if "would_act" in df.columns:
                _plot_categorical(
                    ax,
                    x[mask],
                    y[mask],
                    df.loc[mask, "would_act"].fillna("missing"),
                    f"gate-open (would_act) [{x_name} vs {y_name}]",
                )
            else:
                _plot_continuous(ax, x[mask], y[mask], df.loc[mask, "ell"], f"gate-open (ell)")
        elif name == "veto":
            if "veto" not in df.columns:
                plt.close(fig)
                continue
            mask = df["veto"].fillna(0).to_numpy().astype(int) == 1
            if "veto_reason" in df.columns:
                _plot_categorical(
                    ax,
                    x[mask],
                    y[mask],
                    df.loc[mask, "veto_reason"].fillna("missing"),
                    f"vetoed (reason) [{x_name} vs {y_name}]",
                )
            else:
                _plot_continuous(ax, x[mask], y[mask], df.loc[mask, "ell"], "vetoed (ell)")
        elif name == "instrument":
            if "instrument_pred" not in df.columns:
                plt.close(fig)
                continue
            mask = df["instrument_pred"].notna()
            _plot_categorical(
                ax,
                x[mask],
                y[mask],
                df.loc[mask, "instrument_pred"].fillna("missing"),
                f"instrument_pred [{x_name} vs {y_name}]",
            )
        elif name == "opt_tenor":
            if "opt_tenor_pred" not in df.columns:
                plt.close(fig)
                continue
            mask = df["opt_tenor_pred"].notna() & (df["opt_tenor_pred"] != "none")
            _plot_categorical(
                ax,
                x[mask],
                y[mask],
                df.loc[mask, "opt_tenor_pred"].fillna("missing"),
                f"opt_tenor_pred [{x_name} vs {y_name}]",
            )
        elif name == "opt_mny":
            if "opt_mny_pred" not in df.columns:
                plt.close(fig)
                continue
            mask = df["opt_mny_pred"].notna() & (df["opt_mny_pred"] != "none")
            _plot_categorical(
                ax,
                x[mask],
                y[mask],
                df.loc[mask, "opt_mny_pred"].fillna("missing"),
                f"opt_mny_pred [{x_name} vs {y_name}]",
            )
        elif name == "ontology":
            if "ontology_k" in df.columns:
                mask = df["ontology_k"].notna()
                _plot_categorical(
                    ax,
                    x[mask],
                    y[mask],
                    df.loc[mask, "ontology_k"].fillna("missing"),
                    f"ontology_k [{x_name} vs {y_name}]",
                )
            elif {"p_ont_t", "p_ont_r", "p_ont_h"}.issubset(df.columns):
                probs = df[["p_ont_t", "p_ont_r", "p_ont_h"]].to_numpy()
                labels = np.array(["T", "R", "H"], dtype=object)[np.nanargmax(probs, axis=1)]
                _plot_categorical(
                    ax,
                    x,
                    y,
                    labels,
                    f"ontology_k [{x_name} vs {y_name}]",
                )
            else:
                plt.close(fig)
                continue
        elif name in {"p_ont_t", "p_ont_r", "p_ont_h"}:
            if name not in df.columns:
                plt.close(fig)
                continue
            _plot_continuous(
                ax,
                x,
                y,
                df[name].to_numpy(),
                f"{name} [{x_name} vs {y_name}]",
            )
        else:
            plt.close(fig)
            continue

        out_path = _resolve_out(out_base, name)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

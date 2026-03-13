from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from collections import Counter
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _set_csv_field_limit_to_max() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _corr(xs, ys) -> float:
    pts = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 3:
        return float("nan")
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pts)
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= 0 or deny <= 0:
        return float("nan")
    return num / (denx * deny)


def load_rows(path: pathlib.Path) -> list[dict[str, str]]:
    _set_csv_field_limit_to_max()
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _safe_hist(ax, data, bins: int = 30) -> None:
    values = [x for x in data if math.isfinite(x)]
    if not values:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) <= 1e-12:
        ax.hist(values, bins=1)
        return
    ax.hist(values, bins=bins)


def analyze(rows: list[dict[str, str]], horizon: int = 20) -> dict[str, object]:
    entropy = [_float(row, "beam_entropy") for row in rows if math.isfinite(_float(row, "beam_entropy"))]
    flat = [_float(row, "beam_flat_mass") for row in rows if math.isfinite(_float(row, "beam_flat_mass"))]
    score_series = []
    basin_margin_series = []
    curvature_series = []
    long_series = []
    short_series = []
    divergence = []
    flat_nonzero = 0
    hold_count = 0
    hold_reasons = Counter()
    ent_future = []
    future_abs = []
    contraction = []
    trend_strength = []
    basin_edge = []
    next_move = []
    action_future = []
    hold_future = []
    action_signed = []
    live_signed = []
    action_correct = 0
    action_eval = 0
    live_correct = 0
    live_eval = 0
    for idx, row in enumerate(rows):
        live = row.get("live_direction", "")
        shadow = row.get("shadow_direction", "")
        if live != "" and shadow != "":
            divergence.append(1.0 if live != shadow else 0.0)
        if row.get("shadow_hold", "") == "1":
            hold_count += 1
        hold_reasons[row.get("shadow_hold_reason_primary", "")] += 1
        fm = _float(row, "beam_flat_mass")
        if math.isfinite(fm) and abs(fm) > 1e-12:
            flat_nonzero += 1
        score_value = _float(row, "shadow_score_adjusted")
        if not math.isfinite(score_value):
            score_value = _float(row, "shadow_score_value")
        if math.isfinite(score_value):
            score_series.append(score_value)
        margin = abs(_float(row, "beam_long_mass") - _float(row, "beam_short_mass"))
        if math.isfinite(margin):
            basin_margin_series.append(margin)
        curvature = _float(row, "beam_curvature")
        if math.isfinite(curvature):
            curvature_series.append(curvature)
        long_mass = _float(row, "beam_long_mass")
        short_mass = _float(row, "beam_short_mass")
        if math.isfinite(long_mass):
            long_series.append(long_mass)
        if math.isfinite(short_mass):
            short_series.append(short_mass)
        if idx + horizon < len(rows):
            px = _float(row, "price")
            px2 = _float(rows[idx + horizon], "price")
            if math.isfinite(px) and math.isfinite(px2) and px != 0:
                future_ret = px2 / px - 1.0
                future_abs.append(abs(future_ret))
                ent_future.append(_float(row, "beam_entropy"))
                contraction.append(_float(row, "beam_contraction"))
                trend_strength.append(abs(px2 / px - 1.0))
                basin_edge.append(_float(row, "beam_long_mass") - _float(row, "beam_short_mass"))
                next_move.append(future_ret)
                shadow_hold = row.get("shadow_hold", "") == "1"
                shadow_dir = _float(row, "shadow_direction")
                if not shadow_hold and math.isfinite(shadow_dir):
                    action_future.append(future_ret)
                    action_signed.append(shadow_dir * future_ret)
                    if shadow_dir != 0 and future_ret != 0:
                        action_eval += 1
                        if (shadow_dir > 0) == (future_ret > 0):
                            action_correct += 1
                else:
                    hold_future.append(future_ret)
                live_hold = row.get("live_hold", "") == "1"
                live_dir = _float(row, "live_direction")
                if not live_hold and math.isfinite(live_dir):
                    live_signed.append(live_dir * future_ret)
                    if live_dir != 0 and future_ret != 0:
                        live_eval += 1
                        if (live_dir > 0) == (future_ret > 0):
                            live_correct += 1
    return {
        "rows": len(rows),
        "entropy_mean": mean(entropy) if entropy else float("nan"),
        "entropy_min": min(entropy) if entropy else float("nan"),
        "entropy_max": max(entropy) if entropy else float("nan"),
        "flat_mean": mean(flat) if flat else float("nan"),
        "flat_nonzero_ratio": (flat_nonzero / len(rows)) if rows else float("nan"),
        "hold_ratio": (hold_count / len(rows)) if rows else float("nan"),
        "divergence": mean(divergence) if divergence else float("nan"),
        "entropy_absret_corr": _corr(ent_future, future_abs),
        "contraction_trend_corr": _corr(contraction, trend_strength),
        "basin_edge_move_corr": _corr(basin_edge, next_move),
        "hold_reasons": hold_reasons,
        "entropy_series": entropy,
        "flat_series": flat,
        "score_series": score_series,
        "basin_margin_series": basin_margin_series,
        "curvature_series": curvature_series,
        "long_series": long_series,
        "short_series": short_series,
        "entropy_future": ent_future,
        "future_abs": future_abs,
        "action_rate": 1.0 - (hold_count / len(rows)) if rows else float("nan"),
        "action_return_mean": mean(action_future) if action_future else float("nan"),
        "hold_return_mean": mean(hold_future) if hold_future else float("nan"),
        "action_abs_return_mean": mean([abs(x) for x in action_future]) if action_future else float("nan"),
        "hold_abs_return_mean": mean([abs(x) for x in hold_future]) if hold_future else float("nan"),
        "action_sign_accuracy": (action_correct / action_eval) if action_eval else float("nan"),
        "live_sign_accuracy": (live_correct / live_eval) if live_eval else float("nan"),
        "shadow_sharpe": (mean(action_signed) / (np.std(action_signed) + 1e-9)) if action_signed else float("nan"),
        "live_sharpe": (mean(live_signed) / (np.std(live_signed) + 1e-9)) if live_signed else float("nan"),
    }


def parse_inputs(items: list[str]) -> list[tuple[str, pathlib.Path]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected LABEL=PATH, got {item}")
        label, path = item.split("=", 1)
        parsed.append((label, pathlib.Path(path)))
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="LABEL=PATH shadow CSV input (repeatable)")
    ap.add_argument("--report", required=True, help="Output markdown report path")
    ap.add_argument("--plot", required=True, help="Output plot path")
    ap.add_argument("--horizon", type=int, default=20, help="Forward horizon for correlation calculations")
    args = ap.parse_args()

    inputs = parse_inputs(args.input)
    results = [(label, analyze(load_rows(path), horizon=args.horizon)) for label, path in inputs]

    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Shadow Signal Report\n\n")
        for label, stats in results:
            fh.write(f"## {label}\n\n")
            fh.write(f"- rows: `{stats['rows']}`\n")
            fh.write(
                f"- entropy mean/min/max: `{stats['entropy_mean']:.6f}` / `{stats['entropy_min']:.6f}` / `{stats['entropy_max']:.6f}`\n"
            )
            fh.write(f"- flat-mass mean: `{stats['flat_mean']:.6f}`\n")
            fh.write(f"- flat-mass nonzero ratio: `{stats['flat_nonzero_ratio']:.6f}`\n")
            fh.write(f"- shadow hold ratio: `{stats['hold_ratio']:.6f}`\n")
            fh.write(f"- shadow action rate: `{stats['action_rate']:.6f}`\n")
            fh.write(f"- live vs shadow direction divergence: `{stats['divergence']:.6f}`\n")
            fh.write(f"- entropy vs future abs return corr ({args.horizon}): `{stats['entropy_absret_corr']:.6f}`\n")
            fh.write(f"- contraction vs trend strength corr ({args.horizon}): `{stats['contraction_trend_corr']:.6f}`\n")
            fh.write(f"- basin edge vs next move corr ({args.horizon}): `{stats['basin_edge_move_corr']:.6f}`\n")
            if stats["curvature_series"]:
                fh.write(f"- curvature mean: `{mean(stats['curvature_series']):.6f}`\n")
            if stats["long_series"] and stats["short_series"] and stats["flat_series"]:
                fh.write(
                    "- predicted class mean (long/short/flat): "
                    f"`{mean(stats['long_series']):.4f}` / `{mean(stats['short_series']):.4f}` / `{mean(stats['flat_series']):.4f}`\n"
                )
            fh.write(f"- shadow action sign accuracy ({args.horizon}): `{stats['action_sign_accuracy']:.6f}`\n")
            fh.write(f"- live action sign accuracy ({args.horizon}): `{stats['live_sign_accuracy']:.6f}`\n")
            fh.write(f"- shadow action return mean ({args.horizon}): `{stats['action_return_mean']:.6f}`\n")
            fh.write(f"- shadow hold return mean ({args.horizon}): `{stats['hold_return_mean']:.6f}`\n")
            fh.write(f"- shadow action abs return mean ({args.horizon}): `{stats['action_abs_return_mean']:.6f}`\n")
            fh.write(f"- shadow hold abs return mean ({args.horizon}): `{stats['hold_abs_return_mean']:.6f}`\n")
            fh.write(f"- shadow action Sharpe proxy ({args.horizon}): `{stats['shadow_sharpe']:.6f}`\n")
            fh.write(f"- live action Sharpe proxy ({args.horizon}): `{stats['live_sharpe']:.6f}`\n")
            fh.write(f"- hold reasons: `{stats['hold_reasons'].most_common()}`\n\n")

    rows_per_label = 3
    fig, axes = plt.subplots(len(results) * rows_per_label, 3, figsize=(16, 4 * max(len(results), 1) * rows_per_label))
    if len(results) == 1:
        axes = [axes] if rows_per_label == 1 else axes
    for idx, (label, stats) in enumerate(results):
        row_base = idx * rows_per_label
        ax_entropy = axes[row_base][0]
        ax_flat = axes[row_base][1]
        ax_score = axes[row_base][2]
        ax_basin = axes[row_base + 1][0]
        ax_action = axes[row_base + 1][1]
        ax_ent_profit = axes[row_base + 1][2]
        ax_curve = axes[row_base + 2][0]
        ax_curve_profit = axes[row_base + 2][1]
        ax_curve_hold = axes[row_base + 2][2]

        _safe_hist(ax_entropy, stats["entropy_series"], bins=30)
        ax_entropy.set_title(f"{label} entropy")
        _safe_hist(ax_flat, stats["flat_series"], bins=30)
        ax_flat.set_title(f"{label} flat mass")
        _safe_hist(ax_score, stats["score_series"], bins=30)
        ax_score.set_title(f"{label} score")

        _safe_hist(ax_basin, stats["basin_margin_series"], bins=30)
        ax_basin.set_title(f"{label} basin margin")

        action_mean = stats["action_return_mean"]
        hold_mean = stats["hold_return_mean"]
        ax_action.bar(["act", "hold"], [action_mean, hold_mean])
        ax_action.set_title(f"{label} future return ({args.horizon})")
        ax_action.axhline(0.0, color="black", linewidth=0.8)

        ent_future = [x for x in stats["entropy_future"] if math.isfinite(x)]
        future_abs = [x for x in stats["future_abs"] if math.isfinite(x)]
        if ent_future and future_abs and len(ent_future) == len(future_abs):
            ax_ent_profit.scatter(ent_future, future_abs, s=6, alpha=0.5)
        ax_ent_profit.set_title(f"{label} entropy vs abs return")

        _safe_hist(ax_curve, stats["curvature_series"], bins=30)
        ax_curve.set_title(f"{label} curvature")

        curvature = stats["curvature_series"]
        if curvature and future_abs and len(curvature) >= len(future_abs):
            curvature = curvature[: len(future_abs)]
        if curvature and future_abs and len(curvature) == len(future_abs):
            ax_curve_profit.scatter(curvature, future_abs, s=6, alpha=0.5)
        ax_curve_profit.set_title(f"{label} curvature vs abs return")

        ax_curve_hold.bar(["act", "hold"], [stats["action_abs_return_mean"], stats["hold_abs_return_mean"]])
        ax_curve_hold.set_title(f"{label} abs return (act/hold)")
    plt.tight_layout()
    plot_path = pathlib.Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)


if __name__ == "__main__":
    main()

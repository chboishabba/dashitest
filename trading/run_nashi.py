from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from nashi.phase9 import CapitalParams
from nashi.runtime import NashiArtifacts, default_bars, run_nashi_bars


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the contract-first Nashi trader.")
    ap.add_argument("--csv", type=Path, default=None, help="Optional input CSV. Defaults to cached BTC/Stooq data.")
    ap.add_argument("--symbol", default="BTCUSDT", help="Symbol label for logs/viz surfaces.")
    ap.add_argument("--log-prefix", default="logs/nashi/trading_log_nashi", help="Output stem for CSV/NDJSON logs.")
    ap.add_argument("--duckdb", type=Path, default=Path("logs/research/nashi.duckdb"), help="DuckDB path for Nashi decisions and OHLC.")
    ap.add_argument("--base-size", type=float, default=1.0, help="Maximum target exposure magnitude.")
    ap.add_argument("--default-spread-bps", type=float, default=2.0, help="Synthetic full spread in basis points when bid/ask quotes are unavailable.")
    ap.add_argument("--progress-every", type=int, default=5000, help="Emit progress every N bars. Use 0 to disable.")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap on input rows for bounded validation runs.")
    ap.add_argument("--contextual-hazard-csv", type=Path, default=None, help="Optional offline contextual hazard window CSV from score_bad_windows.py or emit_news_windows.py.")
    ap.add_argument("--phase9-min-expected-surplus", type=float, default=0.0, help="Minimum expected surplus required by the Phase-9 gate.")
    ap.add_argument("--phase9-min-actionability", type=float, default=0.20, help="Minimum actionability required by the Phase-9 gate.")
    ap.add_argument("--phase9-min-edge", type=float, default=0.01, help="Minimum edge magnitude used for positive-edge framing.")
    ap.add_argument("--phase9-microstructure-floor", type=float, default=1.05, help="Base cost-survival ratio floor for explicit microstructure kill decisions.")
    ap.add_argument("--phase9-microstructure-floor-min", type=float, default=0.60, help="Lower bound for the adaptive microstructure survival floor.")
    ap.add_argument("--phase9-microstructure-relief", type=float, default=0.40, help="How strongly strong edge/actionability relax the microstructure survival floor.")
    ap.add_argument("--phase9-microstructure-min-turnover", type=float, default=1e-3, help="Minimum exposure change before microstructure kill logic activates.")
    ap.add_argument("--phase9-microstructure-min-gross", type=float, default=1e-6, help="Minimum expected gross surplus before microstructure kill logic activates.")
    ap.add_argument("--phase9-hazard-reentry-threshold", type=float, default=0.22, help="Hazard level below which governance hazard fully releases for re-entry.")
    ap.add_argument("--phase9-hazard-reentry-relief", type=float, default=0.35, help="How much sub-clamp hazard is attenuated near re-entry before full tightening resumes.")
    ap.add_argument("--phase9-hazard-clamp-threshold", type=float, default=0.48, help="Hazard level where exposure tightening starts.")
    ap.add_argument("--phase9-hazard-hold-threshold", type=float, default=0.78, help="Hazard level where hold/observe directives start.")
    ap.add_argument("--phase9-hazard-ban-threshold", type=float, default=0.96, help="Hazard level where ban directives start.")
    ap.add_argument("--phase9-hazard-survival-floor-add", type=float, default=0.40, help="Extra microstructure survival floor added under hazard.")
    ap.add_argument("--phase9-hazard-exposure-tightening", type=float, default=0.60, help="How strongly hazard tightens the max exposure clamp.")
    ap.add_argument("--phase9-hazard-min-exposure-scale", type=float, default=0.15, help="Minimum residual exposure scale under hazard tightening.")
    args = ap.parse_args()

    bars, source_label = default_bars(args.csv, default_spread_bps=args.default_spread_bps)
    if args.max_rows > 0:
        bars = bars.head(args.max_rows).copy()
    prefix = Path(args.log_prefix)
    artifacts = NashiArtifacts(
        step_log_path=prefix.with_suffix(".csv"),
        decision_ndjson_path=prefix.with_name(f"{prefix.name}_decisions.ndjson"),
        ohlc_ndjson_path=prefix.with_name(f"{prefix.name}_ohlc.ndjson"),
        duckdb_path=args.duckdb,
        family_csv_path=prefix.with_name(f"{prefix.name}_family.csv"),
        family_ndjson_path=prefix.with_name(f"{prefix.name}_family.ndjson"),
    )
    phase9_params = CapitalParams(
        min_expected_surplus=args.phase9_min_expected_surplus,
        min_actionability=args.phase9_min_actionability,
        min_edge=args.phase9_min_edge,
        microstructure_survival_floor=args.phase9_microstructure_floor,
        microstructure_survival_floor_min=args.phase9_microstructure_floor_min,
        microstructure_relief_strength=args.phase9_microstructure_relief,
        microstructure_min_turnover=args.phase9_microstructure_min_turnover,
        microstructure_min_expected_gross=args.phase9_microstructure_min_gross,
        hazard_reentry_threshold=args.phase9_hazard_reentry_threshold,
        hazard_reentry_relief=args.phase9_hazard_reentry_relief,
        hazard_clamp_threshold=args.phase9_hazard_clamp_threshold,
        hazard_hold_threshold=args.phase9_hazard_hold_threshold,
        hazard_ban_threshold=args.phase9_hazard_ban_threshold,
        hazard_survival_floor_add=args.phase9_hazard_survival_floor_add,
        hazard_exposure_tightening=args.phase9_hazard_exposure_tightening,
        hazard_min_exposure_scale=args.phase9_hazard_min_exposure_scale,
    )
    def progress_fn(done: int, total: int, elapsed_s: float) -> None:
        if args.progress_every <= 0:
            return
        if done != total and done % args.progress_every != 0:
            return
        rate = done / elapsed_s if elapsed_s > 0 else 0.0
        remaining = max(total - done, 0)
        eta_s = remaining / rate if rate > 0 else 0.0
        eta_ts = datetime.now(timezone.utc).timestamp() + eta_s
        eta_text = datetime.fromtimestamp(eta_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        pct = (100.0 * done / total) if total > 0 else 100.0
        print(
            f"[nashi] progress {done}/{total} ({pct:.1f}%) "
            f"speed={rate:.1f} bars/s eta={eta_text}"
        )

    df = run_nashi_bars(
        bars,
        symbol=args.symbol,
        artifacts=artifacts,
        source_label=source_label,
        base_size=args.base_size,
        phase9_params=phase9_params,
        default_spread_bps=args.default_spread_bps,
        contextual_hazard_csv=args.contextual_hazard_csv,
        progress_fn=progress_fn,
    )
    print(f"Wrote {len(df)} rows to {artifacts.step_log_path}")
    print(f"Decision NDJSON: {artifacts.decision_ndjson_path}")
    print(f"OHLC NDJSON: {artifacts.ohlc_ndjson_path}")
    print(f"Family CSV: {artifacts.family_csv_path}")
    print(f"Family NDJSON: {artifacts.family_ndjson_path}")
    print(f"DuckDB: {artifacts.duckdb_path}")


if __name__ == "__main__":
    main()

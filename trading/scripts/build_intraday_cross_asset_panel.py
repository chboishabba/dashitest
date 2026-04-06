#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIMESTAMP_CANDIDATES = ("timestamp", "ts", "datetime", "date", "data")
PRICE_CANDIDATES = ("price", "close", "zamkniecie", "last", "price_exec")
RETURN_CANDIDATES = ("return", "ret", "price_ret")


@dataclass(frozen=True)
class AssetSpec:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        action="append",
        required=True,
        help="Asset input in NAME=path form; repeatable.",
    )
    parser.add_argument(
        "--reference-asset",
        help="Optional asset name whose timestamps define the base clock. Defaults to the first asset.",
    )
    parser.add_argument(
        "--freq",
        default=None,
        help="Optional pandas frequency for resampling before alignment, for example '1s' or '1min'.",
    )
    parser.add_argument(
        "--align",
        choices=("intersection", "reference"),
        default="reference",
        help="Use only shared timestamps or keep the reference asset clock and expose missingness.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=120,
        help="Rows per availability window in the summary.",
    )
    parser.add_argument("--output-csv", help="Optional CSV path for the aligned panel.")
    parser.add_argument("--summary-out", help="Optional JSON path for the panel summary.")
    return parser.parse_args()


def _parse_assets(raw_items: list[str]) -> list[AssetSpec]:
    specs: list[AssetSpec] = []
    for raw in raw_items:
        name, sep, value = raw.partition("=")
        if not sep or not name.strip() or not value.strip():
            raise SystemExit(f"invalid --asset {raw!r}; expected NAME=path")
        path = Path(value.strip())
        if not path.exists():
            raise SystemExit(f"missing asset file: {path}")
        specs.append(AssetSpec(name=name.strip().upper(), path=path))
    return specs


def _find_col(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.notna().any():
        return parsed

    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        return parsed

    abs_numeric = numeric.abs()
    nanos_mask = abs_numeric >= 10**17
    millis_mask = (abs_numeric >= 10**11) & ~nanos_mask
    seconds_mask = (abs_numeric >= 10**9) & ~nanos_mask & ~millis_mask

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    if nanos_mask.any():
        result.loc[nanos_mask] = pd.to_datetime(numeric.loc[nanos_mask], unit="ns", utc=True, errors="coerce")
    if millis_mask.any():
        result.loc[millis_mask] = pd.to_datetime(numeric.loc[millis_mask], unit="ms", utc=True, errors="coerce")
    if seconds_mask.any():
        result.loc[seconds_mask] = pd.to_datetime(numeric.loc[seconds_mask], unit="s", utc=True, errors="coerce")
    return result


def _load_asset(spec: AssetSpec, *, freq: str | None) -> pd.DataFrame:
    frame = pd.read_csv(spec.path)
    ts_col = _find_col(frame, TIMESTAMP_CANDIDATES)
    if ts_col is None:
        raise SystemExit(f"{spec.name}: could not infer timestamp column from {spec.path}")

    price_col = _find_col(frame, PRICE_CANDIDATES)
    return_col = _find_col(frame, RETURN_CANDIDATES)
    if price_col is None and return_col is None:
        raise SystemExit(f"{spec.name}: could not infer price/return columns from {spec.path}")

    out = pd.DataFrame()
    out["timestamp"] = _coerce_timestamp(frame[ts_col])
    if price_col is not None:
        out["price"] = pd.to_numeric(frame[price_col], errors="coerce")
    if return_col is not None:
        out["return"] = pd.to_numeric(frame[return_col], errors="coerce")

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"{spec.name}: no valid timestamps in {spec.path}")

    if freq:
        agg_map: dict[str, str] = {}
        if "price" in out.columns:
            agg_map["price"] = "last"
        if "return" in out.columns:
            agg_map["return"] = "last"
        out = out.set_index("timestamp").resample(freq).agg(agg_map).reset_index()
        out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    if "price" in out.columns:
        out["price"] = out["price"].ffill()
        if "return" not in out.columns or out["return"].isna().all():
            out["return"] = out["price"].pct_change().fillna(0.0)
        else:
            out["return"] = out["return"].fillna(out["price"].pct_change())
    else:
        out["return"] = out["return"].fillna(0.0)

    out = out.dropna(subset=["return"]).drop_duplicates(subset=["timestamp"], keep="last")
    out = out.rename(
        columns={
            "price": f"{spec.name}__price",
            "return": f"{spec.name}__return",
        }
    )
    out[f"{spec.name}__observed"] = 1
    return out


def _build_panel(asset_frames: dict[str, pd.DataFrame], *, reference_asset: str, align: str) -> pd.DataFrame:
    ref = asset_frames[reference_asset].copy()
    panel = ref.copy()
    if align == "intersection":
        for name, frame in asset_frames.items():
            if name == reference_asset:
                continue
            panel = panel.merge(frame, on="timestamp", how="inner")
    else:
        for name, frame in asset_frames.items():
            if name == reference_asset:
                continue
            panel = panel.merge(frame, on="timestamp", how="left")

    observed_cols = [f"{name}__observed" for name in asset_frames]
    for col in observed_cols:
        if col not in panel.columns:
            panel[col] = 0
    panel[observed_cols] = panel[observed_cols].fillna(0).astype(int)
    return panel.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _window_availability(panel: pd.DataFrame, asset_names: list[str], *, window_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed_cols = [f"{name}__observed" for name in asset_names]
    for start in range(0, len(panel), window_size):
        chunk = panel.iloc[start : start + window_size]
        if chunk.empty:
            continue
        record: dict[str, Any] = {
            "row_start": int(start),
            "row_end": int(start + len(chunk) - 1),
            "timestamp_start": str(chunk.iloc[0]["timestamp"]),
            "timestamp_end": str(chunk.iloc[-1]["timestamp"]),
            "row_count": int(len(chunk)),
            "full_overlap_rows": int((chunk[observed_cols].sum(axis=1) == len(asset_names)).sum()),
        }
        for name in asset_names:
            record[f"{name}_observed_frac"] = float(chunk[f"{name}__observed"].mean())
        rows.append(record)
    return rows


def _summarize_panel(
    panel: pd.DataFrame,
    asset_specs: list[AssetSpec],
    asset_frames: dict[str, pd.DataFrame],
    *,
    reference_asset: str,
    align: str,
    freq: str | None,
    window_size: int,
) -> dict[str, Any]:
    asset_names = [spec.name for spec in asset_specs]
    observed_cols = [f"{name}__observed" for name in asset_names]
    overlap_counts = {name: int(panel[f"{name}__observed"].sum()) for name in asset_names}
    full_overlap = int((panel[observed_cols].sum(axis=1) == len(asset_names)).sum()) if not panel.empty else 0
    pairwise: dict[str, int] = {}
    for i, left in enumerate(asset_names):
        for right in asset_names[i + 1 :]:
            pairwise[f"{left}__{right}"] = int(
                ((panel[f"{left}__observed"] == 1) & (panel[f"{right}__observed"] == 1)).sum()
            )

    return {
        "assets": {spec.name: str(spec.path) for spec in asset_specs},
        "asset_inputs": {
            spec.name: {
                "raw_row_count": int(len(asset_frames[spec.name])),
                "timestamp_start": str(asset_frames[spec.name].iloc[0]["timestamp"]) if not asset_frames[spec.name].empty else None,
                "timestamp_end": str(asset_frames[spec.name].iloc[-1]["timestamp"]) if not asset_frames[spec.name].empty else None,
            }
            for spec in asset_specs
        },
        "reference_asset": reference_asset,
        "align": align,
        "freq": freq,
        "row_count": int(len(panel)),
        "timestamp_start": str(panel.iloc[0]["timestamp"]) if not panel.empty else None,
        "timestamp_end": str(panel.iloc[-1]["timestamp"]) if not panel.empty else None,
        "asset_overlap_counts": overlap_counts,
        "pairwise_overlap_counts": pairwise,
        "full_overlap_rows": full_overlap,
        "full_overlap_ratio": float(full_overlap / len(panel)) if len(panel) else 0.0,
        "missingness_by_asset": {
            name: float(1.0 - panel[f"{name}__observed"].mean()) for name in asset_names
        }
        if not panel.empty
        else {name: 1.0 for name in asset_names},
        "availability_windows": _window_availability(panel, asset_names, window_size=window_size),
    }


def main() -> None:
    args = parse_args()
    asset_specs = _parse_assets(args.asset)
    reference_asset = (args.reference_asset or asset_specs[0].name).upper()
    if reference_asset not in {spec.name for spec in asset_specs}:
        raise SystemExit(f"--reference-asset {reference_asset!r} is not present in --asset inputs")

    asset_frames = {spec.name: _load_asset(spec, freq=args.freq) for spec in asset_specs}
    panel = _build_panel(asset_frames, reference_asset=reference_asset, align=args.align)
    summary = _summarize_panel(
        panel,
        asset_specs,
        asset_frames,
        reference_asset=reference_asset,
        align=args.align,
        freq=args.freq,
        window_size=int(args.window_size),
    )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out, index=False)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

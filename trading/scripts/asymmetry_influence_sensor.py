#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from signals.triadic_ops import planes_to_symbol, score_to_planes


def _parse_symbol_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise SystemExit(f"Malformed --symbol argument: {arg}")
    name, path = arg.split("=", 1)
    return name, Path(path)


def _load_returns(path: Path, close_col: str | None = None) -> pd.Series:
    if not path.exists():
        raise SystemExit(f"Missing price file: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise SystemExit(f"{path.name} missing timestamp column")
    if close_col is None:
        candidates = ["close", "Close", "price", "Price"]
    else:
        candidates = [close_col]
    for col in candidates:
        if col in df.columns:
            closes = df[col].astype(float)
            break
    else:
        raise SystemExit(f"No close column found in {path}")
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    diff = np.log(closes).diff().to_numpy()
    series = pd.Series(diff, index=timestamps)
    return series.rename(path.stem)


def _build_dataframe(symbols: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for name, path in symbols.items():
        series = _load_returns(path).rename(name)
        frames.append(series)
    if not frames:
        raise SystemExit("No symbols provided")
    df = pd.concat(frames, axis=1).dropna()
    return df


def _serialize_rows(rows: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str))
            fh.write("\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Asymmetry influence tensor sensor.")
    ap.add_argument(
        "--symbol",
        action="append",
        required=True,
        help="Symbol=price.csv pairs (repeatable).",
    )
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--log-dir", type=Path, default=Path("logs/asymmetry"))
    args = ap.parse_args()

    symbol_map: dict[str, Path] = {}
    for spec in args.symbol:
        name, path = _parse_symbol_arg(spec)
        symbol_map[name] = path

    df = _build_dataframe(symbol_map)
    rows: list[dict[str, object]] = []
    for target in symbol_map:
        for source in symbol_map:
            if target == source:
                continue
            for lag in args.lags:
                shifted = df[source].shift(lag)
                combined = pd.concat([df[target], shifted], axis=1).dropna()
                if combined.empty:
                    score = float("nan")
                else:
                    score = float(combined[target].corr(combined[source]))
                p, n = score_to_planes(score, threshold=args.threshold)
                rows.append(
                    {
                        "target": target,
                        "source": source,
                        "lag": lag,
                        "score": score,
                        "threshold": args.threshold,
                        "p": p,
                        "n": n,
                        "symbol": planes_to_symbol(p, n),
                    }
                )

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = args.log_dir / f"influence_tensor_{stamp}.jsonl"
    _serialize_rows(rows, out_path)
    summary = Counter(row["symbol"] for row in rows)
    print(f"Wrote {len(rows)} influence entries to {out_path}")
    print("Triadic symbol counts:", dict(summary))


if __name__ == "__main__":
    main()

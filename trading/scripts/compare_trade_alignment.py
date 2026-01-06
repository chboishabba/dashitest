import argparse
import pathlib

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "p_bad",
    "edge_t",
    "edge_ema",
    "stress",
    "plane_abs",
    "plane_sign",
    "actionability",
    "acceptable",
    "belief_state",
    "thesis_depth",
    "capital_pressure",
    "permission",
    "action_t",
]

BELIEF_STATE_MAP = {
    "unk": -1,
    "flat": 0,
    "l1": 1,
    "l2": 2,
    "s1": -1,
    "s2": -2,
    "conflict": 3,
}

DECISION_KIND_MAP = {
    "unknown": -1,
    "flat": 0,
    "long": 1,
    "short": -1,
}


def parse_features(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_FEATURES[:]
    return [f.strip() for f in raw.split(",") if f.strip()]


def load_step_log(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"step log not found: {path}")
    df = pd.read_csv(path)
    if "t" not in df.columns:
        raise ValueError("step log missing column: t")
    return df


def load_trade_log(path: pathlib.Path, step_df: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        return df
    if "trade_closed" in step_df.columns:
        closed = step_df[step_df["trade_closed"] == 1].copy()
        if closed.empty:
            return closed
        closed["trade_id"] = closed.get("trade_id")
        closed["trade_pnl"] = closed.get("trade_pnl")
        closed["entry_step"] = closed.get("entry_step")
        closed["exit_step"] = closed["t"]
        closed["source"] = closed.get("source", "")
        return closed[
            ["trade_id", "trade_pnl", "entry_step", "exit_step", "source"]
        ]
    raise FileNotFoundError(f"trade log not found: {path}")


def encode_feature(series: pd.Series, name: str) -> pd.Series:
    if name == "belief_state":
        return series.map(BELIEF_STATE_MAP).astype(float)
    if name == "decision_kind":
        return series.map(DECISION_KIND_MAP).astype(float)
    if series.dtype == object:
        return series.astype("category").cat.codes.astype(float)
    return pd.to_numeric(series, errors="coerce")


def build_entry_frame(step_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    if "source" in step_df.columns:
        step_idx = step_df.set_index(["source", "t"])
        has_source = True
    else:
        step_idx = step_df.set_index("t")
        has_source = False

    rows = []
    for _, trade in trade_df.iterrows():
        entry_step = trade.get("entry_step")
        if pd.isna(entry_step):
            continue
        entry_step = int(entry_step)
        source = trade.get("source", "") if has_source else None
        try:
            if has_source:
                row = step_idx.loc[(source, entry_step)]
            else:
                row = step_idx.loc[entry_step]
        except KeyError:
            continue
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row_dict = row.to_dict()
        trade_pnl_close = trade.get("trade_pnl")
        row_dict.update(
            {
                "trade_id": trade.get("trade_id"),
                "trade_pnl": trade_pnl_close,
                "trade_pnl_close": trade_pnl_close,
                "entry_step": entry_step,
                "exit_step": trade.get("exit_step"),
                "source": source or trade.get("source", ""),
            }
        )
        rows.append(row_dict)
    return pd.DataFrame(rows)


def zscore_frame(
    df: pd.DataFrame,
    features: list[str],
    ref_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    cols = []
    stats = {}
    for name in features:
        if name not in df.columns:
            continue
        series = encode_feature(df[name], name)
        if series.isna().all():
            continue
        df[name] = series
        cols.append(name)
        ref_series = None
        if ref_df is not None and name in ref_df.columns:
            ref_series = encode_feature(ref_df[name], name)
        if ref_series is None or ref_series.isna().all():
            ref_series = series
        mean = float(ref_series.mean())
        std = float(ref_series.std())
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        stats[name] = (mean, std)
    if not cols:
        return df, []
    for name in cols:
        mean, std = stats[name]
        df[name] = (df[name] - mean) / std
    return df, cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/trading_log.csv", help="Per-step log CSV.")
    ap.add_argument("--trade-log", default="logs/trade_log.csv", help="Trade-close log CSV.")
    ap.add_argument("--features", default=None, help="Comma-separated feature list.")
    ap.add_argument("--top-k", type=int, default=1, help="Closest profitable entries to show.")
    args = ap.parse_args()

    step_df = load_step_log(pathlib.Path(args.log))
    trade_df = load_trade_log(pathlib.Path(args.trade_log), step_df)
    if trade_df.empty:
        print("No trade records found.")
        return 0

    entry_df = build_entry_frame(step_df, trade_df)
    if entry_df.empty:
        print("No entry rows matched in step log.")
        return 0

    features = parse_features(args.features)
    entry_df, used = zscore_frame(entry_df, features, ref_df=step_df)
    if not used:
        print("No usable features found in log.")
        return 0
    entry_df[used] = entry_df[used].fillna(0.0)

    pnl_col = "trade_pnl_close" if "trade_pnl_close" in entry_df.columns else "trade_pnl"
    profitable = entry_df[entry_df[pnl_col] > 0].copy()
    losing = entry_df[entry_df[pnl_col] < 0].copy()
    if profitable.empty:
        print("No profitable trades found to compare.")
        return 0
    if losing.empty:
        print("No losing trades found.")
        return 0

    prof_matrix = profitable[used].to_numpy(dtype=float)

    for _, loss in losing.iterrows():
        loss_vec = loss[used].to_numpy(dtype=float)
        dists = np.linalg.norm(prof_matrix - loss_vec, axis=1)
        order = np.argsort(dists)
        top_k = max(1, args.top_k)
        pnl_val = loss.get(pnl_col)
        print(
            f"Loss trade id={loss.get('trade_id')} source={loss.get('source')} "
            f"entry={loss.get('entry_step')} exit={loss.get('exit_step')} "
            f"pnl={pnl_val:.4f}"
        )
        for rank in range(min(top_k, len(order))):
            idx = order[rank]
            prof = profitable.iloc[idx]
            print(
                f"  closest[{rank+1}] id={prof.get('trade_id')} "
                f"source={prof.get('source')} entry={prof.get('entry_step')} "
                f"pnl={prof.get(pnl_col):.4f} dist={dists[idx]:.4f}"
            )
            snapshot = {k: prof.get(k) for k in used}
            print(f"    inputs={snapshot}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

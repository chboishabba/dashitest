#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def _load_ndjson(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _apply_symbol_filter(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return df
    return df[df["symbol"].isin(symbols)]


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "direction", "target_exposure"])
    df["signed_exposure"] = df["direction"].astype(float) * df["target_exposure"].astype(float)
    return df.sort_values("timestamp")


def _plot_exposure(
    df: pd.DataFrame,
    *,
    show_state: bool,
    urgency_thickness: bool,
    show_posture: bool,
    follow: bool,
    interval_ms: int,
    window_seconds: float | None,
    webm: Path | None,
    fps: int,
    frame_step: int,
    follow_path: Path | None,
    output: Path | None,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    symbols = sorted(df["symbol"].unique().tolist())
    posture_mask = None
    if show_posture and "posture" in df.columns:
        posture_mask = df["posture"] == 0
        _add_background_bands(ax, df, posture_mask, color="#e0e0e0", alpha=0.4)

    lines = {}
    scatters = {}
    for symbol in symbols:
        line, = ax.plot([], [], label=symbol, linewidth=2.0)
        lines[symbol] = line
        scatters[symbol] = {}
        if show_state:
            scatters[symbol]["pos"] = ax.scatter([], [], color="green", s=18, alpha=0.7)
            scatters[symbol]["neg"] = ax.scatter([], [], color="red", s=18, alpha=0.7)
        if urgency_thickness:
            scatters[symbol]["urgency"] = ax.scatter([], [], s=10, alpha=0.6)

    ax.set_title("Signed target exposure (direction × target_exposure)")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("signed exposure")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)

    def _apply_window(frame: pd.DataFrame) -> pd.DataFrame:
        if window_seconds is None or frame.empty:
            return frame
        cutoff = frame["timestamp"].max() - pd.Timedelta(seconds=window_seconds)
        return frame[frame["timestamp"] >= cutoff]

    def _to_mpl_dates(series: pd.Series) -> np.ndarray:
        return mdates.date2num(pd.to_datetime(series, utc=True).to_numpy())

    def _render(frame: pd.DataFrame) -> None:
        frame = _apply_window(frame)
        for symbol in symbols:
            sym_df = frame[frame["symbol"] == symbol]
            if sym_df.empty:
                lines[symbol].set_data([], [])
                for scatter in scatters[symbol].values():
                    scatter.set_offsets([])
                continue
            lines[symbol].set_data(sym_df["timestamp"], sym_df["signed_exposure"])
            if urgency_thickness and "urgency" in sym_df.columns:
                urgency = sym_df["urgency"].clip(0.0, 1.0).fillna(0.0)
                sizes = 6 + 18 * urgency
                offsets = np.column_stack(
                    (_to_mpl_dates(sym_df["timestamp"]), sym_df["signed_exposure"].to_numpy())
                )
                scatters[symbol]["urgency"].set_offsets(
                    offsets
                )
                scatters[symbol]["urgency"].set_sizes(sizes.to_numpy())
            if show_state and "state" in sym_df.columns:
                state_pos = sym_df[sym_df["state"] == 1]
                state_neg = sym_df[sym_df["state"] == -1]
                pos_offsets = np.column_stack(
                    (_to_mpl_dates(state_pos["timestamp"]), state_pos["signed_exposure"].to_numpy())
                )
                neg_offsets = np.column_stack(
                    (_to_mpl_dates(state_neg["timestamp"]), state_neg["signed_exposure"].to_numpy())
                )
                scatters[symbol]["pos"].set_offsets(
                    pos_offsets
                )
                scatters[symbol]["neg"].set_offsets(
                    neg_offsets
                )
        ax.relim()
        ax.autoscale_view()

    if webm:
        if output:
            output = None
        times = pd.to_datetime(df["timestamp"]).sort_values().unique()
        if frame_step > 1:
            times = times[::frame_step]
        series = {}
        for symbol in symbols:
            sym_df = df[df["symbol"] == symbol]
            series[symbol] = {
                "times": sym_df["timestamp"].to_numpy(),
                "exposure": sym_df["signed_exposure"].to_numpy(),
                "state": sym_df["state"].to_numpy() if "state" in sym_df.columns else None,
                "urgency": sym_df["urgency"].to_numpy() if "urgency" in sym_df.columns else None,
            }
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        def _update(idx: int):
            t = times[idx]
            frame_rows = []
            for symbol in symbols:
                sym = series[symbol]
                end = np.searchsorted(sym["times"], t, side="right")
                frame_rows.append(
                    pd.DataFrame(
                        {
                            "timestamp": sym["times"][:end],
                            "symbol": symbol,
                            "signed_exposure": sym["exposure"][:end],
                            "state": sym["state"][:end] if sym["state"] is not None else None,
                            "urgency": sym["urgency"][:end] if sym["urgency"] is not None else None,
                        }
                    )
                )
            frame = pd.concat(frame_rows, ignore_index=True)
            _render(frame)
            return list(lines.values())

        anim = FuncAnimation(fig, _update, frames=len(times), interval=1000 / max(fps, 1))
        webm.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = FFMpegWriter(fps=max(fps, 1), codec="libvpx")
            anim.save(webm, writer=writer)
        except FileNotFoundError as exc:
            raise SystemExit("ffmpeg not found; install it to write WebM.") from exc
    elif follow:
        if follow_path is None:
            raise SystemExit("follow mode requires an NDJSON path")
        plt.ion()
        _render(df)
        plt.show(block=False)
        with follow_path.open("r", encoding="utf-8") as fh:
            fh.seek(0, 2)
            try:
                while True:
                    line = fh.readline()
                    if not line:
                        time.sleep(interval_ms / 1000.0)
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    new_df = pd.DataFrame([payload])
                    new_df = _prepare_frame(new_df)
                    if new_df.empty:
                        continue
                    df = pd.concat([df, new_df], ignore_index=True)
                    if window_seconds is not None:
                        cutoff = df["timestamp"].max() - pd.Timedelta(seconds=window_seconds)
                        df = df[df["timestamp"] >= cutoff]
                    _render(df)
                    plt.pause(0.001)
            except KeyboardInterrupt:
                pass
        plt.ioff()
    else:
        _render(df)
        fig.tight_layout()
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output)
        if show:
            plt.show()
        plt.close(fig)


def _add_background_bands(ax, df: pd.DataFrame, mask: pd.Series, *, color: str, alpha: float) -> None:
    if mask.empty:
        return
    in_band = False
    start = None
    for ts, flag in zip(df["timestamp"], mask):
        if flag and not in_band:
            start = ts
            in_band = True
        elif not flag and in_band:
            ax.axvspan(start, ts, color=color, alpha=alpha)
            in_band = False
    if in_band and start is not None:
        ax.axvspan(start, df["timestamp"].iloc[-1], color=color, alpha=alpha)


def _plot_state(df: pd.DataFrame, *, output: Path | None, show: bool) -> None:
    if "state" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 2.8))
    symbols = sorted(df["symbol"].unique().tolist())
    for symbol in symbols:
        sym_df = df[df["symbol"] == symbol]
        ax.step(sym_df["timestamp"], sym_df["state"], where="post", label=symbol)
    ax.set_title("State over time")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("state")
    ax.set_yticks([-1, 0, 1])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot signed target exposure from decision NDJSON.")
    ap.add_argument("--ndjson", required=True, help="Path to decision NDJSON file")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbol list to include")
    ap.add_argument("--out", default=None, help="Output image path (PNG)")
    ap.add_argument("--show", action="store_true", help="Show interactive plot")
    ap.add_argument("--show-state", action="store_true", help="Overlay state markers on exposure")
    ap.add_argument("--urgency-thickness", action="store_true", help="Scale marker size by urgency")
    ap.add_argument("--show-posture", action="store_true", help="Shade OBSERVE posture periods")
    ap.add_argument("--state-plot", action="store_true", help="Generate a separate state step plot")
    ap.add_argument("--state-out", default=None, help="Output path for state plot (PNG)")
    ap.add_argument("--follow", action="store_true", help="Tail NDJSON and update the plot live")
    ap.add_argument("--interval-ms", type=int, default=1000, help="Live update interval in ms")
    ap.add_argument("--window-seconds", type=float, default=None, help="Keep last N seconds visible")
    ap.add_argument("--webm", default=None, help="Write a WebM timelapse to this path")
    ap.add_argument("--fps", type=int, default=10, help="Frames per second for WebM")
    ap.add_argument("--frame-step", type=int, default=1, help="Use every Nth timestamp for WebM")
    args = ap.parse_args()

    path = Path(args.ndjson)
    if not path.exists():
        raise SystemExit(f"NDJSON not found: {path}")
    df = _load_ndjson(path)
    if df.empty:
        raise SystemExit("No decision rows found.")
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else []
    df = _apply_symbol_filter(df, symbols)
    if df.empty:
        raise SystemExit("No rows after symbol filtering.")
    df = _prepare_frame(df)

    out_path = Path(args.out) if args.out else None
    _plot_exposure(
        df,
        show_state=args.show_state,
        urgency_thickness=args.urgency_thickness,
        show_posture=args.show_posture,
        follow=args.follow,
        interval_ms=args.interval_ms,
        window_seconds=args.window_seconds,
        webm=Path(args.webm) if args.webm else None,
        fps=args.fps,
        frame_step=max(1, args.frame_step),
        follow_path=path,
        output=out_path,
        show=args.show,
    )
    if args.state_plot:
        state_out = Path(args.state_out) if args.state_out else None
        _plot_state(df, output=state_out, show=args.show)


if __name__ == "__main__":
    main()

"""
plot_energy_landscape.py
------------------------
Plot a 2D energy landscape grid (heatmap) and optional slope map.

Usage:
  python plot_energy_landscape.py --grid path/to/landscape.npy --save outputs/landscape.png
  python plot_energy_landscape.py --grid path/to/landscape.csv --save outputs/landscape.png --slope
"""

import argparse
import pathlib
import datetime
import re

import matplotlib.pyplot as plt
import numpy as np


_TS_RE = re.compile(r".*_[0-9]{8}T[0-9]{6}Z$")


def timestamped_path(path: pathlib.Path) -> pathlib.Path:
    if _TS_RE.match(path.stem):
        return path
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if path.suffix:
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")
    return path.with_name(f"{path.name}_{ts}")


def load_grid(path: pathlib.Path) -> np.ndarray:
    if path.suffix == ".npy":
        grid = np.load(path)
    elif path.suffix == ".csv":
        grid = np.loadtxt(path, delimiter=",")
    else:
        raise SystemExit("Unsupported grid format. Use .npy or .csv.")
    if grid.ndim != 2:
        raise SystemExit(f"Expected 2D grid, got shape {grid.shape}.")
    return grid


def plot_grid(grid: np.ndarray, title: str, ax) -> None:
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, required=True, help="Path to .npy or .csv 2D grid")
    ap.add_argument("--save", type=str, required=True, help="Output image path")
    ap.add_argument("--slope", action="store_true", help="Plot gradient magnitude alongside grid")
    args = ap.parse_args()

    grid_path = pathlib.Path(args.grid)
    grid = load_grid(grid_path)

    if args.slope:
        grad_y, grad_x = np.gradient(grid)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_grid(grid, "Energy landscape", axes[0])
        plot_grid(slope, "Slope magnitude", axes[1])
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        plot_grid(grid, "Energy landscape", axes)

    plt.tight_layout()
    save_path = timestamped_path(pathlib.Path(args.save))
    plt.savefig(save_path, dpi=200)
    print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    main()

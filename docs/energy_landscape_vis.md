# Energy Landscape Visualization

This notes how to visualize energy landscape grids from dashifine experiments.
Context reference: `CONTEXT.md#L1269`.

## Expected inputs

- 2D grid saved as `.npy` (NumPy) or `.csv` (comma-delimited).
- Grid values represent energy (or score) over a 2D parameter sweep.

## Usage

```
python plot_energy_landscape.py --grid path/to/landscape.npy --save outputs/landscape.png
python plot_energy_landscape.py --grid path/to/landscape.csv --save outputs/landscape.png
python plot_energy_landscape.py --grid path/to/landscape.npy --save outputs/landscape.png --slope
```

Outputs are auto-timestamped to avoid overwriting prior runs (see
`CONTEXT.md#L2532`).

## Outputs

- Heatmap of the landscape values.
- Optional slope map (gradient magnitude) when `--slope` is set.

## Open items

- Confirm where `dashifine/newtest1` and `dashifine/newtest2` outputs live.
- Confirm grid orientation (axis labels or parameter names) to label plots.

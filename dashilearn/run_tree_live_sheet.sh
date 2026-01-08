#!/usr/bin/env bash
set -euo pipefail

# Run the tree diffusion benchmark with live sheet visualization and recording.

LIVE_SHEET_PATH="/tmp/tree_live_sheet.npy"

# Ensure the live sheet path is cleaned up on exit
trap "rm -f \"$LIVE_SHEET_PATH\"" EXIT

# Run tree_diffusion_bench.py in the background to continuously update the sheet file
python tree_diffusion_bench.py \
    --adv-op \
    --depth-decay-strength 0.5 \
    --depth-decay-mode linear \
    --dump-live-sheet "$LIVE_SHEET_PATH" \
    --alpha 0.8 \
    --decay 0.2 \
    --depth 4 \
    --rollout-steps 1000 \
    &
BENCH_PID=$!

# Wait a moment for the first sheet file to be written
sleep 1.0

# Launch the Vulkan previewer to read the live sheet and record it
# The previewer handles dynamic sheet sizes and stretching
python vulkan_compute/compute_image_preview.py \
    --sheet \
    --sheet-data "$LIVE_SHEET_PATH" \
    --sheet-data-interval 0.1 \
    --record-video \
    --record-fps 30 \
    --record-out "tree_diffusion_$(date +%Y%m%dT%H%M%SZ).mp4" \
    --alpha 0.97 \
    --block-px 16 \
    --sheet-w 27 --sheet-h 27 \
    --vmax 1.0 --vmin -1.0 # Assuming signal is roughly in [-1, 1] for visualization

# Wait for the previewer to finish, or for the benchmark to finish if previewer stays open
wait $BENCH_PID

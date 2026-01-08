#!/usr/bin/env bash
set -euo pipefail

# Run the learner (stays open to refresh the sheet) and the Vulkan preview
# together so you can watch/record band-energy sheets without juggling shells.

SHEET_PATH="$(dirname "$0")/sheet_energy.npy"
export SHEET_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json

python dashilearn/bsmoe_train.py --epochs 5 &
TRAINER_PID=$!

for _ in $(seq 1 200); do
  if [[ -f "$SHEET_PATH" ]]; then
    break
  fi
  sleep 0.1
done

SHEET_DIMS=$(python - <<'PY'
import numpy as np
from pathlib import Path
import os

path = Path(os.environ["SHEET_PATH"])
if not path.exists():
    raise SystemExit("sheet_energy.npy not found yet")
arr = np.load(path)
if arr.ndim != 2:
    raise SystemExit(f"sheet_energy.npy is not 2D: {arr.ndim}D")
print(arr.shape[0], arr.shape[1])
PY
)
read -r SHEET_H SHEET_W <<< "$SHEET_DIMS"

python vulkan_compute/compute_image_preview.py --sheet --sheet-data "$SHEET_PATH" --sheet-w "$SHEET_W" --sheet-h "$SHEET_H" --record-video --record-fps 30 --record-out sheet.mp4 "$@" \
  && wait "$TRAINER_PID"

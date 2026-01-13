from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vk_qfeat import QFeatTape


@dataclass
class VulkanTapeAdapter:
    """
    Adapts a precomputed qfeat/ℓ tape into the LearnerAdapter interface.
    The caller must populate `ts_to_index` so timestamps map to dense bar
    indices matching the tape that was generated offline.
    """
    tape: QFeatTape
    series_index: int = 0
    ts_to_index: dict[int, int] | None = None

    def __post_init__(self):
        if self.ts_to_index is None:
            self.ts_to_index = {}

    def update(self, ts: int, payload: dict) -> tuple[float, np.ndarray]:
        idx = self.ts_to_index.get(int(ts))
        if idx is None:
            raise KeyError(f"timestamp {ts} is not indexed in the tape map")

        record = self.tape.mm[self.series_index, idx]
        qfeat = record[:6].astype(np.float32, copy=True)
        ell = float(record[6])
        return ell, qfeat

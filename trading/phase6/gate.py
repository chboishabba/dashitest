from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd


class Phase6ExposureGate:
    """Reads the latest Phase-6 capital control log and reports whether exposure is allowed."""

    def __init__(self, log_dir: str | Path = "logs/phase6"):
        self.log_dir = Path(log_dir)

    def _latest_file(self) -> Path | None:
        candidates = sorted(self.log_dir.glob("capital_controls_*.jsonl"))
        return candidates[-1] if candidates else None

    def _entries(self) -> Iterator[dict]:
        latest = self._latest_file()
        if latest is None or not latest.exists():
            return
        with latest.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def allowed_slip(self) -> float | None:
        """Return the slip threshold currently marked as allowed."""
        for entry in self._entries():
            if entry.get("allowed"):
                slip = entry.get("slip_bps")
                try:
                    return float(slip)
                except (TypeError, ValueError):
                    continue
        return None

    def is_allowed(self) -> bool:
        """Returns True when a slip level is currently approved."""
        return self.allowed_slip() is not None

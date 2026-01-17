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
            return iter(())
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
        allowed = self._latest_allowed()
        if allowed is None:
            return None
        try:
            return float(allowed.get("slip_bps"))
        except (TypeError, ValueError):
            return None

    def is_allowed(self) -> bool:
        """Returns True when a slip level is currently approved."""
        return self.allowed_slip() is not None

    def snapshot(self) -> dict[str, object]:
        """Return a diagnostic snapshot of the current gate verdict."""
        latest = self._latest_file()
        allowed = self._latest_allowed()
        snapshot: dict[str, object] = {
            "open": allowed is not None,
            "source": latest.name if latest else None,
            "allowed_slip_bps": None,
            "reason": None,
        }
        if allowed is None:
            return snapshot
        slip = allowed.get("slip_bps")
        try:
            snapshot["allowed_slip_bps"] = float(slip) if slip is not None else None
        except (TypeError, ValueError):
            snapshot["allowed_slip_bps"] = None
        if "reason" in allowed:
            snapshot["reason"] = allowed.get("reason")
        if "timestamp" in allowed:
            snapshot["timestamp"] = allowed.get("timestamp")
        return snapshot

    def _latest_allowed(self) -> dict | None:
        """Return the latest allowed entry from the current log."""
        allowed_entry = None
        for entry in self._entries():
            if entry.get("allowed"):
                allowed_entry = entry
        return allowed_entry

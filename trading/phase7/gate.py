from __future__ import annotations

import json
from collections import deque
from pathlib import Path


class Phase7ReadinessGate:
    """Reads the latest Phase-07 readiness log and reports a persistence verdict."""

    def __init__(
        self,
        log_path: str | Path = "logs/phase7/density_status.log",
        *,
        target: str | None = None,
        persistence_window: int = 8,
        persistence_required: int = 6,
    ) -> None:
        self.log_path = Path(log_path)
        self.target = target or ""
        self.persistence_window = max(1, int(persistence_window))
        self.persistence_required = max(1, int(persistence_required))
        self._cache_mtime: float | None = None
        self._cache_entries: dict[str, list[dict]] = {}

    def _load_entries(self) -> dict[str, list[dict]]:
        if not self.log_path.exists():
            self._cache_entries = {}
            self._cache_mtime = None
            return self._cache_entries
        try:
            mtime = self.log_path.stat().st_mtime
        except OSError:
            return self._cache_entries
        if self._cache_mtime == mtime:
            return self._cache_entries
        entries: dict[str, deque] = {}
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target = (
                    payload.get("target")
                    or payload.get("symbol")
                    or payload.get("tape")
                    or payload.get("name")
                    or "default"
                )
                if not isinstance(target, str):
                    target = "default"
                bucket = entries.setdefault(target, deque(maxlen=self.persistence_window))
                bucket.append(payload)
        self._cache_mtime = mtime
        self._cache_entries = {key: list(val) for key, val in entries.items()}
        return self._cache_entries

    def snapshot(self, target: str | None = None) -> dict[str, object]:
        target_name = (target or self.target or "default").strip() or "default"
        entries = self._load_entries()
        candidates = entries.get(target_name) or entries.get("default") or []
        if not candidates:
            return {
                "open": False,
                "reason": "phase7_status_missing",
                "target": target_name,
                "ready_count": 0,
                "window_size": 0,
                "required": self.persistence_required,
            }
        window = candidates[-self.persistence_window :]
        required = self.persistence_required
        ready_count = sum(1 for entry in window if entry.get("phase7_ready") is True)
        metrics = {}
        if window:
            last = window[-1]
            maybe = last.get("phase7_metrics") or last.get("metrics") or {}
            if isinstance(maybe, dict):
                metrics = dict(maybe)
                count = metrics.get("count")
                total = metrics.get("window")
                if count is not None and total:
                    try:
                        metrics["activity_rate"] = float(count) / float(total)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
        if len(window) < required:
            reason = f"phase7_waiting {len(window)}/{required}"
            open_gate = False
        elif ready_count < required:
            last_reason = ""
            for entry in reversed(window):
                reason_val = entry.get("phase7_reason")
                if reason_val:
                    last_reason = str(reason_val)
                    break
            reason = last_reason or f"phase7_not_ready {ready_count}/{required}"
            open_gate = False
        else:
            reason = "phase7_ready"
            open_gate = True
        return {
            "open": open_gate,
            "reason": reason,
            "target": target_name,
            "ready_count": ready_count,
            "window_size": len(window),
            "required": required,
            "metrics": metrics,
        }

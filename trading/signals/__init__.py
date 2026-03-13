"""Signal extraction helpers."""

try:
    from .asymmetry_sensor import InfluenceTensorMonitor
except ModuleNotFoundError:  # Optional dependency path (duckdb) for lightweight imports.
    InfluenceTensorMonitor = None

__all__ = ["InfluenceTensorMonitor"]

from execution.base import BaseExecution


class BarExecution(BaseExecution):
    """Placeholder for existing bar-level execution (use run_trader logic)."""

    def execute(self, intents):
        # This backend is already effectively in run_trader; this stub exists
        # to keep the execution interface consistent. To be wired in runner.
        return [], {}

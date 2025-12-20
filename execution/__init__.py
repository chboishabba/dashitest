"""
Execution adapters:

BaseExecution: interface for intent -> fills
BarExecution: existing bar-level execution (stub to be wired)
LOBReplayExecution: hftbacktest-based LOB replay (stub)
"""

class BaseExecution:
    def execute(self, intents):
        """Given a list of intents, return list of fills and summary metrics."""
        raise NotImplementedError

"""
Compatibility shim.

The trading engine under `trading/engine/` imports `execution.*` when run from
within the `trading/` directory, where `trading/execution/` is the intended
package. Some test runners add the repo root to `sys.path`, which can cause
imports to resolve to the repo-root `execution/` package instead.

Provide thin re-exports so `execution.accounting` works in both layouts.
"""

from trading.execution.accounting import *  # noqa: F401,F403


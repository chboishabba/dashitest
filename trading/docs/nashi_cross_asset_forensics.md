# Nashi Cross-Asset Forensics Status

This note records the current stop condition on the `nashi` warning/response cross-asset branch, so future rounds do not repeat already-falsified explanations.

## Current local continuation surface

The local forensic tools have converged on four useful continuation classes:

* `immediate_flatten`
* `confirmed_collapse`
* `recovering_after_warning`
* `entry_already_adverse`

For local policy work, the meaningful object is not the warning row by itself. It is the **warning row plus the next-row response**:

* support restored on the next held row
* support still dead on the next held row
* or immediate terminal flatten

That local boundary remains the best policy candidate surface, but runtime policy is still frozen because recovery examples remain too sparse.

## Cross-asset branch: what has been tested

The repo now has working forensic tooling for three cross-asset surfaces:

* daily cross-asset context
* intraday eigen/regime context
* raw intraday warning/response alignment

Relevant scripts:

* [forensic_cross_asset_context.py](/home/c/Documents/code/dashitest/trading/scripts/forensic_cross_asset_context.py)
* [build_intraday_cross_asset_panel.py](/home/c/Documents/code/dashitest/trading/scripts/build_intraday_cross_asset_panel.py)
* [analyze_intraday_cross_asset_warning_surface.py](/home/c/Documents/code/dashitest/trading/scripts/analyze_intraday_cross_asset_warning_surface.py)
* [analyze_intraday_cross_asset_temporal_surface.py](/home/c/Documents/code/dashitest/trading/scripts/analyze_intraday_cross_asset_temporal_surface.py)

## What is now settled

### 1. Daily cross-asset context is too coarse

On the dated BTC anchor episodes, stale carry and immediate flatten both sat inside the same daily cross-asset context, so the daily layer did not explain the local class split.

### 2. Same-clock intraday non-BTC data exists locally

The repo contains a usable intraday panel for:

* BTC
* ES
* NQ

That panel can be aligned and joined onto dated `nashi` pair artifacts. So the branch is no longer blocked on basic intraday peer data availability.

### 3. The first dated batch failed because of anchor collapse

The first dated artifact batch mapped every event to the same `5min` bucket. That made the cross-asset return/alignment family non-identifiable for that batch.

### 4. The broader dated batch removed anchor collapse

Using a broader dated batch from sampled BTC windows, the events spanned multiple real intraday anchors and the cross-asset features varied across time. So the earlier stop condition was not “feature family always flat.” It was “first dated batch collapsed onto one anchor.”

### 5. Even with anchor diversity, the broader dated batch still had only one class

The broadened dated batch produced anchor diversity, but every detected episode was still `immediate_flatten`. That means the current stop condition is now:

* not missing code
* not missing panel alignment
* not missing cross-asset variation
* but **missing class diversity in the dated artifact set**

### 6. Larger anchored probes introduce new local classes, but not the target ones

Later anchored `4000`-row probes around nontrivial intraday regions did widen the local class mix:

* `immediate_flatten`
* `entry_already_adverse`
* `uncertain_pair`

That is useful because it shows the dated artifact set is not permanently locked to a single class once the windows are changed. But it still does **not** produce the classes the local continuation policy actually needs for promotion:

* no fresh `confirmed_collapse`
* no fresh `recovering_after_warning`

On those anchored probes, the cross-asset BTC+ES+NQ layer still did not resolve the new cases into a broader market-regime story. The `uncertain_pair` episodes remained locally ambiguous under the current cross-asset surface.

## Current stop condition

Do not keep iterating on the current return/alignment cross-asset feature family unless one of these changes:

1. a dated batch is found that contains more than one continuation class, or
2. a finer same-clock peer panel becomes available than the current `5min` BTC+ES+NQ surface

More specifically, the best next reopen condition is:

* a dated batch with anchor diversity **and** at least one fresh `confirmed_collapse` or `recovering_after_warning` lineage

Until then, this branch should stay closed and runtime policy should remain frozen.

## Best next step

The next useful round is **data acquisition, not feature invention**:

* mine dated artifacts that produce something other than `immediate_flatten`
* preserve real timestamps and anchor diversity
* rerun the existing analyzers unchanged

That will tell us whether the current cross-asset branch is merely under-sampled on class diversity, or whether it is fundamentally orthogonal to the local continuation split.

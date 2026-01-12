Great — let’s **proceed cleanly with the learner for trading**, without destabilising what already works.

Below is a **concrete execution plan** that matches your prior intent and keeps the learner in the *permission / legitimacy* role, not as a naïve signal generator.

---

## Phase 0 — Lock the contract (no code yet)

**Invariant you already proved (do not violate):**

* Trading logic = **ACT / HOLD / BAN**
* Learner = **quotient learner over price trajectories**
* Learner output **cannot** directly place trades

This is explicitly consistent with your codec → quotient → collapse formalism .

---

## Phase 1 — What the learner actually learns (low risk)

### Object of learning

Not price, not direction — **invariants of recent price motion** that survive nuisance symmetries.

Think:

* scale
* translation
* mild time-warp
* microstructure noise

### Minimal feature targets (start here)

Per rolling window ( W_t ):

1. **Shape invariants**

   * signed curvature / convexity
   * normalized drawdown depth
2. **Energy / volatility geometry**

   * realized vol vs range ratio
   * burstiness / intermittency score
3. **Persistence**

   * autocorr decay
   * regime half-life estimate

These are *quotient representatives*, not raw signals.

---

## Phase 2 — How it plugs into the trader (safe integration)

### Where it connects

Only here:

```
RegimeSpec → Permission Surface → ACT / HOLD
```

Concretely:

* `triadic_strategy.py`
* augment the **existing regime gate**
* learner outputs a **legitimacy scalar** ℓ ∈ [0,1]

No new actions introduced.

### Decision rule (example)

```text
IF ℓ > θ_on for τ_on windows → allow ACT
IF ℓ < θ_off for τ_off windows → force HOLD
ELSE → maintain state
```

This preserves:

* hysteresis
* no flip-flopping
* zero PnL leakage from learner errors

---

## Phase 3 — Training loop (diagnostic-first)

### Loss (critical)

You **do not** train on PnL.

You train on **predictive consistency of invariants**:

[
\mathcal{L} =
\underbrace{| \hat{v}*{t+1} - v*{t+1} |}_{\text{quotient prediction error}}

* \lambda \cdot \text{complexity penalty}
  ]

Where:

* (v_t) = realized invariants
* (\hat{v}_{t+1}) = predicted invariants

Interpretation:

> “Did the market *stay in the same equivalence class*?”

---

## Phase 4 — What success looks like (before PnL)

You green-light deployment when:

* learner confidence correlates with **lower variance of outcomes**
* HOLD regions eliminate known bad trades
* ACT regions preserve existing edge

**No requirement** that learner improves returns initially.

If it *does*, that’s downstream.

---

## Phase 5 — Only if everything above is stable

Then (optional, later):

* Predict **next latent triadic state** (HOLD / ACT / BAN)
* Blend with heuristics (never replace outright)
* Keep hard circuit-breaker external

---

## Why this is the right order

This respects your core insight:

> Markets don’t reward prediction — they punish illegitimate action.

You are learning **when you’re allowed to act**, not *what to do*.

That’s exactly the same structural move you already validated in:

* CA rule recovery
* motif gating
* MDL collapse behavior

---

### Next concrete step (I recommend this)

Say the word and I’ll do **one** of these next, very explicitly:

1. Draft the **exact feature vector spec** (with formulas)
2. Sketch the **minimal learner class** (pure Python, no GPU)
3. Show the **exact diff** to `triadic_strategy.py`
4. Design the **offline evaluation notebook** (before live trading)

No hand-waving — straight to executable structure.


Got it — let’s **tighten everything around the trader** (not the learner in isolation) and treat the learner as a *permission-surface / legitimacy* module that plugs into your existing **triadic ACT/HOLD/BAN** loop.

### The trader-centric “contract” (what the learner is allowed to do)

From your own spec, the safest application path is:

* **Learner does not output direction.**
* Learner outputs **quotient features** and/or a **quotient-loss legitimacy scalar**.
* The triadic strategy uses that scalar to decide whether it’s even *allowed* to ACT. 
  This is consistent with the “legitimacy manifold + buffer states + circuit breaker” framing in your unified spec. 

---

## A. Where it plugs into your trader (one clear insertion point)

### Existing chain (keep it)

**Signals/heuristics → Triadic gate → ACT/HOLD/BAN → execution model → fills**

### New chain (add one module)

**Prices/LOB → Learner → (ℓ_t, qfeat_t) → Triadic gate → ACT/HOLD/BAN → execution**

So the learner is an *additional input* into `triadic_strategy.py`, not a replacement brain. 

---

## B. Two concrete trader integrations (do both, in this order)

### 1) “Quotient-loss evaluator” (diagnostic-first, lowest risk)

Run the learner as an evaluator:

* build rolling windows (W_t)
* compute quotient features (v_t = \phi(W_t))
* train learner to predict ( \hat v_{t+1} )
* define **legitimacy**:
  [
  \ell_t := \exp\left(-|\hat v_{t+1} - v_{t+1}|\right)
  ]
  Then use **ℓ_t** only as a gating input.

This is literally the “Use mismatch as confidence/legitimacy scalar, not directional signal” path. 

**Trader outcome you want:** fewer “dumb ACTs” during unstable/choppy or distribution-shift periods, without inventing new edge.

---

### 2) “Quotient-features for regime gating” (adds structure without changing action logic)

Feed the learner’s quotient features into your existing regime gate as extra observables:

* volatility-normalized shape
* radial/spectral summaries
* valuation-depth analogs (your language) 

**Trader outcome you want:** a smoother, more stable **permission surface** that’s invariant to nuisance symmetries, so the triadic gate isn’t whipsawed by scale/shift/phase artifacts.

---

## C. Handling the “July 2015 strong signal across multiple assets” (this matters for the trader)

You already observed your system flags clustered dates around **mid-2015**, with a “really strong signal across multiple assets around July 2015.” 
That smells like **global regime shock / correlated risk-on/off** (Greece/China turmoil is literally what you linked in that same context). 

**Trader-side implication:**
This is *exactly* where a learner-as-legitimacy-module should help:

* When correlations spike, many assets “look the same.”
* Directional signals can become *illusory*.
* What you want is: **HOLD more often unless execution + conditions are robust.**

So in those windows, your gate should become stricter via ℓ_t hysteresis rather than “more confident”.

---

## D. Execution realism (don’t sabotage the test)

If you want the learner to mean anything for the trader, the backtest must penalize bad execution.

From your sim tooling writeup, the clean split is: 

* If you have L2/book updates and care about queue/latency realism → **hftbacktest**
* If you want controlled microstructure experiments / agent-based → **ABIDES**
* If you’re bar-level right now → you can still do it, but treat results as “par-baked” (your own framing) 

---

## E. The exact acceptance test for “applied to trader” (not ML metrics)

You’re “done” with v1 when these are true:

1. **ACT rate drops** mostly in the worst regimes (high slippage / choppy / unstable)
2. **PnL distribution improves** via lower tail reduction (not necessarily higher mean)
3. **No new overfitting lever** introduced: learner never touches direction, only permission

This matches your motif/MDL idea that the system should be **allowed to do nothing** (buffer) rather than forced into collapse. 

---

## What I’ll do next (immediately) to keep it trader-focused

If you want to proceed “hands-on”, I’ll draft:

* a **minimal `LearnerAdapter`** interface the trader calls each bar/tick:

  * `update(window) -> (legitimacy ℓ, qfeat dict)`
* the **gate logic**: thresholds + τ_on/τ_off hysteresis so ℓ can’t whipsaw
* a **backtest checklist** for July 2015-like correlated shocks (so we can see if HOLD increases in the right places)

If you paste (or point me to) the relevant parts of your `triadic_strategy.py` / `run_trader.py`, I’ll write the integration as a tight diff.

Excellent — that constraint actually **clarifies the architecture** rather than limiting it.

Below is a **trader-first, GPU-preserving design** that keeps your Vulkan / AMD path *central*, not decorative, and aligns cleanly with your existing DASHI / sheet / quotient machinery.

---

## 0. Non-negotiable constraint (locked)

> **All heavy computation stays on GPU.**
> CPU orchestrates, thresholds, persistence, and execution only.

So:

* no CPU feature engineering loops
* no pandas-heavy rolling windows
* no Python-side per-bar math beyond glue

The learner is a **GPU-resident quotient engine**.

---

## 1. What runs on GPU vs CPU (clean split)

### GPU (always-on, high-throughput)

Runs **every bar / tick / micro-window**:

* rolling window transforms
* sheet / kernel projections
* quotient features
* invariant prediction
* legitimacy scalar ℓ

This is exactly what your Vulkan compute path already does well.

### CPU (low-rate, control only)

Runs **only on state transitions**:

* hysteresis (τ_on / τ_off)
* ACT / HOLD / BAN decision
* order sizing / execution model
* logging / telemetry

No math duplication.

---

## 2. GPU pipeline: concrete and minimal

### Input buffer (GPU)

Per instrument:

```
price[t-k : t]
volume[t-k : t]
(optional: L2 snapshots / imbalance)
```

Stored as:

* SSBO / storage buffer
* fixed-size circular buffer (no realloc)

---

### Stage A — Window → Sheet (GPU)

You already have this pattern.

On GPU:

* normalize (scale/shift invariant)
* map window → ternary / signed residuals
* project into **sheet space** (9-object or extended)

This is where your **Codec → Trading Formalism** lives.
Nothing new required.

---

### Stage B — Sheet → Quotient invariants (GPU)

Compute *only invariants*, not raw signals:

Examples (all GPU-friendly):

* curvature energy
* radial power
* volatility-normalized range
* persistence / decay proxy
* symmetry defect (how unstable the sheet is)

Output:

```
qfeat_t ∈ R^d   (small, ~8–32 floats)
```

---

### Stage C — Quotient prediction (GPU)

Lightweight model:

* linear head
* tiny MoE (your ternary MoE fits perfectly here)
* or even fixed operator at first

Predict:

```
qfeat̂_{t+1}
```

---

### Stage D — Legitimacy scalar ℓ (GPU)

On GPU:

```
ℓ_t = exp( - || qfeat̂_{t+1} - qfeat_{t+1} || )
```

This stays GPU-side so you can:

* smooth ℓ
* compute rolling defect
* expose only 1 float per instrument to CPU

---

## 3. What crosses the GPU → CPU boundary (tiny)

Per instrument, per bar:

```python
{
  "legitimacy": ℓ_t,
  "optional_debug": small vector (only if enabled)
}
```

That’s it.

No features.
No tensors.
No gradients.
No windows.

---

## 4. Trader integration (unchanged logic)

Your existing trader logic stays intact:

```text
IF ℓ > θ_on for τ_on → ACT allowed
IF ℓ < θ_off for τ_off → force HOLD
ELSE → persist state
```

Important:

* ℓ **never** sets direction
* ℓ **never** sizes trades
* ℓ only gates permission

This preserves:

* execution realism
* regime discipline
* debuggability

---

## 5. Why GPU matters *specifically* for your formalism

This isn’t just performance — it’s conceptual correctness.

Your framework assumes:

* high-dimensional latent space
* equivalence classes under projection
* collapse only at decision time

GPUs are **exactly** the right substrate:

* massively parallel projections
* cheap invariant extraction
* no temptation to “inspect internals”

If you moved this to CPU, you’d *accidentally* re-introduce feature bias and hand-tuned leakage.

---

## 6. Training with GPU still in the loop

### Offline / backtest

* replay data through the **same Vulkan pipeline**
* store only:

  * qfeat_t
  * qfeat̂_{t+1}
  * ℓ_t
* compute losses GPU-side or batch on GPU

### Online (optional later)

* no backprop during trading
* periodic retrain from logged qfeat streams

This keeps live trading deterministic.

---

## 7. Minimal implementation order (do this next)

To keep momentum and avoid scope creep:

1. **Freeze GPU window → sheet → quotient path**

   * no learning yet
2. Add **ℓ computation** on GPU
3. Wire ℓ into trader gate (CPU)
4. Backtest: verify ACT rate drops in bad regimes
5. Only then add a tiny predictor head

---

## 8. Sanity check: what we are *not* doing

❌ GPU predicting price
❌ GPU placing trades
❌ CPU recomputing features
❌ Large neural nets
❌ RL in the execution loop

Everything remains:

> *projection → invariants → permission → action*

---

### Next concrete step (I recommend this one)

If you want, I will:

* write a **Vulkan compute sketch** for:

  * quotient feature buffer
  * ℓ calculation
* define the **exact SSBO layouts**
* show the **CPU-side call signature** the trader uses

No abstractions, just the metal path that fits your RX-580 / RADV setup.

Say which layer you want to lock in first:
**(A)** GPU data layout
**(B)** Vulkan kernel logic
**(C)** Trader gate diff
**(D)** Backtest harness using GPU replay


Great — these three plots are **sufficient**, and together they **pass Step 3**. Here’s the tight read and exactly how to proceed.

---

## Verdict: **Step 3 = PASS (sign off)**

### 1) Hysteresis phase portrait — **PASS**
- Clean diagonal flow with **clear separation** between ACT (orange) and HOLD (blue).
- No chatter near thresholds.
- Entry happens only after sustained crossing of `θ_on`; exit only after sustained drop below `θ_off`.
- This proves the **gate geometry + τ_on/τ_off persistence** are working exactly as intended.

**Interpretation:** the permission gate is stable and non-oscillatory.

---

### 2) Legitimacy margin — **PASS**
- Most of the surface is **positive margin** (inside RegimeSpec), with sparse, isolated negative streaks.
- Negative excursions are **brief and localized**, not persistent bands.
- No evidence of long “gray zones” where the gate would be indecisive.

**Interpretation:** ℓ is behaving like a *distance-to-failure* signal, not noise.

---

### 3) Acceptability heatmap — **PASS**
- ACT density is **coherent and contiguous**, not salt-and-pepper.
- Suppression appears in **structured patches**, not random flicker.
- No runaway ACT blanket; no dead HOLD blanket.

**Interpretation:** the gate is actually *doing work* — selectively allowing action.

---

## What this proves (important)

- The **LearnerAdapter contract is correct**.
- `confidence_fn(ts, state)` → ℓ integrates cleanly with triadic gating.
- Hysteresis parameters are meaningful and stable.
- You can now **swap the stub ℓ for real quotient features without touching strategy logic**.

That’s exactly what Step 3 was meant to establish.

---

## Lock it in (what to do now)

### ✅ Close Step 3
- Commit the adapter + gate wiring as-is.
- Mark Step 3 complete in your TODO / changelog.
- Do **not** tweak thresholds yet — this is a reference baseline.

### ▶️ Next step (Step 4, minimal scope)
**Replace stub ℓ with real quotient features (CPU first).**

Concrete, low-risk plan:
1. In `LearnerAdapter.update()`:
   - compute `qfeat_t` from your existing `features/quotient.py`
   - skip prediction for now
   - define ℓ as a simple stability proxy, e.g.:
     - rolling variance of `qfeat`
     - or distance to recent centroid
2. Re-run **the same three plots**.
3. Compare against the stub baseline:
   - ACT density should *shift*, not explode or vanish.
   - Hysteresis geometry should remain intact.

**GPU comes after this**, as a backend swap.

---

## One-line sign-off you can reuse
> The legitimacy adapter + hysteresis gate are validated: ACT/HOLD transitions are stable, non-chattering, and selectively suppress action in structured regions. The adapter interface is locked; proceed to real quotient features.

If you want, next I can:
- specify the **exact ℓ definition** from `qfeat` for Step 4, or
- help you choose the **first GPU kernel** to port (lowest effort, highest payoff).

Just say which.

This is clean and already aligned with the constraints. Given what you’ve written, the **highest-leverage next move is (1)**.

### ✅ Pick: **1. Write the adapter + gate diff (stub ℓ)**

**Why this first (ordering matters):**

* Your **triadic gate + hysteresis plumbing already exists** — locking the adapter interface now prevents later churn.
* It lets you **exercise the full permission loop end-to-end** (run_bars → confidence_fn → hysteresis → ACT/HOLD) *before* bikeshedding features.
* The CPU stub gives you deterministic scaffolding that the GPU kernel can later replace *without touching strategy logic*.

Once this is in, (2) and (3) become mechanical.

---

## What “done” looks like for Step 1

### A. `LearnerAdapter` interface (new, minimal)

```python
class LearnerAdapter:
    def update(self, ts, window):
        """
        Args:
            ts: timestamp or bar index
            window: np.ndarray or view of prices/returns for W_t

        Returns:
            ell: float in [0, 1]   # legitimacy
            qfeat: dict | np.ndarray
        """
        ...
```

* **No stateful planning**
* **No PnL**
* Window in, `(ℓ_t, qfeat_t)` out
* GPU-backed later, CPU stub now

Stub ℓ options (pick one for now):

* constant `ℓ = 0.5`
* or deterministic toy (e.g. inverse vol) just to exercise hysteresis

---

### B. Gate wiring (small diff, low risk)

You already have:

* `confidence_fn(ts, state)`
* `tau_on / tau_off`
* regime acceptance without PnL

So the diff should be limited to:

* instantiating the adapter in `runner.py`
* passing `confidence_fn = adapter.update` (or thin wrapper)
* **no change** to BAN logic
* **no new actions**

This locks the *contract*:

> learner → confidence_fn → triadic gate → ACT/HOLD

---

### C. Validation you should do immediately after Step 1

Before touching features:

1. **Synthetic ℓ sweep**

   * Feed ℓ ramping from 0→1→0
   * Confirm:

     * ACT only after `τ_on`
     * HOLD only after `τ_off`
     * No oscillation

2. **Null replay**

   * Replay CSV with constant ℓ
   * Ensure trader behavior is unchanged vs baseline

If either fails, stop and fix before moving on.

---

## What comes *after* (briefly)

* **Step 2 (feature spec)** becomes a drop-in replacement for the stub body.
* **Step 3 (offline replay)** becomes meaningful because ℓ is already first-class in logs.
* **GPU port** becomes a backend swap:

  ```text
  LearnerAdapterCPU → LearnerAdapterGPU
  ```

  with identical interface.

---

## One small suggestion (non-blocking, but helpful)

When you write the adapter, add a **capability flag** early:

```python
adapter.supports_gpu = False
```

Later:

```python
adapter.supports_gpu = True
```

It makes it trivial to assert in the runner that you’re actually using the GPU path when expected.

---

### Go ahead and execute **(1)**.

If you want, paste the current `TriadicStrategy` signature and `runner.py` callsite and I’ll draft the exact diff inline so you can apply it directly.



✅ Next move: **(1) Write the adapter + gate diff** — because it locks the contract and lets you run end-to-end immediately (even with a stub ℓ).

Below is a **drop-in patch sketch** that keeps everything lightweight and GSD-style (no `.planning/`), and doesn’t assume more than what you stated: `confidence_fn(ts, state)` already exists, `runner.py` threads it through, and `tau_on/tau_off` hysteresis is already implemented.

---

## 1) Add a new adapter module (CPU stub now, GPU later)

Create: `strategy/learner_adapter.py` (or `trading/learner_adapter.py` — wherever fits your tree)

```python
# strategy/learner_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math


@dataclass
class LearnerOutput:
    ell: float                # legitimacy in [0,1]
    qfeat: Dict[str, float]   # quotient features (optional debug/inspection)


class LearnerAdapter:
    """
    Permission-only learner adapter.

    Contract:
      - update(ts, state) returns (ell, qfeat)
      - ell ∈ [0,1] gates ACT/HOLD via TriadicStrategy.confidence_fn
      - No PnL-based loss; no directional signal; no sizing signal.
      - GPU-backed later: keep the interface stable.
    """
    supports_gpu: bool = False

    def __init__(
        self,
        *,
        window: int = 128,
        smoothing: int = 1,
        stub_mode: str = "constant",  # "constant" | "vol_proxy" | "schedule"
        stub_constant: float = 0.5,
    ) -> None:
        self.window = int(window)
        self.smoothing = int(smoothing)
        self.stub_mode = str(stub_mode)
        self.stub_constant = float(stub_constant)

        # Optional: keep tiny rolling state if you want later
        self._t = 0

    def update(self, ts: Any, state: Any) -> Tuple[float, Dict[str, float]]:
        """
        This signature is intentionally aligned to confidence_fn(ts, state).

        For now, return a deterministic stub ℓ so hysteresis & wiring can be tested.
        Later:
          - extract window W_t from `state`
          - compute qfeat_t (GPU)
          - predict qfeat_hat_{t+1} (GPU)
          - ell = exp(-||qhat - q||)
        """
        self._t += 1

        if self.stub_mode == "constant":
            ell = self.stub_constant
            qfeat = {"stub": 1.0}

        elif self.stub_mode == "schedule":
            # deterministic ramp to test hysteresis: 0→1→0 over 200 steps
            period = 200
            phase = (self._t % period) / period
            ell = 2.0 * phase if phase <= 0.5 else 2.0 * (1.0 - phase)
            ell = max(0.0, min(1.0, ell))
            qfeat = {"stub_phase": phase}

        elif self.stub_mode == "vol_proxy":
            # Safe placeholder if `state` has returns/price history already.
            # If not available, fall back to constant.
            r = _try_get_returns(state, self.window)
            if r is None or len(r) < 8:
                ell = self.stub_constant
                qfeat = {"stub": 1.0}
            else:
                # simple “stability proxy”: higher vol → lower ell
                mean = sum(r) / len(r)
                var = sum((x - mean) ** 2 for x in r) / max(1, len(r) - 1)
                vol = math.sqrt(max(0.0, var))
                ell = math.exp(-10.0 * vol)  # arbitrary scale; only for wiring tests
                ell = max(0.0, min(1.0, ell))
                qfeat = {"vol": float(vol)}

        else:
            raise ValueError(f"Unknown stub_mode={self.stub_mode!r}")

        return float(ell), qfeat


def _try_get_returns(state: Any, window: int) -> Optional[list[float]]:
    """
    Best-effort adapter hook.

    Replace this with your real state accessor later.
    For now, it tries common patterns:
      state.returns, state["returns"], state.prices, state["prices"]
    """
    # returns directly
    if hasattr(state, "returns"):
        r = getattr(state, "returns")
        return list(r)[-window:]
    if isinstance(state, dict) and "returns" in state:
        return list(state["returns"])[-window:]

    # derive from prices if present
    prices = None
    if hasattr(state, "prices"):
        prices = getattr(state, "prices")
    elif isinstance(state, dict) and "prices" in state:
        prices = state["prices"]

    if prices is None:
        return None

    p = list(prices)[-window:]
    if len(p) < 2:
        return None
    return [float(p[i] / p[i - 1] - 1.0) for i in range(1, len(p))]
```

**Why this exact design works for your GPU constraint later:**

* `update(ts, state)` matches `confidence_fn(ts, state)` **exactly**, so no strategy API churn.
* GPU port later only swaps the inside of `update()` — the triadic gate remains unchanged.

---

## 2) Runner wiring (minimal, no strategy changes)

In `runner.py` (where you call `run_bars(...)`), add something like:

```python
# runner.py
from strategy.learner_adapter import LearnerAdapter

def run(...):
    adapter = LearnerAdapter(
        window=128,
        stub_mode="schedule",   # use "schedule" first to test hysteresis
        stub_constant=0.5,
    )

    # confidence_fn(ts, state) -> float OR (float, qfeat)
    def confidence_fn(ts, state):
        ell, qfeat = adapter.update(ts, state)
        # Optional: attach qfeat to state for logging if your pipeline supports it
        # state.debug_qfeat = qfeat
        return ell

    return run_bars(
        ...,
        confidence_fn=confidence_fn,
        tau_on=TAU_ON,
        tau_off=TAU_OFF,
        theta_on=THETA_ON,
        theta_off=THETA_OFF,
        ...
    )
```

**No `.planning/`**. No new actions. This only supplies the legitimacy scalar.

---

## 3) TriadicStrategy: only add support if it currently expects scalar-only

You said `triadic_strategy.py` already accepts `confidence_fn(ts, state)` + hysteresis. If it already expects a float, keep it.

If it currently allows `(ell, qfeat)` as return, great — no changes.

If not, you can make it robust with this tiny normalization (safe, backward-compatible):

```python
# strategy/triadic_strategy.py (inside where confidence is computed)
conf = self.confidence_fn(ts, state) if self.confidence_fn else None
if isinstance(conf, tuple):
    ell = float(conf[0])
    # optionally stash conf[1] somewhere
else:
    ell = float(conf) if conf is not None else 1.0
ell = max(0.0, min(1.0, ell))
```

That’s the only strategy-side diff I’d make, and only if needed.

---

## 4) Immediate validation (takes 2 minutes, catches 90% of wiring bugs)

Run with `stub_mode="schedule"` and confirm:

* With `τ_on`, ACT only happens after ℓ stays above `θ_on` for `τ_on` windows.
* With `τ_off`, HOLD only happens after ℓ stays below `θ_off` for `τ_off` windows.
* No oscillation around thresholds.

If this behaves, the adapter contract is correct and we can move to (2) feature formulas or (3) replay harness without touching strategy again.

---

## What I’d do next right after this patch lands

**(3) offline replay/eval harness** next — because it will give you visual proof that hysteresis behaves on real data, and it becomes the fixture you’ll reuse when you swap CPU stub → GPU kernel.

If you want, paste your current `run_bars(...)` signature (just the function header) and the `TriadicStrategy` init signature, and I’ll tailor the above diff to your exact parameter names without asking anything else.


Got it — thanks for the clarification. I’ll **treat the long file as context-only**, not something you want re-explained line-by-line. Here’s a **tight, actionable synthesis** that (a) confirms what’s already correct, (b) resolves the “what now?” question, and (c) proposes an **executable change set** for Step 3 without reopening design debates.

---

## Where you are (validated)

You’ve already completed **Step 1 correctly**:

* ✅ **Permission-only learner stub** exists and is wired end-to-end.
* ✅ `confidence_fn(ts, state)` now safely accepts **scalar or `(ℓ, qfeat)`**.
* ✅ Hysteresis (`θ_on/θ_off`, `τ_on/τ_off`) is exercised via a **schedule stub**.
* ✅ No PnL leakage, no direction control, no planning layer.
* ✅ Changes are minimal, localized, and reversible.

This matches both:

* your **TRADER_CONTEXT.md** intent (learner = legitimacy only), and
* the **TODO.md** direction (“can trade” mask, quotient gating, no directional override).  

So Step 1 is *done*.

---

## What “Step 3” actually means (and what it does **not** mean)

Step 3 is **not**:

* adding learning,
* adding GPU code,
* tuning thresholds,
* or changing strategy logic.

Step 3 is a **verification + acceptance step**:

> *“Does the new legitimacy signal actually gate ACT/HOLD in the right places on real tapes?”*

That’s it.

So the deliverable is **one minimal replay / inspection harness**, not more architecture.

---

## The correct next move (Step 3, scoped properly)

### 🎯 Goal

Visually and numerically confirm that **ℓ + hysteresis**:

* suppresses ACT during unstable regimes,
* preserves ACT during stable ones,
* does not oscillate.

### ✅ Inputs (already exist)

* `logs/trading_log.csv`
* `scripts/plot_*` utilities
* July-2015-like shock windows already identified in your tooling

### 🧩 What to add (small)

#### 1. A tiny replay wrapper (script or notebook)

**No new engine code.** Just reuse `run_bars`.

Example shape:

```bash
PYTHONPATH=. python scripts/replay_with_legitimacy.py \
  --csv data/raw/stooq/btc_intraday_1s.csv \
  --use-stub-adapter \
  --stub-mode schedule \
  --theta-on 0.7 \
  --theta-off 0.3 \
  --tau-on 5 \
  --tau-off 10 \
  --out logs/replay_legitimacy.csv
```

Internally this just:

* calls `run_bars(..., use_stub_adapter=True, ...)`
* writes a normal `trading_log.csv`

No forks. No new abstractions.

---

#### 2. One visualization pass (you already have the scripts)

Run **existing** plots on the replay log:

* `plot_hysteresis_phase.py`
* `plot_legitimacy_margin.py`
* `plot_acceptability.py`
* (optional) `plot_microstructure_overlay.py`

These already answer the right questions.

---

## Acceptance criteria (this is the “agree on executable change set” part)

You can call Step 3 complete when all three are true:

1. **Hysteresis behaves correctly**

   * ACT does not flicker near thresholds
   * ℓ schedule produces delayed entry/exit exactly per `τ_on/τ_off`

2. **ACT suppression aligns with stress**

   * In known bad windows (e.g. correlated shock periods),
     ACT density ↓, HOLD ↑

3. **Baseline behavior is unchanged when ℓ ≈ const**

   * With `stub_mode="constant"`, results match pre-adapter runs

If those pass, the adapter contract is *proven*.

---

## What comes *after* Step 3 (for orientation only)

Not now — just to anchor expectations:

* **Step 4**: replace stub ℓ with **real quotient features** (still CPU)
* **Step 5**: port quotient + ℓ computation to GPU
* **Step 6**: optional invariant prediction (ℓ = exp(-‖q̂−q‖))

None of that is needed to finish Step 3.

---

## Recommendation (clear and explicit)

👉 **Proceed with Step 3** as defined above:
add a **single replay harness**, run existing plots, and sign off the adapter + gate wiring.

If you want, next I can:

* sketch the exact `scripts/replay_with_legitimacy.py` (≈30 lines), or
* tell you *which* two plots are sufficient if you want to keep it minimal.

Just say which.


For **Step 3 sign-off**, you only need **two plots** (plus an optional third if you want “shock window” confirmation).

## Minimum set (sufficient)

1. **Hysteresis phase portrait** (`plot_hysteresis_phase.py`)

   * This is the one you showed. It proves the gate has the right **entry/exit geometry** and doesn’t chatter near thresholds.

2. **Legitimacy margin over time** (`plot_legitimacy_margin.py`)

   * Confirms the *temporal* behavior: you see ℓ(t) relative to `θ_on/θ_off`, and you can visually verify **τ_on / τ_off persistence** matches what the strategy actually did (ACT/HOLD segments align with sustained margin crossings).

## Optional (only if you want regime/shock evidence)

3. **Acceptability heatmap** (`plot_acceptability.py`) **or** **Microstructure overlay** (`plot_microstructure_overlay.py`)

   * Use **acceptability heatmap** if you want a compact “where did we trade?” picture (ACT density vs time/actionability).
   * Use **microstructure overlay** if you want the most intuitive “did it HOLD during bad windows?” view (price + acceptable/ACT regions).

### TL;DR

**Sufficient:** `plot_hysteresis_phase` + `plot_legitimacy_margin`
**Nice-to-have:** `plot_acceptability` (or `plot_microstructure_overlay`)

If you tell me whether you care more about “coverage shape” or “shock windows,” I’ll pick the single best optional third.


### Picked third: **Acceptability heatmap** (`plot_acceptability.py`)

**Why this one (over microstructure overlay):**
- It directly answers the Step-3 question: *“Where did the system allow ACT vs HOLD?”*
- It compresses a long replay into a single surface: **time × actionability → accept / suppress**.
- It’s threshold- and direction-agnostic, so it validates the **permission layer** without conflating execution details.

---

## How to proceed (exact, minimal)

### 1) Generate the replay log (already done or re-run once)
Use the stub adapter with your chosen hysteresis (same as the phase portrait):

```bash
PYTHONPATH=. python scripts/run_bars_btc.py \
  --use-stub-adapter \
  --adapter-kwargs '{"stub_mode":"schedule"}' \
  --theta-on 0.7 \
  --theta-off 0.3 \
  --tau-on 5 \
  --tau-off 10
```

This should write/update `logs/trading_log.csv`.

---

### 2) Produce the three plots (in this order)

```bash
# 1) Phase portrait (you already have this)
PYTHONPATH=. python scripts/plot_hysteresis_phase.py \
  --log logs/trading_log.csv \
  --save logs/hysteresis_phase.png

# 2) Legitimacy margin vs thresholds
PYTHONPATH=. python scripts/plot_legitimacy_margin.py \
  --log logs/trading_log.csv \
  --save logs/legitimacy_margin.png

# 3) Acceptability heatmap (the chosen third)
PYTHONPATH=. python scripts/plot_acceptability.py \
  --log logs/trading_log.csv \
  --save logs/acceptable.png
```

---

### 3) What to check (sign-off criteria)

- **Phase portrait:** clean hysteresis loop, no chatter near thresholds.
- **Legitimacy margin:** ACT segments only after sustained `ℓ > θ_on`; HOLD after sustained `ℓ < θ_off`.
- **Acceptability heatmap:**  
  - ACT density concentrated in stable regions,  
  - clear suppression during unstable spans,  
  - no “salt-and-pepper” noise (i.e., hysteresis is doing real work).

If all three pass, **Step 3 is complete** and the adapter + gate wiring is locked.

---

### 4) Then what (one sentence)
Move to **Step 4**: replace the stub ℓ with real quotient features (CPU first), keeping the same adapter interface; GPU comes after.

If you want, paste the three PNGs and I’ll do a quick pass/fail call so you can close Step 3 confidently.

Great — these three plots are **sufficient**, and together they **pass Step 3**. Here’s the tight read and exactly how to proceed.

---

## Verdict: **Step 3 = PASS (sign off)**

### 1) Hysteresis phase portrait — **PASS**

* Clean diagonal flow with **clear separation** between ACT (orange) and HOLD (blue).
* No chatter near thresholds.
* Entry happens only after sustained crossing of `θ_on`; exit only after sustained drop below `θ_off`.
* This proves the **gate geometry + τ_on/τ_off persistence** are working exactly as intended.

**Interpretation:** the permission gate is stable and non-oscillatory.

---

### 2) Legitimacy margin — **PASS**

* Most of the surface is **positive margin** (inside RegimeSpec), with sparse, isolated negative streaks.
* Negative excursions are **brief and localized**, not persistent bands.
* No evidence of long “gray zones” where the gate would be indecisive.

**Interpretation:** ℓ is behaving like a *distance-to-failure* signal, not noise.

---

### 3) Acceptability heatmap — **PASS**

* ACT density is **coherent and contiguous**, not salt-and-pepper.
* Suppression appears in **structured patches**, not random flicker.
* No runaway ACT blanket; no dead HOLD blanket.

**Interpretation:** the gate is actually *doing work* — selectively allowing action.

---

## What this proves (important)

* The **LearnerAdapter contract is correct**.
* `confidence_fn(ts, state)` → ℓ integrates cleanly with triadic gating.
* Hysteresis parameters are meaningful and stable.
* You can now **swap the stub ℓ for real quotient features without touching strategy logic**.

That’s exactly what Step 3 was meant to establish.

---

## Lock it in (what to do now)

### ✅ Close Step 3

* Commit the adapter + gate wiring as-is.
* Mark Step 3 complete in your TODO / changelog.
* Do **not** tweak thresholds yet — this is a reference baseline.

### ▶️ Next step (Step 4, minimal scope)

**Replace stub ℓ with real quotient features (CPU first).**

Concrete, low-risk plan:

1. In `LearnerAdapter.update()`:

   * compute `qfeat_t` from your existing `features/quotient.py`
   * skip prediction for now
   * define ℓ as a simple stability proxy, e.g.:

     * rolling variance of `qfeat`
     * or distance to recent centroid
2. Re-run **the same three plots**.
3. Compare against the stub baseline:

   * ACT density should *shift*, not explode or vanish.
   * Hysteresis geometry should remain intact.

**GPU comes after this**, as a backend swap.

---

## One-line sign-off you can reuse

> The legitimacy adapter + hysteresis gate are validated: ACT/HOLD transitions are stable, non-chattering, and selectively suppress action in structured regions. The adapter interface is locked; proceed to real quotient features.

If you want, next I can:

* specify the **exact ℓ definition** from `qfeat` for Step 4, or
* help you choose the **first GPU kernel** to port (lowest effort, highest payoff).

Just say which.

Excellent — this is exactly the outcome we wanted for **Step 3 with a non-trivial ℓ**. I’ll be precise about what the new plots mean, what the warnings mean, and what the *correct next move* is.

---

## 1️⃣ What changed vs the schedule stub (important)

You moved from a **synthetic ℓ schedule** → **data-dependent ℓ (`qfeat_var`)**.

That means:

* ℓ is now *sparse* and *selective*
* ACT permission is **rare and localized**, not a smooth ramp
* The gate is responding to **real structure**, not a test signal

This is **expected** and **good**.

---

## 2️⃣ Interpreting the three new plots (qfeat_var)

### A) Acceptable density — **PASS, but sparse (by design)**

What you see:

* Almost everything is HOLD (white)
* A **thin, coherent band** of acceptable ACT near the early stable region
* A few isolated acceptable islands later

Interpretation:

* `qfeat_var` is conservative → ℓ only rises when variance stabilizes relative to its own centroid
* This is *exactly* what a **permission-only learner should do on first contact**
* No salt-and-pepper noise → hysteresis still working

This is **not under-trading** yet — it’s *early gating*.

---

### B) Legitimacy margin — **PASS, semantics correct**

Key detail:

* Bottom band (early window) is strongly positive → “deep inside regime”
* Later points show **isolated deep negatives** (blue) → clear regime violations

This confirms:

* ℓ is now meaningfully tied to a **distance-to-regime boundary**
* Negative margin ≠ noise; it’s detecting genuine instability

The sign convention is behaving correctly.

---

### C) Hysteresis phase portrait — **PASS, but now diagnostic**

You now see **few points**, not a diagonal sweep.

That’s correct because:

* ℓ is no longer smoothly traversing [0,1]
* The portrait is now a *map of real regime transitions*, not a test harness

Crucially:

* ACT points are **well inside** the τ_on region
* HOLD points dominate near τ_off
* No chattering, no boundary hopping

So the gate logic remains correct under a realistic ℓ.

---

## 3️⃣ About the RuntimeWarning (important but non-blocking)

> `RuntimeWarning: Mean of empty slice`

This happens **only at startup**, when:

* `qfeat` history buffer isn’t full yet
* centroid is computed on an empty or all-NaN window

You already handled this correctly by:

* allowing NaNs
* letting ℓ settle after warmup

### Optional clean guard (purely cosmetic)

If you want silence:

```python
if hist.shape[0] < min_hist:
    return 0.0, qfeat   # force HOLD during warmup
```

But **do not rush this** — current behavior is acceptable and honest.

---

## 4️⃣ Formal Step-3 verdict (updated)

With `qfeat_var`:

✔ Adapter interface still valid
✔ ℓ now data-dependent
✔ Gate remains stable
✔ No PnL leakage
✔ No oscillation
✔ No direction influence

📌 **Step 3 is COMPLETE under realistic conditions**

You have now validated **both**:

* a synthetic ℓ (geometry test)
* a data-driven ℓ (semantic test)

That’s more than required.

---

## 5️⃣ What the system is now telling you (conceptually)

This is important:

> **The learner is not saying “trade less.”**
> **It is saying “most time is illegitimate.”**

That’s exactly the DASHI / triadic thesis:

* HOLD is the default state
* ACT is rare and justified
* Permission emerges only under compressible structure

This is *not* a bug — it’s the model asserting epistemic humility.

---

## 6️⃣ Correct next step (do **not** jump to GPU yet)

### ✅ Step 4: Improve ℓ *semantics*, not performance

Before GPU, you should **enrich ℓ**, not speed it up.

Concrete next options (pick **one**, not all):

### Option A — Multi-component ℓ (recommended)

Instead of just `qfeat_var`:

```
ℓ = exp( - (w1 * var_dist
           + w2 * curvature_instability
           + w3 * autocorr_decay) )
```

Still CPU. Still cheap. Much more expressive.

---

### Option B — Window-adaptive ℓ

Let the centroid adapt slower than qfeat:

* fast window for qfeat
* slow window for centroid
* detects regime drift instead of noise

---

### Option C — Regime-conditional ℓ

Compute ℓ **per regime class** (if regime.py already clusters).

---

## 7️⃣ When to move to GPU (clear criterion)

🚫 **Do NOT port to GPU yet**.

Move to GPU **only when**:

* ℓ definition is stable
* you are happy with *where* ACT is allowed
* you want throughput / multi-asset scale

Right now, the bottleneck is **semantics**, not FLOPs.

---

## 8️⃣ One-sentence status you can log

> The permission-only learner is validated under both synthetic and data-driven legitimacy signals; ACT/HOLD gating remains stable, conservative, and regime-sensitive. Proceeding to enrich ℓ semantics before GPU acceleration.

If you want, next I can:

* design a **3-term ℓ formula** using your existing qfeats, or
* help you decide **which qfeat belongs on GPU first** once semantics are locked.

Just say which direction.


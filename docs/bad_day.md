## Bad-day flag: definition, signals, and policy

This distills the “bad day” discussion into something you can wire and test.

### What “bad day” means here
- It is **regime admissibility**, not return direction. A bad day is when the market is structurally adversarial to directional strategies.
- Output: either `bad_day ∈ {0,1}` or `p_bad ∈ [0,1]`.
  - `bad_day = 1` → do not deploy directional capital.
  - Policy example: `p_bad > 0.7 → BAN`, `0.3 < p_bad ≤ 0.7 → HOLD`, `p_bad ≤ 0.3 → ACT`.

### Posture vs direction (keep them separate)
- You already have multiple action types; they’re just conflated. Think in two axes:
  - **Posture** ∈ {ACT, HOLD, BAN} (capital permission).
  - **Direction** ∈ {−1, 0, +1} (price view).
- The bad-day flag lives on the **posture** axis.

### Minimal structural stress scalar
- You already log the ingredients: coarse-scale change rates, fatigue, anchor conflict, flip rate.
- Define a stress score (weights can be hand-tuned):
  ```
  S(t) = w1·Δ_CA_scale4
       + w2·Δ_CA_scale8
       + w3·fatigue
       + w4·anchor_conflict
       + w5·flip_rate
  ```
- Squash to a probability (soft boundary between “tension” and “shutdown”):
  ```
  p_bad(t) = σ(S(t) - θ)
  ```
  or use `p_bad = clamp(S / S_max, 0, 1)` if you dislike sigmoids. θ/S_max is a calibration cut, not model training.

### Testing it honestly
- **Structural test first:** mark bars/days where `p_bad > threshold`; verify realized vol, flip rate, spread/impact, and max adverse excursions spike.
- **Economic test second:** remove directional trades on `bad_day` bars; compare drawdown, tail loss, and recovery time. PnL uplift is *not* the primary metric; regime separation is.

### How it sits in the M-hierarchy
- M₃: local signal (trend/reversal)
- M₆: competing signals → tension metric
- M₉: admissibility gate → “bad game”
- `p_bad` is the M₉ gate over M₆ tension. It does not replace triadic direction; it decides if the game is playable.

### Next wiring steps (minimal)
- Log `p_bad` and `bad_flag` (already in place) and condition forward returns on them (`ret_bad`, `ret_good`).
- Optionally add a “pathology posture” (hedge/inverse) so BAN/HOLD are not the only responses to structural stress.
- Use coarse CA instability as a pre-gate to avoid paying execution costs just to learn that structure is bad.

### Relating `bad_day` to real-world news (correlation workflow)
- **Anchor by time, not narrative:** `bad_flag` is bar-aligned; keep UTC timestamps. When you see a spike, search headlines/time-stamped feeds in that window.
- **Build a quick table:** from `logs/trading_log*.csv`, pull rows where `bad_flag==1` (or `p_bad>0.7`) and export `(t, price, p_bad, bad_flag, ret_fwd)`. That gives you event markers.
- **Manual/newsfeed overlay:** overlay those timestamps on a headline feed (Bloomberg/WSJ/API of choice). You’re looking for clustering of major events near elevated `p_bad`.
- **Test for “major news days” bias:** bucket days by `bad_rate` (share of bars with `bad_flag`) and check if high-`bad_rate` days coincide with known macro/news shocks.
- **Remember the semantics:** `bad_day` detects structural stress; news is often the visible cause/effect. A lack of headlines can still yield `bad_flag` if the tape is forced (e.g., liquidations).
- **Daily rollup for calendar heatmaps:** aggregate intraday `p_bad` into per-day stats (mean, max, run-length). Example: `bad_score_d = α·mean(p_bad_d) + β·max(p_bad_d) + γ·runlen(p_bad>0.7)`. Plot as a calendar; annotate the most negative days with headlines to sanity-check alignment.

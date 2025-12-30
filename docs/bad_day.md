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
- **Helper script:** `PYTHONPATH=. python trading/scripts/rollup_bad_days.py --log logs/trading_log.csv --out logs/bad_days.csv --top 10` produces the per-day badness table (mean/max/run-length/bad_score) for quick news overlays.
- **Fetch headlines/events for a slice:** `PYTHONPATH=. python trading/scripts/news_slice.py --provider newsapi --start 2025-03-01T09:00:00Z --end 2025-03-01T12:00:00Z --out logs/news_slice.csv` (requires `NEWSAPI_KEY`). For event-level intensity instead of headlines, use `--provider gdelt` (no key).
- **Automatic windows → events:** `PYTHONPATH=. python trading/scripts/emit_news_windows.py --log logs/trading_log.csv --out-dir logs/news_events` detects bad windows (p_bad≥0.7 & bad_flag) and fetches GDELT events per window (no API key by default).
- **Coverage/404 behavior:** GDELT effectively covers ~2015-06 onward; future dates are skipped. 404 responses are treated as “no events” and logged once. Windows/days are capped by trigger count so you don’t spam the endpoint on deep histories.
- **Synthetic validation (no news needed):** `PYTHONPATH=. python trading/scripts/score_bad_windows.py --log logs/trading_log.csv --out logs/bad_windows.csv` adds synthetic bad flags (|return|>k·σ or drawdown slope) and ranks top-N windows by summed `p_bad` over a sliding window, so you can verify p_bad/bad_flag against price/vol shocks directly.

### Regime windows (inter-arrival of bad)
- Define bad onsets: τ_k where bad flips from {!–1} → –1. Define windows W_k = [τ_k, τ_{k+1}). Measure:
  - window length distribution
  - PnL per window
  - volatility/drawdown concentration near window start
- This turns the problem into hazard/survival analysis: “how long until the next bad regime?” rather than bar-by-bar prediction.

### Tri-state p_bad (symmetric semantics)
- Use p_bad ∈ {−1,0,+1} (or mapped from [0,1] with thresholds):
  - +1: coherent/exploitable
  - 0: undecidable/entropy
  - −1: incoherent/adversarial
- Policy: ACT if p_bad>θ₊; BAN if p_bad<θ₋; HOLD otherwise. This keeps three postures distinct from directional view.

### “Legal” BAN monetisation (no cheating on direction)
- Do not let BAN imply “go short directionally.” BAN monetises structure, not price:
  - Vol-only exposure (long vol proxies / straddle-like behavior).
  - Capital preservation (flat when others bleed).
  - Fee/slippage avoidance in churny regimes.
- Directional inversion is a later, separate strategy gated by the same posture logic.

### Posture semantics (disentangle axes)
- Three axes, keep them distinct:
  1) **Regime classification** (world type / admissibility).
  2) **Posture** g∈{−1,0,+1} = {BAN, HOLD, ACT} (permission to express belief).
  3) **Execution / PnL** (realized outcome).
- g values are epistemic/control states, not returns:
  - **+1 (ACT)**: “structure exists I trust” → directional expression allowed; mean >0 if edge, high variance.
  - **0 (HOLD)**: “no reliable structure” → defer belief; ≈0 mean, low variance.
  - **−1 (BAN)**: “structure is adversarial” → forbid directional belief; mean can be ≤0 or >0, variance very low; monetize second-order effects only.
- BAN ≠ HOLD: HOLD defers belief; BAN forbids it and can trigger active defense.

### BAN as a hazard operator (minimal “legal” rule)
- Definition: BAN means “do not express first-order price belief.” Allowed monetisation is second-order:
  - Exposure annihilation + cooldown + regime-boundary alignment (forced flatten, no churn).
  - Symmetric/convex-only payoffs (vol/convexity capture, straddle-like logic, absolute-return caps).
  - Inter-regime capital reallocation (lower next notional, delay redeploy, reset leverage).
- Illegal during BAN: directional price bets, sign inversion as a proxy for “short the world,” hindsight hedging.

### Window-level evaluation (litmus test)
- Litmus: is mean return during BAN windows worse than mean return during ACT windows?
- Implementation sketch on a per-bar log with `ret` and posture `g ∈ {−1,0,+1}`:
  ```python
  import pandas as pd
  log = pd.read_csv("logs/trading_log.csv")
  # forward or contemporaneous return; use whatever the log encodes
  log["ret"] = log["pnl"].diff()  # example; replace with your bar return field
  stats = (
      log.assign(
          posture=log["g"],  # or map from ban/hold/action columns
      )
      .groupby("posture")["ret"]
      .agg(mean="mean", std="std", count="count")
  )
  print(stats.loc[1], "ACT"); print(stats.loc[0], "HOLD"); print(stats.loc[-1], "BAN")
  ```
- Windowed version (run-level hazard view):
  ```python
  def windows(df):
      df = df.reset_index(drop=True)
      starts = df.index[(df["g"].shift(1) != df["g"]) | (df.index == 0)]
      for i, s in enumerate(starts):
          e = starts[i+1] if i+1 < len(starts) else len(df)
          yield df.loc[s:e-1], df.loc[s, "g"]

  rows = []
  for w, g in windows(log):
      rows.append({"posture": g, "ret_sum": w["ret"].sum(), "ret_mean": w["ret"].mean(), "n": len(w)})
  win = pd.DataFrame(rows)
  print(win.groupby("posture")[["ret_sum","ret_mean","n"]].agg(["mean","std","count"]))
  ```
- Interpretation: BAN windows should exhibit lower (or negative) mean and lower variance vs ACT. If BAN mean ≥ ACT mean, the classifier is likely noise.

### Suggested next experiment (minimal)
- Attach a trivial ACT strategy (e.g., single-parameter trend-follow) and compare:
  - strategy alone vs strategy+CA gate (drawdown, Sharpe, turnover).
- Rank bad windows by severity (p_bad sum × trigger count) and align with synthetic bad labels before relying on news.

### Next implementation options
- Formalize BAN as a hazard operator in code: explicit permission mask + cooldown/rate limit on re-entry after bad onsets; make BAN sovereign over HOLD/ACT.
- Wire posture-return audits: use `trading/scripts/posture_returns.py` to print bar/window stats with cumulative sums per posture for quick integrity checks.
- Add a minimal “legal BAN” execution stub (convex-only): symmetric payoffs/vol capture under BAN, no directional expression; gated by the same permission mask.

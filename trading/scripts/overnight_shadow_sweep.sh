#!/usr/bin/env bash
set -euo pipefail

# Overnight sweep for futures-shadow policy debugging.
# Goal: identify whether the remaining blocker is penalty geometry / score structure
# (single-score path) or the need for a two-stage policy (magnitude gate then direction).
#
# Outputs:
# - logs/shadow/trading_log_overnight_<TS>_<TAG>_<symbol>.csv
# - logs/shadow/shadow_signal_report_<TS>_<TAG>_<symbol>.md
# - logs/shadow/shadow_signal_plots_<TS>_<TAG>_<symbol>.png

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR/trading"

TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Increase this if you want more stable statistics. If the tape is shorter, it just stops early.
MAX_STEPS="${MAX_STEPS:-50000}"

# Adaptive gating config (keep constant across runs).
TARGET_ACTION_RATE="${TARGET_ACTION_RATE:-0.10}"
MIN_HISTORY="${MIN_HISTORY:-50}"
HISTORY_SIZE="${HISTORY_SIZE:-400}"
PREFIT_MAX_VALUES="${PREFIT_MAX_VALUES:-2000}"
PREFIT_FAMILY="${PREFIT_FAMILY:-overnight_${TS}}"

LABEL_THRESHOLD="${LABEL_THRESHOLD:-0.05}"

# Entropy attenuation (smooth).
ENT_MODE="${ENT_MODE:-logistic}"
ENT_CENTER="${ENT_CENTER:-0.955}"
ENT_TAU="${ENT_TAU:-0.01}"

PY="../venv/bin/python"
export PYTHONPATH="."

run_one() {
  local symbol_include="$1"
  local tag="$2"
  shift 2

  local log_prefix="../logs/shadow/trading_log_overnight_${TS}_${tag}"
  local log_csv="../logs/shadow/trading_log_overnight_${TS}_${tag}_${symbol_include%.csv}.csv"

  # Run (SPY/BTC only via --all-include filter).
  $PY run_trader.py \
    --all --raw-root ../data/raw/stooq --all-include "$symbol_include" \
    --max-steps "$MAX_STEPS" --inter-run-sleep 0 \
    --shadow-futures \
    --shadow-kernel-mode shrinkage \
    --shadow-kernel-log-dir ../logs/shadow \
    --shadow-kernel-label-mode fixed --shadow-kernel-label-threshold "$LABEL_THRESHOLD" \
    --shadow-score-mode logistic --shadow-score-scale 2.0 \
    --shadow-gating-mode lex \
    --shadow-entropy-gate-mode "$ENT_MODE" \
    --shadow-entropy-gate-center "$ENT_CENTER" \
    --shadow-entropy-gate-tau "$ENT_TAU" \
    --shadow-score-threshold-mode adaptive_quantile \
    --shadow-target-action-rate "$TARGET_ACTION_RATE" \
    --shadow-score-threshold-min-history "$MIN_HISTORY" \
    --shadow-score-threshold-history-size "$HISTORY_SIZE" \
    --shadow-score-threshold-prefit \
    --shadow-score-threshold-prefit-max-values "$PREFIT_MAX_VALUES" \
    --shadow-score-threshold-prefit-family "$PREFIT_FAMILY" \
    --log-prefix "$log_prefix" \
    --log-level quiet --no-geometry-plots --no-tower-log \
    "$@"

  # Analyze.
  $PY scripts/analyze_shadow_signals.py \
    --input "${tag}=../logs/shadow/trading_log_overnight_${TS}_${tag}_${symbol_include%.csv}.csv" \
    --report "../logs/shadow/shadow_signal_report_${TS}_${tag}_${symbol_include%.csv}.md" \
    --plot "../logs/shadow/shadow_signal_plots_${TS}_${tag}_${symbol_include%.csv}.png"

  echo "[ok] ${symbol_include} ${tag}"
}

echo "[sweep] TS=$TS MAX_STEPS=$MAX_STEPS target=$TARGET_ACTION_RATE family=$PREFIT_FAMILY"

# SPY: core A/B matrix.
run_one "spy.us.csv" "SPY_dir_exp_adj" \
  --shadow-score-penalty-mode explicit \
  --shadow-score-return-mode directional \
  --shadow-score-threshold-source adjusted

run_one "spy.us.csv" "SPY_dir_merge_adj" \
  --shadow-score-penalty-mode merged_uncertainty \
  --shadow-score-return-mode directional \
  --shadow-score-threshold-source adjusted

run_one "spy.us.csv" "SPY_abs_merge_adj" \
  --shadow-score-penalty-mode merged_uncertainty \
  --shadow-score-return-mode abs \
  --shadow-score-threshold-source adjusted

# Standardized threshold source variants (calibration on).
run_one "spy.us.csv" "SPY_dir_merge_stdadj" \
  --shadow-score-calibration-mode per_asset_zscore_shrunk \
  --shadow-score-calibration-min-history "$MIN_HISTORY" \
  --shadow-score-calibration-shrinkage-samples "$HISTORY_SIZE" \
  --shadow-score-calibration-std-floor 0.05 \
  --shadow-score-threshold-source standardized_adjusted \
  --shadow-score-penalty-mode merged_uncertainty \
  --shadow-score-return-mode directional

run_one "spy.us.csv" "SPY_abs_merge_stdadj" \
  --shadow-score-calibration-mode per_asset_zscore_shrunk \
  --shadow-score-calibration-min-history "$MIN_HISTORY" \
  --shadow-score-calibration-shrinkage-samples "$HISTORY_SIZE" \
  --shadow-score-calibration-std-floor 0.05 \
  --shadow-score-threshold-source standardized_adjusted \
  --shadow-score-penalty-mode merged_uncertainty \
  --shadow-score-return-mode abs

# BTC: keep as secondary validation (same best-guess config).
run_one "btc_intraday_1s.csv" "BTC_dir_merge_adj" \
  --shadow-score-penalty-mode merged_uncertainty \
  --shadow-score-return-mode directional \
  --shadow-score-threshold-source adjusted

echo "[done] sweep complete: $TS"


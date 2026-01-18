#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${LOG_PATH:-$ROOT/logs/trading_log.csv}"
DASH_REFRESH="${DASH_REFRESH:-1.0}"

cleanup() {
  if [[ -n "${TRADER_PID:-}" ]] && kill -0 "$TRADER_PID" 2>/dev/null; then
    kill "$TRADER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

(
  cd "$ROOT"
  PYTHONPATH=. python run_trader.py "$@"
) &
TRADER_PID=$!

for _ in $(seq 1 200); do
  if [[ -s "$LOG_PATH" ]]; then
    break
  fi
  sleep 0.1
done

if [[ ! -f "$LOG_PATH" ]]; then
  echo "[warn] log not found yet: $LOG_PATH"
fi

dash_cmd=(python training_dashboard_pg.py --log "$LOG_PATH" --refresh "$DASH_REFRESH" --graph-internals)
if [[ -n "${DASH_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$DASH_ARGS"
  dash_cmd+=("${extra_args[@]}")
fi

(
  cd "$ROOT"
  PYTHONPATH=. "${dash_cmd[@]}"
)

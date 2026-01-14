#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_phase3_sweep.sh --runs runs.csv --out-dir DIR --config CONFIG [--run-extra "..."] [--train-extra "..."]

runs.csv columns:
  id,tape,prices,dir_model,meta_features,inst_model

Notes:
- meta_features and inst_model can be empty.
- run-extra is appended to run_proposals.py calls.
- train-extra is appended to train_per_ontology.py calls.
EOF
}

RUNS=""
OUT_DIR=""
CONFIG=""
RUN_EXTRA=""
TRAIN_EXTRA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --run-extra) RUN_EXTRA="$2"; shift 2 ;;
    --train-extra) TRAIN_EXTRA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$RUNS" || -z "$OUT_DIR" || -z "$CONFIG" ]]; then
  usage
  exit 1
fi

mkdir -p "$OUT_DIR"

while IFS=, read -r id tape prices dir_model meta_features inst_model; do
  if [[ "$id" == "id" || -z "$id" ]]; then
    continue
  fi

  base_log="${OUT_DIR}/proposals_${id}.phase2.csv"
  phase3_log="${OUT_DIR}/proposals_${id}.phase3.csv"
  weights="${OUT_DIR}/weights_phase3_${id}.json"
  compare_out="${OUT_DIR}/compare_${id}.txt"
  option_out="${OUT_DIR}/options_${id}.txt"

  cmd_base=(PYTHONPATH=. python scripts/run_proposals.py
    --tape "$tape"
    --prices-csv "$prices"
    --dir-model "$dir_model"
    --use-ontology-legitimacy
    --proposal-log "$base_log"
  )
  if [[ -n "${meta_features:-}" ]]; then
    cmd_base+=(--meta-features "$meta_features")
  fi
  if [[ -n "${inst_model:-}" ]]; then
    cmd_base+=(--inst-model "$inst_model" --use-inst-model)
  fi
  if [[ -n "$RUN_EXTRA" ]]; then
    cmd_base+=($RUN_EXTRA)
  fi
  "${cmd_base[@]}"

  PYTHONPATH=. python scripts/train_per_ontology.py \
    --tape "$tape" \
    --proposal-log "$base_log" \
    --prices-csv "$prices" \
    --config "$CONFIG" \
    --out "$weights" \
    ${TRAIN_EXTRA:-}

  cmd_phase3=(PYTHONPATH=. python scripts/run_proposals.py
    --tape "$tape"
    --prices-csv "$prices"
    --dir-model "$dir_model"
    --use-ontology-legitimacy
    --use-learned-weights "$weights"
    --proposal-log "$phase3_log"
  )
  if [[ -n "${meta_features:-}" ]]; then
    cmd_phase3+=(--meta-features "$meta_features")
  fi
  if [[ -n "${inst_model:-}" ]]; then
    cmd_phase3+=(--inst-model "$inst_model" --use-inst-model)
  fi
  if [[ -n "$RUN_EXTRA" ]]; then
    cmd_phase3+=($RUN_EXTRA)
  fi
  "${cmd_phase3[@]}"

  PYTHONPATH=. python scripts/compare_proposals.py \
    --base "$base_log" \
    --ont "$phase3_log" | tee "$compare_out"

  python - <<PY | tee "$option_out"
import pandas as pd

path = "$phase3_log"
df = pd.read_csv(path)
opt_rows = df["instrument_pred"].astype(str) == "option"
opt_count = int(opt_rows.sum())

tenor = df.loc[opt_rows, "opt_tenor_pred"].astype(str)
mny = df.loc[opt_rows, "opt_mny_pred"].astype(str)

valid_tenor = tenor[(tenor != "none") & tenor.notna()]
valid_mny = mny[(mny != "none") & mny.notna()]

unique_tenor = sorted(valid_tenor.unique().tolist())
unique_mny = sorted(valid_mny.unique().tolist())

print(f"option_rows={opt_count}")
print(f"opt_tenor_unique={unique_tenor} count={len(unique_tenor)}")
print(f"opt_mny_unique={unique_mny} count={len(unique_mny)}")
PY
done < "$RUNS"

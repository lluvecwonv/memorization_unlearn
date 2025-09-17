#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_aggregate_eval.sh <retain_result.json> <ckpt_result.json> [method_name] [save_file]

# Defaults
RETAIN_RESULT_DEFAULT="/root/tofu/files/models/ToFU_finetuned_phi_retain90/checkpoint-562/eval_results/ds_size300/eval_log.json"
CKPT_RESULT_DEFAULT="/root/npo/results/TOFU_phi_simnpo_forget10/checkpoint-60/eval_results/ds_size300/eval_log.json"

RETAIN_RESULT="${1:-$RETAIN_RESULT_DEFAULT}"
CKPT_RESULT="${2:-$CKPT_RESULT_DEFAULT}"

# Derive METHOD_NAME and SPLIT from ckpt path if not provided
if [[ -n "${3:-}" ]]; then
  METHOD_NAME="$3"
else
  # Extract model identifier from different path structures
  if [[ "$CKPT_RESULT" == *"/root/npo/results/"* ]]; then
    # Path like: /root/npo/results/TOFU_phi_grad_ascent_forget10/checkpoint-60/eval_results/ds_size300/eval_log.json
    # We want: TOFU_phi_grad_ascent_forget10 (4 levels up from eval_log.json)
    MODEL_TAG="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$CKPT_RESULT")")")")")"
  elif [[ "$CKPT_RESULT" == *"/root/tofu/files/models/"* ]]; then
    # Path like: /root/tofu/files/models/ToFU_full_phi/checkpoint-625/eval_results/ds_size300/eval_log.json
    # We want: ToFU_full_phi (4 levels up from eval_log.json)
    MODEL_TAG="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$CKPT_RESULT")")")")")"
  else
    # Fallback: try to extract from four levels up
    MODEL_TAG="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$CKPT_RESULT")")")")")"
  fi
  
  # Try to parse for METHOD_NAME if it follows standard pattern
  if [[ "$MODEL_TAG" == *"_"* ]]; then
    IFS='_' read -r CK_DATASET CK_MODEL CK_ALG CK_SPLIT <<<"$MODEL_TAG"
    METHOD_NAME="${CK_MODEL}_${CK_ALG:-unknown}"
    SPLIT="${CK_SPLIT:-forget10}_perturbed"
  else
    # For cases like "ToFU_full_phi", use the full name as method
    METHOD_NAME="$MODEL_TAG"
    SPLIT="forget10_perturbed"
  fi
fi

if [[ -n "${4:-}" ]]; then
  SAVE_FILE="$4"
else
  # Always use MODEL_TAG for consistent naming
  if [[ -n "${MODEL_TAG:-}" ]]; then
    SAVE_FILE="./results/${MODEL_TAG}_aggregated_eval.json"
  else
    SAVE_FILE="./results/${METHOD_NAME}_${SPLIT}_aggregated_eval.json"
  fi
fi

python /root/npo/aggregate_eval_stat.py \
  retain_result=${RETAIN_RESULT} \
  ckpt_result=${CKPT_RESULT} \
  method_name=${METHOD_NAME} \
  save_file=${SAVE_FILE}



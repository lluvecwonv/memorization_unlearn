#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by env or flags)
PORT="${PORT:-2705}"
MODEL_FAMILY="${MODEL_FAMILY:-phi}"
MODEL_PATH="/root/npo/results/TOFU_phi_simnpo_forget10/checkpoint-60"
SPLIT="forget10_perturbed"

# Flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --model-family) MODEL_FAMILY="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

echo "Using settings:"
echo "  PORT         = $PORT"
echo "  MODEL_FAMILY = $MODEL_FAMILY"
echo "  MODEL_PATH   = $MODEL_PATH"
echo "  SPLIT        = $SPLIT"

export PYTHONPATH="/root/npo:${PYTHONPATH:-}"
CMD="CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$PORT evaluate.py --config-path /root/npo/config --config-name eval_everything model_family=$MODEL_FAMILY split=$SPLIT model_path=$MODEL_PATH"
echo "Running: $CMD"
eval "$CMD"
#!/usr/bin/env bash
set -euo pipefail

# Disable DeepSpeed compilation to avoid CPU Adam errors
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_AIO=0

# Defaults (can be overridden by env or flags)
PORT="${PORT:-2705}"
MODEL_FAMILY="${MODEL_FAMILY:-phi}"
MODEL_PATH="/root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/8GPU_simnpo_1e-05_forget10_epoch10_batch4_accum4_beta2.0_gamma0.0_grad_diff_coeff1.0_reffine_tuned_evalsteps_per_epoch_seed1001_1/checkpoint-125"
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

export PYTHONPATH="/root/Unlearn-Simple/TOFU:${PYTHONPATH:-}"
CMD="CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$PORT evaluate_util.py --config-path /root/npo/config --config-name eval_everything model_family=$MODEL_FAMILY split=$SPLIT model_path=$MODEL_PATH"
echo "Running: $CMD"
eval "$CMD"
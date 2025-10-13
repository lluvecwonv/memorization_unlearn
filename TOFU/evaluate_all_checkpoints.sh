#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate tofu

# Configuration
BASE_DIR="/root/tnpo/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/8GPU_tnpo_1e-05_forget10_epoch10_batch4_accum4_beta4.5_gamma0.0_grad_diff_coeff1.0_reffine_tuned_evalsteps_per_epoch_seed1001_1"
MODEL_FAMILY="llama2-7b"
SPLIT="forget10_perturbed"
PORT=2705

# Disable DeepSpeed compilation
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_AIO=0
export PYTHONPATH="/root/tnpo/TOFU:${PYTHONPATH:-}"

# Find all checkpoint directories
CHECKPOINTS=($(ls -d ${BASE_DIR}/checkpoint-* | sort -V))

echo "Found ${#CHECKPOINTS[@]} checkpoints to evaluate"
echo "================================================"

# Retain result path
RETAIN_RESULT="/root/tnpo/TOFU/data/retain90_llama_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"

# Loop through each checkpoint
for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_NAME=$(basename ${CKPT_PATH})
    echo ""
    echo "Evaluating checkpoint: ${CKPT_NAME}"
    echo "Path: ${CKPT_PATH}"
    echo "----------------------------------------"

    # Run evaluation
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=${PORT} \
        /root/tnpo/TOFU/evaluate_util.py \
        --config-path /root/tnpo/TOFU/config \
        --config-name eval_everything \
        model_family=${MODEL_FAMILY} \
        split=${SPLIT} \
        model_path=${CKPT_PATH}

    echo "Finished evaluating ${CKPT_NAME}"

    # Run aggregate statistics
    echo "Running aggregate statistics for ${CKPT_NAME}..."
    CKPT_RESULT="${CKPT_PATH}/eval_log_aggregated.json"
    METHOD_NAME="tnpo_${CKPT_NAME}"
    SAVE_FILE="${CKPT_PATH}/aggregate_results"

    python /root/tnpo/aggregate_eval_stat.py \
        retain_result=${RETAIN_RESULT} \
        ckpt_result=${CKPT_RESULT} \
        method_name=${METHOD_NAME} \
        save_file=${SAVE_FILE}

    echo "Aggregate statistics saved to ${SAVE_FILE}"
    echo "----------------------------------------"
done

echo ""
echo "================================================"
echo "All checkpoints evaluated successfully!"
echo "================================================"

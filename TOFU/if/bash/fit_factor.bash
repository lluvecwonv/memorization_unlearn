#!/bin/bash
set -e

# ========== Model and Data Configuration ==========
MODEL_NAME="path/to/your/finetuned/model"  # TODO: 실제 모델 경로로 변경
MODEL_FAMILY="llama2-7b"  # "llama2-7b" or "pythia"
DATA_PATH="locuslab/TOFU"
SPLIT="full"  # "full", "forget10", "retain90", etc.
QUESTION_KEY="question"
MAX_LENGTH=256

# ========== Factor Configuration ==========
FACTOR_STRATEGY="ekfac"  # "ekfac", "kfac", or "diagonal"
TRAIN_BATCH_SIZE=1
OUTPUT_DIR="./kronfluence_factors"

# ========== Script Path Setup ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # TOFU 디렉토리로 이동
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# ========== GPU / Memory Configuration ==========
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=18765
export NPROC_PER_NODE=1  # GPU 개수에 맞게 조정
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"

echo "🚀 Starting Kronfluence factor fitting..."
echo "Model: ${MODEL_NAME}"
echo "Model Family: ${MODEL_FAMILY}"
echo "Dataset: ${DATA_PATH} (split: ${SPLIT})"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} | Processes: ${NPROC_PER_NODE} | Port: ${MASTER_PORT}"

# ========== Run Factor Fitting ==========
torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
  if/fit_factor.py \
  --model_name "${MODEL_NAME}" \
  --model_family "${MODEL_FAMILY}" \
  --data_path "${DATA_PATH}" \
  --split "${SPLIT}" \
  --question_key "${QUESTION_KEY}" \
  --max_length ${MAX_LENGTH} \
  --factor_strategy "${FACTOR_STRATEGY}" \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --output_dir "${OUTPUT_DIR}" \
  --use_half_precision \
  --covariance_module_partitions 4 \
  --lambda_module_partitions 4 \
  --covariance_data_partitions 4 \
  --lambda_data_partitions 4

echo "✅ Factor fitting completed!"
echo "Output directory: ${OUTPUT_DIR}"

#!/bin/bash
set -e

MODEL_NAME="/root/tofu/files/models/ToFU_full_phi/checkpoint-625"
MODEL_FAMILY="phi"
FACTOR_STRATEGY="ekfac"
TRAIN_BATCH_SIZE=1
OUTPUT_DIR="/workspace/nas_chaen/kronfluence_factors_phi"
MAX_LENGTH=128          # ↓ 256→128로 줄여 메모리 여유
SPLIT="forget10"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# ========== GPU / 메모리 환경 ==========
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=18765
export NPROC_PER_NODE=2
# 파편화 줄이기 + 세그먼트 확장 허용
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"
# 디버깅시: export CUDA_LAUNCH_BLOCKING=1

echo "� Starting Kronfluence factor fitting..."
echo "GPUs: ${CUDA_VISIBLE_DEVICES}  procs:${NPROC_PER_NODE}  port:${MASTER_PORT}"

torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
  /root/npo/if/fit_factor.py \
  --model_name "${MODEL_NAME}" \
  --model_family "${MODEL_FAMILY}" \
  --factor_strategy "${FACTOR_STRATEGY}" \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --output_dir "${OUTPUT_DIR}" \
  --max_length ${MAX_LENGTH} \
  --split "${SPLIT}" \
  --use_half_precision \
  --covariance_module_partitions 8 \
  --lambda_module_partitions 12 \
  --covariance_data_partitions 8 \
  --lambda_data_partitions 12

echo "✅ Factor fitting completed!"

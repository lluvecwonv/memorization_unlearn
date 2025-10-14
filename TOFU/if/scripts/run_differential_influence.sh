#!/bin/bash

# Differential Token Influence Scoring Script
# S_ij = (g_forget - g_retain)^T * H^{-1} * ∇_θ L(x_ij; θ)

set -e

# Default configuration
CHECKPOINT_DIR="/root/files/models/ToFU_finetuned_llama2-7b_full/checkpoint-625"
FACTORS_PATH="/workspace/nas_chaen/kronfluence_factors_llama-7b"
FACTORS_NAME=""
TOFU_DATA_PATH="locuslab/TOFU"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
MODEL_FAMILY="llama2-7b"
QUERY_BATCH_SIZE=1
TRAIN_BATCH_SIZE=1
MAX_LENGTH=256
SAVE_DIR="/root/Unlearn-Simple/TOFU/token_weight/forget_masks"
GPUS="0,1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --factors-path) FACTORS_PATH="$2"; shift 2 ;;
        --factors-name) FACTORS_NAME="$2"; shift 2 ;;
        --tofu-data-path) TOFU_DATA_PATH="$2"; shift 2 ;;
        --forget-split) FORGET_SPLIT="$2"; shift 2 ;;
        --retain-split) RETAIN_SPLIT="$2"; shift 2 ;;
        --model-family) MODEL_FAMILY="$2"; shift 2 ;;
        --batch-size) QUERY_BATCH_SIZE="$2"; TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        -h|--help) 
            echo "Usage: $0 [options]"
            echo "  --checkpoint-dir PATH"
            echo "  --factors-path PATH"
            echo "  --factors-name NAME"
            echo "  --tofu-data-path PATH"
            echo "  --forget-split NAME"
            echo "  --retain-split NAME"
            echo "  --model-family NAME"
            echo "  --batch-size INT"
            echo "  --max-length INT"
            echo "  --save-dir PATH"
            echo "  --gpus LIST               (default: 0)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "🚀 Starting differential influence computation..."
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "📊 Batch size: $QUERY_BATCH_SIZE, Max length: $MAX_LENGTH"
echo "📦 Factors: $FACTORS_PATH / $FACTORS_NAME"
echo "🧪 Splits: forget=$FORGET_SPLIT retain=$RETAIN_SPLIT"
echo "💾 Save to: $SAVE_DIR"
echo ""

# Check paths
[[ ! -d "$CHECKPOINT_DIR" ]] && { echo "❌ Checkpoint not found: $CHECKPOINT_DIR"; exit 1; }
if [[ ! -d "$FACTORS_PATH" ]]; then
    echo "⚠️ Factors directory not found: $FACTORS_PATH"
    exit 1
else
    echo "✅ Factors directory found: $FACTORS_PATH"
fi

# Create output directory
mkdir -p "$SAVE_DIR"

# Set environment and run
export PYTHONPATH=/root:/root/Unlearn-Simple/TOFU
export CUDA_VISIBLE_DEVICES="$GPUS"

SAVE_ID="differential_$(date +%Y%m%d_%H%M%S)"

# Use torchrun for multi-GPU setup
NPROC_PER_NODE=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MASTER_PORT=${MASTER_PORT:-29500}

echo "🔧 Using $NPROC_PER_NODE GPUs: $GPUS"
echo "🔌 Master Port: $MASTER_PORT"
echo ""

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    /root/Unlearn-Simple/TOFU/if/compute_influence.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --tofu_data_path "$TOFU_DATA_PATH" \
    --forget_split "$FORGET_SPLIT" \
    --retain_split "$RETAIN_SPLIT" \
    --factors_path "$FACTORS_PATH" \
    --factors_name "ekfac_half" \
    --query_batch_size "$QUERY_BATCH_SIZE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --model_family "$MODEL_FAMILY" \
    --max_length "$MAX_LENGTH" \
    --save_dir "$SAVE_DIR" \
    --save_id "$SAVE_ID"

echo ""
echo "✅ Differential influence computation completed!"
echo "📁 Check results in: $SAVE_DIR"
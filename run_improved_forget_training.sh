#!/bin/bash

# TNPO + 유해 토큰 마스킹 훈련 스크립트
# torchrun을 사용한 분산 훈련 지원

set -e

# Default configuration
MODEL_FAMILY="phi"
MODEL_PATH="/root/tofu/files/models/ToFU_full_phi/checkpoint-625"
DATA_PATH="locuslab/TOFU"
SPLIT=forget10
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=5
LEARNING_RATE=1e-5
FORGET_LOSS="TNPO"
MASTER_PORT=29500
SAVE_DIR=""  # Will be set after parsing arguments
FORGET_MASK_PATH="/root/npo/token_weight/forget_masks/forget_token_mask_20250907_214404.pt"
FORGET_LAMBDA=1.0
SI_SCORES_PATH="/root/npo/token_weight/forget_masks/si_scores_20250907_214016.pt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-family) MODEL_FAMILY="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --gradient-accumulation-steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
        --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --forget-loss) FORGET_LOSS="$2"; shift 2 ;;
        --forget-mask-path) FORGET_MASK_PATH="$2"; shift 2 ;;
        --toxic-mask-path) TOXIC_MASK_PATH="$2"; shift 2 ;;
        --toxic-lambda) TOXIC_LAMBDA="$2"; shift 2 ;;
        --si-scores-path) SI_SCORES_PATH="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --model-family        Model family (default: phi)"
            echo "  --model-path          Model checkpoint path"
            echo "  --data-path           Dataset path (default: locuslab/TOFU)"
            echo "  --split               Data split (default: forget10)"
            echo "  --save-dir            Save directory"
            echo "  --batch-size          Batch size (default: 4)"
            echo "  --gradient-accumulation-steps  Gradient accumulation (default: 4)"
            echo "  --num-epochs          Number of epochs (default: 5)"
            echo "  --lr                  Learning rate (default: 1e-5)"
            echo "  --forget-loss         Forget loss type (default: TNPO)"
            echo "  --forget-mask-path    Path to forget token mask .pt (TNPO)"
            echo "  --toxic-mask-path     Path to toxic token mask .pt (IF-Guide)"
            echo "  --toxic-lambda        Toxic token suppression strength (default: 1.0)"
            echo "  --si-scores-path      Path to SI/token weights .pt (TNPO)"
            echo "  --master-port         Master port for torchrun (default: 29500)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Set default save directory if not provided
if [[ -z "$SAVE_DIR" ]]; then
    # Extract dataset name from DATA_PATH (e.g., "TOFU" from "locuslab/TOFU")
    DATASET_NAME=$(basename "$DATA_PATH")
    
    # Create descriptive save directory
    if [[ -n "$TOXIC_MASK_PATH" ]]; then
        SAVE_DIR="./results/${DATASET_NAME}_${MODEL_FAMILY}_${FORGET_LOSS}_toxic_${SPLIT}"
    else
        SAVE_DIR="./results/${DATASET_NAME}_${MODEL_FAMILY}_${FORGET_LOSS}_${SPLIT}"
    fi
fi

echo "🧠 Starting TNPO + Toxic Token Masking Training with torchrun..."
echo "📁 Model: $MODEL_PATH"
echo "📊 Data: $DATA_PATH (split: $SPLIT)"
echo "💾 Save to: $SAVE_DIR"
echo "⚙️  Batch size: $BATCH_SIZE, Grad accum: $GRADIENT_ACCUMULATION_STEPS"
echo "🎯 Epochs: $NUM_EPOCHS, LR: $LEARNING_RATE"
echo "🔧 Forget loss: $FORGET_LOSS"
if [[ -n "$FORGET_MASK_PATH" ]]; then echo "🗺️  Forget mask: $FORGET_MASK_PATH"; fi
if [[ -n "$TOXIC_MASK_PATH" ]]; then echo "☠️  Toxic mask: $TOXIC_MASK_PATH (lambda: $TOXIC_LAMBDA)"; fi
if [[ -n "$SI_SCORES_PATH" ]]; then echo "📈 SI scores: $SI_SCORES_PATH"; fi
echo ""

# Check if model exists
[[ -d "$MODEL_PATH" ]] || { echo "❌ Model not found: $MODEL_PATH"; exit 1; }

# Check if toxic mask exists (if provided)
if [[ -n "$TOXIC_MASK_PATH" ]]; then
    [[ -f "$TOXIC_MASK_PATH" ]] || { echo "❌ Toxic mask not found: $TOXIC_MASK_PATH"; exit 1; }
fi

# Check if forget mask exists (if provided)
if [[ -n "$FORGET_MASK_PATH" ]]; then
    [[ -f "$FORGET_MASK_PATH" ]] || { echo "❌ Forget mask not found: $FORGET_MASK_PATH"; exit 1; }
fi

# Change to script directory
cd /root/npo

# Run with torchrun (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 \
FORGET_MASK_PATH="$FORGET_MASK_PATH" \
TOXIC_TOKEN_MASK_PATH="$TOXIC_MASK_PATH" \
TOXIC_LAMBDA="$TOXIC_LAMBDA" \
SI_SCORES_PATH="$SI_SCORES_PATH" \
FORGET_LOSS_TYPE="$FORGET_LOSS" \
torchrun \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    forget.py \
    model_family="$MODEL_FAMILY" \
    model_path="$MODEL_PATH" \
    data_path="$DATA_PATH" \
    split="$SPLIT" \
    save_dir="$SAVE_DIR" \
    batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    num_epochs=$NUM_EPOCHS \
    lr=$LEARNING_RATE \
    forget_loss="$FORGET_LOSS" \
    +toxic_token_mask_path="$FORGET_MASK_PATH" \
    +toxic_lambda=$FORGET_LAMBDA \
    overwrite_dir=true

echo ""
echo "✅ TNPO + Toxic Token Masking training completed!"
echo "📁 Results saved to: $SAVE_DIR"
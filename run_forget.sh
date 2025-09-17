#!/bin/bash

# Simple Forget Training Script using torchrun
# Based on: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml

set -e

# Default configuration
MODEL_FAMILY="phi"
MODEL_PATH="/root/tofu/files/models/ToFU_full_phi/checkpoint-625"
DATA_PATH="locuslab/TOFU"
SPLIT="forget10"
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=5
LEARNING_RATE=1e-5
FORGET_LOSS="simnpo"
MASTER_PORT=25476
SAVE_DIR=""  # Will be set after parsing arguments
FORGET_MASK_PATH="/root/outputs/si_report_20250917_015956/forget_token_mask.pt"

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
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --model-family        Model family (default: pythia-1.4)"
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
            echo "  --master-port         Master port for torchrun (default: 29500)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Set default save directory if not provided
if [[ -z "$SAVE_DIR" ]]; then
    # Extract dataset name from DATA_PATH (e.g., "TOFU" from "locuslab/TOFU")
    DATASET_NAME=$(basename "$DATA_PATH")
    
    # Create descriptive save directory without timestamp
    SAVE_DIR="./results/${DATASET_NAME}_${MODEL_FAMILY}_${FORGET_LOSS}_${SPLIT}"
fi

echo "🧠 Starting Forget Training with torchrun..."
echo "📁 Model: $MODEL_PATH"
echo "📊 Data: $DATA_PATH (split: $SPLIT)"
echo "💾 Save to: $SAVE_DIR"
echo "⚙️  Batch size: $BATCH_SIZE, Grad accum: $GRADIENT_ACCUMULATION_STEPS"
echo "🎯 Epochs: $NUM_EPOCHS, LR: $LEARNING_RATE"
echo "🔧 Forget loss: $FORGET_LOSS"
if [[ -n "$FORGET_MASK_PATH" ]]; then echo "🗺️  Forget mask: $FORGET_MASK_PATH"; fi
echo ""

# Check if model exists
[[ -d "$MODEL_PATH" ]] || { echo "❌ Model not found: $MODEL_PATH"; exit 1; }

# Change to script directory
cd /root/npo

# Run with torchrun (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 FORGET_MASK_PATH="$FORGET_MASK_PATH" FORGET_LOSS_TYPE="$FORGET_LOSS" torchrun \
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
    +forget_mask_path="$FORGET_MASK_PATH" \
    overwrite_dir=true

echo ""
echo "✅ Forget training completed!"
echo "📁 Results saved to: $SAVE_DIR"
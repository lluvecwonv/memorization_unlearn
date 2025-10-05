#!/bin/bash

# Simple wrapper for forget training
# Usage: bash run_forget.sh [split] [npo_coeff] [beta] [additional_args...]
# Example: bash run_forget.sh forget05 0.1375 2.5
# Example: BATCH_SIZE=8 GRAD_ACC=2 bash run_forget.sh forget05 0.1375 2.5

set -e

# Default values
SPLIT="full"
BATCH_SIZE=4
GRAD_ACC=4
MASTER_PORT=29500
GPUS="0,1"
NPROC=2

# Shift the first 3 arguments so remaining args can be passed through
shift 3 2>/dev/null || true

echo "🧠 Starting Forget Training..."
echo "📊 Split: $SPLIT"
echo "⚙️  NPO Coeff: $NPO_COEFF, Beta: $BETA"
echo "📦 Batch Size: $BATCH_SIZE, Grad Acc: $GRAD_ACC"
echo "🔧 GPUs: $GPUS (${NPROC} processes)"
echo "🔌 Master Port: $MASTER_PORT"
echo ""

# Change to script directory
cd /root/Unlearn-Simple/TOFU

# Simple torchrun command like in README
CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    finetune.py \
    --config-name=finetune \
    split=$SPLIT \
    batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$GRAD_ACC \
    "$@"

echo ""
echo "✅ Forget training completed!"
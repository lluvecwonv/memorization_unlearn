#!/bin/bash
set -euo pipefail

# Usage: ./run_eval_augmentation.sh <model_path> <data_path> <split> <eval_task> <base_answer_key> <compare_answer_key> [save_dir] [model_family]
# Example: ./run_eval_augmentation.sh /path/to/model locuslab/TOFU retain augmentation_eval answer answer_perturbed ./results llama2-7b

# Check required arguments
if [ $# -lt 6 ]; then
    echo "Usage: $0 <model_path> <data_path> <split> <eval_task> <base_answer_key> <compare_answer_key> [save_dir] [model_family]"
    echo "Example: $0 /path/to/model locuslab/TOFU retain augmentation_eval answer answer_perturbed ./results llama2-7b"
    echo ""
    echo "This script runs augmentation evaluation using eval_augmentation.py"
    echo "This evaluates model bias by comparing original vs perturbed answers"
    exit 1
fi

MODEL_PATH="$1"
DATA_PATH="$2"
SPLIT="$3"
EVAL_TASK="$4"
BASE_ANSWER_KEY="$5"
COMPARE_ANSWER_KEY="$6"
SAVE_DIR="${7:-./results}"
MODEL_FAMILY="${8:-llama2-7b}"

# Create save directory
mkdir -p "$SAVE_DIR"

echo "=========================================="
echo "Running Augmentation Evaluation (eval_augmentation.py)"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Data Path: $DATA_PATH"
echo "Split: $SPLIT"
echo "Eval Task: $EVAL_TASK"
echo "Base Answer Key: $BASE_ANSWER_KEY"
echo "Compare Answer Key: $COMPARE_ANSWER_KEY"
echo "Save Directory: $SAVE_DIR"
echo "Model Family: $MODEL_FAMILY"
echo "=========================================="

# Check if model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Run augmentation evaluation
echo "Running augmentation evaluation..."
python eval_augmentation.py \
    --config-name=eval_augmentation \
    model_path="$MODEL_PATH" \
    data_path="$DATA_PATH" \
    split="$SPLIT" \
    eval_task="$EVAL_TASK" \
    base_answer_key="$BASE_ANSWER_KEY" \
    compare_answer_key="$COMPARE_ANSWER_KEY" \
    save_dir="$SAVE_DIR" \
    model_family="$MODEL_FAMILY" \
    overwrite=true

echo "=========================================="
echo "Augmentation Evaluation Complete!"
echo "=========================================="
echo "Results saved in: $SAVE_DIR/${EVAL_TASK}.json"
echo "=========================================="

# List generated files
echo "Generated files:"
ls -la "$SAVE_DIR"/*.json 2>/dev/null || echo "No JSON files found"

echo "=========================================="
echo "To view results:"
echo "python3 -m json.tool $SAVE_DIR/${EVAL_TASK}.json"
echo ""
echo "Key metrics to check:"
echo "- ground_truth_loss: Loss on original answers"
echo "- perturb_loss: Loss on perturbed answers"
echo "- Loss ratio: (perturb_loss - ground_truth_loss)"
echo "  - Positive: Model prefers original answers (bias)"
echo "  - Negative: Model prefers perturbed answers"
echo "  - Near zero: Model is neutral"
echo "=========================================="

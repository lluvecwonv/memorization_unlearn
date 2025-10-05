#!/bin/bash
set -euo pipefail

# Usage: ./run_eval_basic.sh <model_path> <data_path> <split> <eval_task> [save_dir] [model_family]
# Example: ./run_eval_basic.sh /path/to/model locuslab/TOFU retain retain_eval ./results llama2-7b

# Check required arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <model_path> <data_path> <split> <eval_task> [save_dir] [model_family]"
    echo "Example: $0 /path/to/model locuslab/TOFU retain retain_eval ./results llama2-7b"
    echo ""
    echo "This script runs basic evaluation using eval.py"
    exit 1
fi

MODEL_PATH="$1"
DATA_PATH="$2"
SPLIT="$3"
EVAL_TASK="$4"
SAVE_DIR="${5:-./results}"
MODEL_FAMILY="${6:-llama2-7b}"

# Create save directory
mkdir -p "$SAVE_DIR"

echo "=========================================="
echo "Running Basic Evaluation (eval.py)"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Data Path: $DATA_PATH"
echo "Split: $SPLIT"
echo "Eval Task: $EVAL_TASK"
echo "Save Directory: $SAVE_DIR"
echo "Model Family: $MODEL_FAMILY"
echo "=========================================="

# Check if model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Run basic evaluation
echo "Running evaluation..."
python eval.py \
    --config-name=eval \
    model_path="$MODEL_PATH" \
    data_path="$DATA_PATH" \
    split="$SPLIT" \
    eval_task="$EVAL_TASK" \
    save_dir="$SAVE_DIR" \
    model_family="$MODEL_FAMILY" \
    overwrite=true \
    save_generated_text=true

echo "=========================================="
echo "Basic Evaluation Complete!"
echo "=========================================="
echo "Results saved in: $SAVE_DIR/${EVAL_TASK}.json"
echo "=========================================="

# List generated files
echo "Generated files:"
ls -la "$SAVE_DIR"/*.json 2>/dev/null || echo "No JSON files found"

echo "=========================================="
echo "To view results:"
echo "python3 -m json.tool $SAVE_DIR/${EVAL_TASK}.json"
echo "=========================================="

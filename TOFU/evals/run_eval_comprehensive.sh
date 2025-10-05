#!/bin/bash
set -euo pipefail

# Usage: ./run_eval_comprehensive.sh <model_path> <data_paths> <splits> <eval_tasks> [save_dir] [model_family]
# Example: ./run_eval_comprehensive.sh /path/to/model "locuslab/TOFU,locuslab/TOFU" "retain,forget" "retain_eval,forget_eval" ./results llama2-7b

# Check required arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <model_path> <data_paths> <splits> <eval_tasks> [save_dir] [model_family]"
    echo "Example: $0 /path/to/model \"locuslab/TOFU,locuslab/TOFU\" \"retain,forget\" \"retain_eval,forget_eval\" ./results llama2-7b"
    echo ""
    echo "This script runs comprehensive evaluation using eval_everything.py"
    echo "Multiple data paths, splits, and tasks should be comma-separated"
    exit 1
fi

MODEL_PATH="$1"
DATA_PATHS="$2"
SPLITS="$3"
EVAL_TASKS="$4"
SAVE_DIR="${5:-./results}"
MODEL_FAMILY="${6:-llama2-7b}"

# Convert comma-separated strings to arrays
IFS=',' read -ra DATA_PATH_ARRAY <<< "$DATA_PATHS"
IFS=',' read -ra SPLIT_ARRAY <<< "$SPLITS"
IFS=',' read -ra EVAL_TASK_ARRAY <<< "$EVAL_TASKS"

# Create save directory
mkdir -p "$SAVE_DIR"

echo "=========================================="
echo "Running Comprehensive Evaluation (eval_everything.py)"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Data Paths: $DATA_PATHS"
echo "Splits: $SPLITS"
echo "Eval Tasks: $EVAL_TASKS"
echo "Save Directory: $SAVE_DIR"
echo "Model Family: $MODEL_FAMILY"
echo "=========================================="

# Check if model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Validate array lengths
if [ ${#DATA_PATH_ARRAY[@]} -ne ${#SPLIT_ARRAY[@]} ] || [ ${#SPLIT_ARRAY[@]} -ne ${#EVAL_TASK_ARRAY[@]} ]; then
    echo "Error: Number of data paths, splits, and eval tasks must match"
    echo "Data paths: ${#DATA_PATH_ARRAY[@]}, Splits: ${#SPLIT_ARRAY[@]}, Tasks: ${#EVAL_TASK_ARRAY[@]}"
    exit 1
fi

# Format arrays for Python
DATA_PATHS_FORMATTED="[$(printf '"%s",' "${DATA_PATH_ARRAY[@]}" | sed 's/,$//')]"
SPLITS_FORMATTED="[$(printf '"%s",' "${SPLIT_ARRAY[@]}" | sed 's/,$//')]"
EVAL_TASKS_FORMATTED="[$(printf '"%s",' "${EVAL_TASK_ARRAY[@]}" | sed 's/,$//')]"

# Run comprehensive evaluation
echo "Running comprehensive evaluation..."
python eval_everything.py \
    --config-name=eval_everything \
    model_path="$MODEL_PATH" \
    data_path="$DATA_PATHS_FORMATTED" \
    split_list="$SPLITS_FORMATTED" \
    eval_task="$EVAL_TASKS_FORMATTED" \
    question_key="[question, question]" \
    answer_key="[answer, answer]" \
    base_answer_key="[answer, answer]" \
    perturbed_answer_key="[answer_perturbed, answer_perturbed]" \
    save_dir="$SAVE_DIR" \
    model_family="$MODEL_FAMILY" \
    overwrite=true

echo "=========================================="
echo "Comprehensive Evaluation Complete!"
echo "=========================================="
echo "Results saved in: $SAVE_DIR"
echo "=========================================="

# List generated files
echo "Generated files:"
for task in "${EVAL_TASK_ARRAY[@]}"; do
    if [ -f "$SAVE_DIR/${task}.json" ]; then
        echo "- $SAVE_DIR/${task}.json"
    fi
done

echo "=========================================="
echo "To view results:"
for task in "${EVAL_TASK_ARRAY[@]}"; do
    if [ -f "$SAVE_DIR/${task}.json" ]; then
        echo "python3 -m json.tool $SAVE_DIR/${task}.json"
    fi
done
echo "=========================================="

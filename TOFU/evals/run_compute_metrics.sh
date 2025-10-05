#!/bin/bash
set -euo pipefail

# Usage: ./run_compute_metrics.sh <retain_eval_json> <unlearned_eval_json> [save_dir] [method_name]
# Example: ./run_compute_metrics.sh /path/to/retain_eval.json /path/to/unlearned_eval.json ./results simnpo_forget10

# Check required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <retain_eval_json> <unlearned_eval_json> [save_dir] [method_name]"
    echo "Example: $0 /path/to/retain_eval.json /path/to/unlearned_eval.json ./results simnpo_forget10"
    echo ""
    echo "This script computes Model Utility and Forget Quality from two evaluation JSON files."
    exit 1
fi

RETAIN_EVAL_JSON="/root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_retain90_seed42_1/checkpoint-562/eval_results/ds_size300/eval_log.json"
UNLEARNED_EVAL_JSON="/root/Unlearn-Simple/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/unlearned/simnpo/checkpoint-125/eval_log.json"
SAVE_DIR="$/root/Unlearn-Simple/TOFU/results"
METHOD_NAME="$simnpo"

# Create save directory
mkdir -p "$SAVE_DIR"

echo "=========================================="
echo "Computing Model Utility and Forget Quality"
echo "=========================================="
echo "Retain Eval JSON: $RETAIN_EVAL_JSON"
echo "Unlearned Eval JSON: $UNLEARNED_EVAL_JSON"
echo "Save Directory: $SAVE_DIR"
echo "Method Name: $METHOD_NAME"
echo "=========================================="

# Check if files exist
if [ ! -f "$RETAIN_EVAL_JSON" ]; then
    echo "Error: Retain eval JSON file not found: $RETAIN_EVAL_JSON"
    exit 1
fi

if [ ! -f "$UNLEARNED_EVAL_JSON" ]; then
    echo "Error: Unlearned eval JSON file not found: $UNLEARNED_EVAL_JSON"
    exit 1
fi

# Run the metrics computation
echo "Computing metrics..."
python3 compute_metrics.py \
    --retain "$RETAIN_EVAL_JSON" \
    --unlearned "$UNLEARNED_EVAL_JSON" \
    --save "$SAVE_DIR/${METHOD_NAME}_metrics_summary.json"

echo "=========================================="
echo "Metrics Computation Complete!"
echo "=========================================="
echo "Results saved in: $SAVE_DIR/${METHOD_NAME}_metrics_summary.json"
echo "=========================================="

# Show the results
echo "Summary of results:"
python3 -c "
import json
with open('$SAVE_DIR/${METHOD_NAME}_metrics_summary.json', 'r') as f:
    data = json.load(f)
    
print('Model Utility:', data['model_utility']['Model Utility'])
print('Forget Quality (KS p-value):', data['forget_quality']['Forget Quality'])
print('KS Test Statistic:', data['forget_quality']['KS Test Forget'])
print()
print('Detailed Model Utility:')
for key, value in data['model_utility'].items():
    if key != 'Model Utility':
        print(f'  {key}: {value}')
"

echo "=========================================="
echo "To view full results:"
echo "python3 -m json.tool $SAVE_DIR/${METHOD_NAME}_metrics_summary.json"
echo "=========================================="

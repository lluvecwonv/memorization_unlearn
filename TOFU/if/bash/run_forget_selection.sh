#!/bin/bash

# Forget token selection: Top-k or normalized threshold mode with word masking

set -euo pipefail

# Get script directory (where this bash script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is 2 levels up from /if/bash/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths (relative to project root)
SCORES_PATH="${PROJECT_ROOT}/if/token_weight/forget_masks/differential_token_ekfac_differential_20250912_200418/pairwise_scores.safetensors"
OUTPUT_DIR="${PROJECT_ROOT}/if/token_weight/forget_masks/simple_$(date +%Y%m%d_%H%M%S)"
PYTHON_SCRIPT="${PROJECT_ROOT}/if/select_forget_tokens.py"

# Default parameters
TOP_K=2  # Number of top tokens per sample

# Parse arguments
if [[ $# -ge 1 ]]; then
    SCORES_PATH="$1"
fi
if [[ $# -ge 2 ]]; then
    OUTPUT_DIR="$2"
fi
if [[ $# -ge 3 ]]; then
    TOP_K="$3"
fi

echo "🔍 Forget token selection"
echo "📊 Scores: $SCORES_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo "🐍 Script: $PYTHON_SCRIPT"
echo "🔢 Top-K: $TOP_K per sample"
echo ""

# Check scores file exists
if [[ ! -e "$SCORES_PATH" ]]; then
    echo "❌ Scores file not found: $SCORES_PATH"
    exit 1
fi

# Check python script exists
if [[ ! -e "$PYTHON_SCRIPT" ]]; then
    echo "❌ Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to TOFU directory so utils.py can find config/model_config.yaml
cd "$PROJECT_ROOT"

# Run selection with top-k mode
python3 "$PYTHON_SCRIPT" \
    --scores_path "$SCORES_PATH" \
    --model_family llama2-7b \
    --output_path "$OUTPUT_DIR/forget_token_mask.pt" \
    --top_n_per_sample "$TOP_K" \
    --word_masking \
    --context_window 0 \
    --verbose

echo ""
echo "✅ Complete! Output files:"
echo "   🎭 Mask: $OUTPUT_DIR/forget_token_mask.pt"
echo "   📊 SI scores: $OUTPUT_DIR/si_scores.pt"
echo "   📄 Selected tokens: $OUTPUT_DIR/forget_token_mask_selected_tokens_si.json"
echo "   📄 All tokens: $OUTPUT_DIR/forget_token_mask_all_tokens_si.json"
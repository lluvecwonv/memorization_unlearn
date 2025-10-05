#!/bin/bash

# Simple forget token selection: Top-1 per sample with word masking

set -euo pipefail

# Simple defaults
SCORES_PATH="/root/Unlearn-Simple/TOFU/if/token_weight/forget_masks/differential_token_ekfac_differential_20250912_200418/pairwise_scores.pt"
OUTPUT_DIR="/root/Unlearn-Simple/TOFU/if/token_weight/forget_masks/simple_$(date +%Y%m%d_%H%M%S)"

# Fixed settings for simplicity
MODEL_CONFIG_PATH="/root/Unlearn-Simple/TOFU/config/model_config.yaml"
TOFU_DATA_PATH="locuslab/TOFU"
FORGET_SPLIT="forget10"
MODEL_FAMILY="llama2-7b"
MAX_LENGTH=256
TOP_N_PER_SAMPLE=1  # Only top-1 per sample
CONTEXT_WINDOW=0    # No context expansion
MIN_SCORE=0.0

# Parse simple arguments
if [[ $# -ge 1 ]]; then
    SCORES_PATH="$1"
fi
if [[ $# -ge 2 ]]; then
    OUTPUT_DIR="$2"
fi

echo "🔍 Simple forget token selection"
echo "📊 Scores: $SCORES_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo "🎯 Top-1 tokens per sample with word masking"
echo ""

# Check scores file exists
if [[ ! -e "$SCORES_PATH" ]]; then
    echo "❌ Scores file not found: $SCORES_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run selection
python /root/Unlearn-Simple/TOFU/if/select_forget_tokens.py \
    --scores_path "$SCORES_PATH" \
    --model_family "$MODEL_FAMILY" \
    --model_config_path "$MODEL_CONFIG_PATH" \
    --tofu_data_path "$TOFU_DATA_PATH" \
    --forget_split "$FORGET_SPLIT" \
    --max_length "$MAX_LENGTH" \
    --context_window "$CONTEXT_WINDOW" \
    --top_n_per_sample "$TOP_N_PER_SAMPLE" \
    --min_score "$MIN_SCORE" \
    --output_path "$OUTPUT_DIR/forget_token_mask.pt" \
    --word_masking \
    --verbose

echo ""
echo "✅ Complete! Output files:"
echo "   🎭 Mask: $OUTPUT_DIR/forget_token_mask.pt"
echo "   📊 SI scores: $OUTPUT_DIR/si_scores.pt"
echo "   📄 Selected tokens: $OUTPUT_DIR/forget_token_mask_selected_tokens_si.json"
echo "   📄 All tokens: $OUTPUT_DIR/forget_token_mask_all_tokens_si.json"
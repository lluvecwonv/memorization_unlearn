#!/bin/bash

# Script to export SI report (JSONL + HTML/PNG) and optionally use an existing mask

set -euo pipefail

# Minimal defaults (edit if needed)
SCORES_PATH="/root/npo/token_weight/forget_masks/differential_token_ekfac_differential_20250912_200418/pairwise_scores.pt"
OUTPUT_DIR="/root/outputs/si_report_$(date +%Y%m%d_%H%M%S)"
MASK_PATH=""               # Optional existing mask (.pt)
BUILD_MASK=1               # Always build mask with word-masking enabled by default

# Internal defaults (not exposed as CLI)
CHECKPOINT_DIR="/root/tofu/files/models/ToFU_finetuned_pythia-410m_full/checkpoint-625"
MODEL_CONFIG_PATH="/root/npo/config/model_config.yaml"
TOFU_DATA_PATH="locuslab/TOFU"
FORGET_SPLIT="forget10"
MODEL_FAMILY="pythia-1.4"
MAX_LENGTH=256
MASK_CONTEXT_WINDOW=0
TOP_N_PER_SAMPLE=2
MIN_SCORE=0.0
TOP_TOKENS_SELECTED_ONLY=1
FLOAT_PRECISION=6
VIZ_TOP_K_SAMPLES=50
VERBOSE=1
# Plot sizes
TOP_K_TOKENS=30   # set -1 for ALL
# New: context examples for words in export_si_report.py
CONTEXT_WINDOW_WORDS=0
EXAMPLES_PER_ITEM=2
TOP_K_WORDS=30

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scores-path) SCORES_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --mask-path) MASK_PATH="$2"; shift 2 ;;
        --build-mask) BUILD_MASK=1; shift 1 ;;
        -h|--help)
            echo "Usage: $0 --scores-path PATH [--output-dir PATH] [--mask-path PATH] [--build-mask]"
            exit 0 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Basic checks
if [[ -z "$SCORES_PATH" || ! -e "$SCORES_PATH" ]]; then
    echo "❌ Scores path not found: $SCORES_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Optionally build mask beforehand (word-masking enabled by default)
if [[ "$BUILD_MASK" == "1" ]]; then
    echo "🧱 Building forget mask via select_forget_tokens.py (word_masking=ON)"
    [[ -z "$MASK_PATH" ]] && MASK_PATH="$OUTPUT_DIR/forget_token_mask.pt"
    mkdir -p "$(dirname "$MASK_PATH")"

    BUILD_CMD=(env PYTHONPATH="/root:/root/npo" python "/root/npo/if/select_forget_tokens.py"
        --scores_path "$SCORES_PATH"
        --checkpoint_dir "$CHECKPOINT_DIR"
        --model_config_path "$MODEL_CONFIG_PATH"
        --tofu_data_path "$TOFU_DATA_PATH"
        --forget_split "$FORGET_SPLIT"
        --model_family "$MODEL_FAMILY"
        --max_length "$MAX_LENGTH"
        --context_window "$MASK_CONTEXT_WINDOW"
        --top_n_per_sample "$TOP_N_PER_SAMPLE"
        --min_score "$MIN_SCORE"
        --output_path "$MASK_PATH"
        --word_masking
    )
    echo "Running: ${BUILD_CMD[*]}"
    "${BUILD_CMD[@]}"
    echo "✅ Built mask at: $MASK_PATH"
fi

# Run SI report export
PYTHON_BIN="python"

CMD=(env PYTHONPATH="/root:/root/npo" "$PYTHON_BIN" "/root/npo/if/export_si_report.py"
    --scores_path "$SCORES_PATH"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --model_config_path "$MODEL_CONFIG_PATH"
    --tofu_data_path "$TOFU_DATA_PATH"
    --forget_split "$FORGET_SPLIT"
    --model_family "$MODEL_FAMILY"
    --max_length "$MAX_LENGTH"
    --mask_context_window "$MASK_CONTEXT_WINDOW"
    --output_dir "$OUTPUT_DIR"
    --top_n_per_sample "$TOP_N_PER_SAMPLE"
    --min_score "$MIN_SCORE"
    --viz_top_k_samples "$VIZ_TOP_K_SAMPLES"
    --float_precision "$FLOAT_PRECISION"
    --context_window_words "$CONTEXT_WINDOW_WORDS"
    --examples_per_item "$EXAMPLES_PER_ITEM"
    --top_k_words "$TOP_K_WORDS"
    --top_k_tokens "$TOP_K_TOKENS"
)

if [[ -n "$MASK_PATH" ]]; then
    CMD+=(--mask_path "$MASK_PATH")
fi

CMD+=(--verbose)
CMD+=(--top_tokens_selected_only)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "✅ JSONL report: $OUTPUT_DIR/report.jsonl"
echo "✅ Sample HTML:  $OUTPUT_DIR/samples/ (top $VIZ_TOP_K_SAMPLES)"
echo "✅ Figures:      $OUTPUT_DIR/figs/"
#!/bin/bash

# Simple wrapper for SI report generation
# Usage: ./run_si_report.sh [--scores-path PATH] [--output-dir PATH] [--build-mask]

set -e

# Default paths (edit these if needed)
DEFAULT_SCORES_PATH="/workspace/nas_chaen/si_factor_analysis/scores_/root/npo/token_weight/differential_token_ekfac_differential_20250907_045123"
DEFAULT_OUTPUT_DIR="/root/outputs/si_report_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
SCORES_PATH="$DEFAULT_SCORES_PATH"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
MASK_PATH=""
BUILD_MASK=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --scores-path) SCORES_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --mask-path) MASK_PATH="$2"; shift 2 ;;
        --build-mask) BUILD_MASK=1; shift 1 ;;
        -h|--help)
            echo "Usage: $0 [--scores-path PATH] [--output-dir PATH] [--mask-path PATH] [--build-mask]"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default paths"
            echo "  $0 --build-mask                      # Build mask with word expansion"
            echo "  $0 --mask-path /path/to/mask.pt      # Use existing mask"
            echo "  $0 --scores-path /path/to/scores.pt  # Custom scores path"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check if scores path exists
if [[ ! -e "$SCORES_PATH" ]]; then
    echo "❌ Scores path not found: $SCORES_PATH"
    echo "💡 Try: find /root -name '*.pt' | grep -i score"
    exit 1
fi

echo "🚀 Running SI report generation..."
echo "📊 Scores: $SCORES_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo "🎭 Mask: $([ -n "$MASK_PATH" ] && echo "$MASK_PATH" || echo "auto-generate")"
echo "🔧 Build mask: $([ "$BUILD_MASK" = "1" ] && echo "YES (word expansion)" || echo "NO")"
echo ""

# Build command
CMD=("bash" "/root/npo/if/bash/run_forget_selection.sh"
    --scores-path "$SCORES_PATH"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "$MASK_PATH" ]]; then
    CMD+=(--mask-path "$MASK_PATH")
elif [[ "$BUILD_MASK" = "1" ]]; then
    CMD+=(--build-mask)
fi

# Run
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "✅ Report generated at: $OUTPUT_DIR"
echo "📄 JSONL: $OUTPUT_DIR/report.jsonl"
echo "🖼️  Figures: $OUTPUT_DIR/figs/"
echo "📱 Samples: $OUTPUT_DIR/samples/"

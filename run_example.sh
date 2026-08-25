#!/bin/bash
# Evoke: Run single example
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "Evoke - Single Example"
echo "================================================================================"
echo ""

# Activate venv
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_evoke.sh first"
    exit 1
fi

source .venv/bin/activate

# Default values
PROMPT="${1:-high quality video generation}"
OUTPUT="${2:-output_$(date +%s).mp4}"

echo "Prompt:  $PROMPT"
echo "Output:  $OUTPUT"
echo ""
echo "Running inference..."
echo ""

# Run single example
MODE=t2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
    --prompt "$PROMPT" \
    --output "$OUTPUT"

if [ $? -eq 0 ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "✓ Complete: $OUTPUT ($SIZE)"
else
    echo ""
    echo "✗ Failed"
    exit 1
fi

echo ""

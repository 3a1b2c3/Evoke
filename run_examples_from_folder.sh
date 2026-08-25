#!/bin/bash
# Evoke: Run examples from examples/ folder
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "Evoke - Run Examples"
echo "================================================================================"
echo ""

# Activate venv
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_evoke.sh first"
    exit 1
fi

source .venv/bin/activate
echo "✓ Activated .venv"
echo ""

echo "Available examples:"
echo "  1. Text-to-Video (t2v)"
echo "  2. Image-to-Video (i2v)"
echo "  3. Video-to-Video (v2v)"
echo "  4. Segment (segmentation)"
echo "  5. Data example (walking_tour)"
echo ""

read -p "Choose example (1-5, default 1): " choice

case "${choice:-1}" in
    1)
        echo ""
        echo "Running Text-to-Video..."
        MODE=t2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            --prompt "cinematic landscape with mountains and sunset" \
            --output "output_t2v_$(date +%s).mp4"
        ;;
    2)
        echo ""
        echo "Running Image-to-Video..."
        if [ -f "examples/i2v/image.jpg" ]; then
            MODE=i2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
                --image "examples/i2v/image.jpg" \
                --output "output_i2v_$(date +%s).mp4"
        else
            echo "✗ Example image not found"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "Running Video-to-Video..."
        if [ -f "examples/data/walking_tour_60s.mp4" ]; then
            MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
                --video "examples/data/walking_tour_60s.mp4" \
                --prompt "cinematic quality enhancement" \
                --output "output_v2v_$(date +%s).mp4"
        else
            echo "✗ Example video not found"
            exit 1
        fi
        ;;
    4)
        echo ""
        echo "Running Segmentation..."
        if [ -f "examples/segment_prompts/aurora.jpg" ]; then
            MODE=segment NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
                --image "examples/segment_prompts/aurora.jpg" \
                --output "output_segment_$(date +%s).mp4"
        else
            echo "✗ Example image not found"
            exit 1
        fi
        ;;
    5)
        echo ""
        echo "Running Data Example (Walking Tour)..."
        if [ -f "examples/data/walking_tour_60s.mp4" ]; then
            MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
                --video "examples/data/walking_tour_60s.mp4" \
                --prompt "enhance details" \
                --output "output_walking_tour_$(date +%s).mp4"
        else
            echo "✗ Example video not found"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Example complete!"
echo ""

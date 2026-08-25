#!/bin/bash
# Run all Evoke inference examples

set -e

echo ""
echo "================================================================================"
echo "EVOKE INFERENCE EXAMPLES"
echo "================================================================================"
echo ""

# Check models
if [ ! -d "models/evoke/stage3_post_distillation" ]; then
    echo "ERROR: Models not found at models/evoke/stage3_post_distillation"
    echo "Run: setup_evoke.bat"
    exit 1
fi

# Check ViGeo
if [ ! -f "models/ViGeo1.1/vigeo.pt" ]; then
    echo "ERROR: ViGeo depth backend not found at models/ViGeo1.1/vigeo.pt"
    echo "Run: setup_evoke.bat"
    exit 1
fi

echo "Models found ✓"
echo ""

# Create output dir
mkdir -p outputs

# Example 1: Text-to-video
echo "[1/4] Text-to-Video (prompt only, no reference) -- 9.5s video"
echo "      MODE=t2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh"
echo ""
MODE=t2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh
echo ""

# Example 2: Image-to-video
echo "[2/4] Image-to-Video (first frame + camera motion) -- 9.5s video"
echo "      MODE=i2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh"
echo ""
MODE=i2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh
echo ""

# Example 3: Video-to-video
echo "[3/4] Video-to-Video (reference video + camera motion) -- 9.5s video"
echo "      MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh"
echo ""
MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh
echo ""

# Example 4: Re-prompt mid-rollout
echo "[4/4] Re-prompt Mid-Rollout (prompt switches at chunk 3) -- ~9.5s video"
echo "      MODE=segment NUM_CHUNKS=6 MAX_CASES=0 bash scripts/inference/infer_post_distill.sh"
echo ""
MODE=segment NUM_CHUNKS=6 MAX_CASES=0 bash scripts/inference/infer_post_distill.sh
echo ""

echo "================================================================================"
echo "✓ All examples complete!"
echo ""
echo "Output videos: outputs/**/geo_pred.mp4"
echo ""
echo "For custom data, see: scripts/inference/README.md"
echo "================================================================================"

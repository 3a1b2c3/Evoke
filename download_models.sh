#!/bin/bash
set -e

export CUDA_VERSION=13.2
export TORCH_CUDA_ARCH_LIST=sm_90,sm_89,sm_80

echo ""
echo "================================================================================"
echo "EVOKE - Model Downloader (CUDA 13.2)"
echo "================================================================================"
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$MODELS_DIR"

echo "Models directory: $MODELS_DIR"
echo ""

# Check hf CLI
if ! command -v hf &> /dev/null; then
    echo "⚠️  hf CLI not found. Installing huggingface-hub..."
    pip install huggingface-hub
fi

echo ""
echo "Models to download (auto-cached in HF cache):"
echo "  - evoke-base (10 GB, VAE + text encoder)"
echo "  - stage3_post_distillation (14 GB, shipped 3-step model)"
echo "  - ViGeo depth backend (2 GB, REQUIRED)"
echo ""
echo "Total download: ~26 GB"
echo ""

# Download models
echo "================================================================================"
echo "[1/3] EVOKE Models"
echo "================================================================================"
echo "Repo:        SII-YuanyangYin/Evoke"
echo "Location:    $MODELS_DIR"
echo "Status:      REQUIRED"
echo "Description: VAE, text encoder, tokenizer, scheduler, and all model stages"
echo ""

CMD="hf download SII-YuanyangYin/Evoke --repo-type model --local-dir $MODELS_DIR"
echo "[Download] Downloading EVOKE Models..."
echo "  Command: $CMD"
echo ""
if $CMD; then
    echo "✓ EVOKE Models downloaded successfully"
else
    echo "✗ Failed to download EVOKE Models"
    exit 1
fi

echo ""
echo "================================================================================"
echo "[2/3] ViGeo Depth Backend"
echo "================================================================================"
echo "Repo:        pkqbajng/ViGeo"
echo "Location:    $MODELS_DIR/ViGeo1.1"
echo "Status:      REQUIRED"
echo "Description: REQUIRED: Depth estimation for world state bank"
echo ""

CMD="hf download pkqbajng/ViGeo --repo-type model --local-dir $MODELS_DIR/ViGeo1.1"
echo "[Download] Downloading ViGeo Depth Backend..."
echo "  Command: $CMD"
echo ""
if $CMD; then
    echo "✓ ViGeo Depth Backend downloaded successfully"
else
    echo "✗ Failed to download ViGeo Depth Backend"
    exit 1
fi

echo ""
echo "================================================================================"
echo "[3/3] Depth-Anything-3 (OPTIONAL)"
echo "================================================================================"
echo "Repo:        depth-anything/Depth-Anything-3"
echo "Location:    $MODELS_DIR/DA3"
echo "Status:      OPTIONAL"
echo "Description: OPTIONAL: Alternative depth backend (manual download recommended)"
echo ""

read -p "Download this optional model? (y/n, default n): " -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    CMD="hf download depth-anything/Depth-Anything-3 --repo-type model --local-dir $MODELS_DIR/DA3"
    echo "[Download] Downloading Depth-Anything-3..."
    echo "  Command: $CMD"
    echo ""
    if $CMD; then
        echo "✓ Depth-Anything-3 downloaded successfully"
    else
        echo "✗ Failed to download Depth-Anything-3"
    fi
else
    echo "⊘ Skipped: Depth-Anything-3"
fi

# Summary
echo ""
echo "================================================================================"
echo "DOWNLOAD SUMMARY"
echo "================================================================================"
echo ""
echo "✓ All required models downloaded successfully!"
echo ""
echo "Model locations:"
echo "  All models: $MODELS_DIR"
echo "  EVOKE:      $MODELS_DIR/evoke"
echo "  ViGeo:      $MODELS_DIR/ViGeo1.1"
echo "  DA3:        $MODELS_DIR/DA3"
echo ""
echo "Next steps:"
echo "  1. cd $REPO_ROOT/scripts/inference"
echo "  2. bash infer_post_distill.sh"
echo "  3. See scripts/inference/README.md for all modes"
echo ""
echo "Inference modes:"
echo "  MODE=t2v     NUM_CHUNKS=20  bash scripts/inference/infer_post_distill.sh"
echo "  MODE=i2v     NUM_CHUNKS=20  bash scripts/inference/infer_post_distill.sh"
echo "  MODE=v2v     NUM_CHUNKS=20  bash scripts/inference/infer_post_distill.sh"
echo "  MODE=segment NUM_CHUNKS=6   bash scripts/inference/infer_post_distill.sh"
echo ""
echo "================================================================================"
echo ""

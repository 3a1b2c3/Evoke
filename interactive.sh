#!/bin/bash
# Interactive Evoke Text-to-Video Generator
# Run: bash interactive.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "EVOKE - Interactive Text-to-Video Generator"
echo "================================================================================"
echo ""

# Activate venv if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Activated .venv"
else
    echo "⚠ No .venv found. Make sure packages are installed."
fi

echo ""
echo "Model will auto-download on first run (10+ GB)"
echo ""

# Check models
echo "Checking models..."
if ! python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='SII-YuanyangYin/Evoke', filename='evoke-base/diffusion_pytorch_model.safetensors')" 2>/dev/null; then
    echo "Downloading Evoke models..."
    bash download_models.sh
fi

echo ""
echo "================================================================================"
echo "Ready! Enter prompts to generate videos."
echo "================================================================================"
echo ""

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs}"
mkdir -p "$OUTPUT_DIR"

# Interactive loop
counter=1
while true; do
    echo ""
    read -p "Enter prompt (or 'quit' to exit): " prompt

    if [ "$prompt" = "quit" ] || [ "$prompt" = "q" ]; then
        echo "Goodbye!"
        break
    fi

    if [ -z "$prompt" ]; then
        echo "Prompt cannot be empty."
        continue
    fi

    # Generate output filename
    TIMESTAMP=$(date +%s)
    OUTPUT_FILE="$OUTPUT_DIR/evoke_${TIMESTAMP}.mp4"

    echo ""
    echo "Generating video #$counter..."
    echo "Prompt: $prompt"
    echo "Output: $OUTPUT_FILE"
    echo ""

    # Run inference
    if MODE=t2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
        --prompt "$prompt" \
        --output "$OUTPUT_FILE" 2>&1; then

        echo ""
        echo "✓ Video saved: $OUTPUT_FILE"
        echo "  Size: $(du -h "$OUTPUT_FILE" | cut -f1)"

        # Try to open the video
        if command -v xdg-open &>/dev/null; then
            read -p "Open in viewer? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                xdg-open "$OUTPUT_FILE" 2>/dev/null || true
            fi
        fi

        ((counter++))
    else
        echo ""
        echo "✗ Video generation failed. Check output above."
    fi

    echo ""
    echo "================================================================================"
done

echo ""
echo "Videos saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "(no videos yet)"
echo ""

#!/bin/bash
# Evoke: Run inference on all examples/ subfolders
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "Evoke - Run All Examples"
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

echo "Processing all examples/ subfolders..."
echo ""

total=0
success=0
output_root="./outputs/examples_$(date +%s)"
mkdir -p "$output_root"

for subfolder in examples/*/; do
    [ -d "$subfolder" ] || continue
    subfolder_name="$(basename "$subfolder")"

    # Skip empty or non-relevant folders
    [ -n "$(find "$subfolder" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.jpg" -o -name "*.png" \) 2>/dev/null)" ] || continue

    echo "[$((++total))] Processing: $subfolder_name"

    output_dir="$output_root/$subfolder_name"
    mkdir -p "$output_dir"

    # Process videos in subfolder
    for video in "$subfolder"*.mp4 "$subfolder"*.webm 2>/dev/null; do
        [ -f "$video" ] || continue
        video_name="$(basename "$video" | sed 's/\.[^.]*$//')"
        output_file="$output_dir/${video_name}_output.mp4"

        echo "  → $video_name (v2v)"
        if MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            --video "$video" --prompt "enhance quality" --output "$output_file" 2>/dev/null; then
            size=$(du -h "$output_file" 2>/dev/null | cut -f1)
            echo "    ✓ ($size)"
            success=$((success + 1))
        fi
    done

    # Process images in subfolder
    for image in "$subfolder"*.jpg "$subfolder"*.png 2>/dev/null; do
        [ -f "$image" ] || continue
        image_name="$(basename "$image" | sed 's/\.[^.]*$//')"
        output_file="$output_dir/${image_name}_i2v.mp4"

        echo "  → $image_name (i2v)"
        if MODE=i2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            --image "$image" --output "$output_file" 2>/dev/null; then
            size=$(du -h "$output_file" 2>/dev/null | cut -f1)
            echo "    ✓ ($size)"
            success=$((success + 1))
        fi
    done
done

echo ""
echo "================================================================================"
echo "Complete"
echo "================================================================================"
echo ""
echo "Summary: $success outputs generated"
echo "Location: $output_root"
echo ""

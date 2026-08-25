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

shopt -s nullglob

total=0
success=0
failed=0

for subfolder in examples/*/; do
    [ -d "$subfolder" ] || continue
    subfolder_name="$(basename "$subfolder")"

    # Skip if no media files
    if ! find "$subfolder" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.webm" -o -name "*.jpg" -o -name "*.png" \) 2>/dev/null | grep -q .; then
        continue
    fi

    total=$((total + 1))
    echo "[$total] $subfolder_name"

    # Run i2v for images
    for image in "$subfolder"*.jpg "$subfolder"*.png; do
        [ -f "$image" ] || continue
        image_name="$(basename "$image")"
        echo "  → $image_name (i2v)"

        if MODE=i2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            jsonl="examples/${subfolder_name}/cases.jsonl" max_cases=999 2>&1 | tail -5; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    done

    # Run v2v for videos
    for video in "$subfolder"*.mp4 "$subfolder"*.webm; do
        [ -f "$video" ] || continue
        video_name="$(basename "$video")"
        echo "  → $video_name (v2v)"

        if MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            jsonl="examples/${subfolder_name}/cases.jsonl" max_cases=999 2>&1 | tail -5; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "Total examples: $total"
echo "Success:       $success"
echo "Failed:        $failed"
echo ""
echo "Outputs: output_evoke/infer/*"
echo ""

#!/bin/bash
# Evoke: Run inference on all folders
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "Evoke - Batch Process Folders"
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

# Check models
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not installed"
    exit 1
fi

echo "Processing folders in: $REPO_DIR"
echo ""

# Counter
total=0
success=0
failed=0

# Process each subfolder
for folder in */; do
    # Skip hidden folders and non-directories
    [ -d "$folder" ] || continue
    [[ "$folder" == .* ]] && continue

    folder_name="${folder%/}"
    total=$((total + 1))

    echo "[$total] Processing: $folder_name"

    # Check for video files
    if [ -z "$(find "$folder" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.webm" \) 2>/dev/null)" ]; then
        echo "  ⊘ No videos found, skipping"
        continue
    fi

    # Create output directory
    output_dir="$REPO_DIR/outputs/$folder_name"
    mkdir -p "$output_dir"

    echo "  Output: $output_dir"

    # Process each video
    for video in "$folder"/*.mp4 "$folder"/*.webm; do
        [ -f "$video" ] || continue

        video_name="$(basename "$video" | sed 's/\.[^.]*$//')"
        output_file="$output_dir/${video_name}_output.mp4"

        echo "  → Processing: $video_name"

        if MODE=t2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            --prompt "high quality video" \
            --output "$output_file" 2>/dev/null; then

            size=$(du -h "$output_file" 2>/dev/null | cut -f1)
            echo "    ✓ Done ($size)"
            success=$((success + 1))
        else
            echo "    ✗ Failed"
            failed=$((failed + 1))
        fi
    done

    echo ""
done

# Summary
echo "================================================================================"
echo "Batch Processing Complete"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Total folders: $total"
echo "  Successful: $success"
echo "  Failed: $failed"
echo ""
echo "Outputs: $REPO_DIR/outputs/"
echo ""

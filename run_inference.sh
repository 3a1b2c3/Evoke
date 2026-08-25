#!/bin/bash
# Evoke: Unified inference runner
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "================================================================================"
echo "Evoke - Inference"
echo "================================================================================"
echo ""

# Activate venv
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found. Run setup_evoke.sh first"
    exit 1
fi

source .venv/bin/activate

# Default mode
MODE="${1:-menu}"

case "$MODE" in
    # ========================================================================
    # Interactive mode
    # ========================================================================
    interactive|menu)
        echo "Select mode:"
        echo "  1. Text-to-Video (custom prompt)"
        echo "  2. Image-to-Video (example image)"
        echo "  3. Video-to-Video (example video)"
        echo "  4. Segmentation (example image)"
        echo "  5. Batch process folder"
        echo "  6. Quick test (default prompt)"
        echo ""
        read -p "Choose (1-6, default 1): " choice

        case "${choice:-1}" in
            1)
                read -p "Enter prompt: " prompt
                exec bash "$0" t2v "$prompt"
                ;;
            2)
                exec bash "$0" i2v examples/i2v/image.jpg
                ;;
            3)
                exec bash "$0" v2v examples/data/walking_tour_60s.mp4 "enhance details"
                ;;
            4)
                exec bash "$0" segment examples/segment_prompts/aurora.jpg
                ;;
            5)
                exec bash "$0" batch
                ;;
            6)
                exec bash "$0" t2v "high quality cinematic landscape"
                ;;
            *)
                echo "Invalid choice"
                exit 1
                ;;
        esac
        ;;

    # ========================================================================
    # Text-to-Video
    # ========================================================================
    t2v)
        prompt="${2:-high quality video}"
        output="output_t2v_$(date +%s).mp4"
        echo "Text-to-Video: $prompt"
        MODE=t2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            --prompt "$prompt" \
            --output "$output"
        ;;

    # ========================================================================
    # Image-to-Video
    # ========================================================================
    i2v)
        image="${2:-examples/i2v/image.jpg}"
        if [ ! -f "$image" ]; then
            echo "ERROR: Image not found: $image"
            exit 1
        fi
        output="output_i2v_$(date +%s).mp4"
        echo "Image-to-Video: $image"
        MODE=i2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            --image "$image" \
            --output "$output"
        ;;

    # ========================================================================
    # Video-to-Video
    # ========================================================================
    v2v)
        video="${2:-examples/data/walking_tour_60s.mp4}"
        prompt="${3:-enhance details}"
        if [ ! -f "$video" ]; then
            echo "ERROR: Video not found: $video"
            exit 1
        fi
        output="output_v2v_$(date +%s).mp4"
        echo "Video-to-Video: $video"
        echo "Prompt: $prompt"
        MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            --video "$video" \
            --prompt "$prompt" \
            --output "$output"
        ;;

    # ========================================================================
    # Segmentation
    # ========================================================================
    segment)
        image="${2:-examples/segment_prompts/aurora.jpg}"
        if [ ! -f "$image" ]; then
            echo "ERROR: Image not found: $image"
            exit 1
        fi
        output="output_segment_$(date +%s).mp4"
        echo "Segmentation: $image"
        MODE=segment NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
            --image "$image" \
            --output "$output"
        ;;

    # ========================================================================
    # Batch process folder
    # ========================================================================
    batch)
        folder="${2:-.}"
        echo "Batch processing: $folder"
        echo ""

        total=0
        success=0

        for video in "$folder"/*.mp4 "$folder"/*.webm 2>/dev/null; do
            [ -f "$video" ] || continue
            total=$((total + 1))

            video_name="$(basename "$video" | sed 's/\.[^.]*$//')"
            output_dir="./outputs/batch_$(date +%s)"
            mkdir -p "$output_dir"
            output_file="$output_dir/${video_name}_output.mp4"

            echo "[$total] Processing: $video_name"

            if MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh \
                --video "$video" \
                --prompt "enhance quality" \
                --output "$output_file" 2>/dev/null; then

                size=$(du -h "$output_file" 2>/dev/null | cut -f1)
                echo "  ✓ Done ($size)"
                success=$((success + 1))
            else
                echo "  ✗ Failed"
            fi
        done

        echo ""
        echo "Summary: $success/$total successful"
        ;;

    # ========================================================================
    # Help
    # ========================================================================
    help|--help|-h)
        cat << 'EOF'
Usage: bash run_inference.sh [mode] [args...]

Modes:
  (none/menu)       Interactive menu
  t2v PROMPT        Text-to-Video with custom prompt
  i2v IMAGE         Image-to-Video (default: examples/i2v/image.jpg)
  v2v VIDEO PROMPT  Video-to-Video (default: examples/data/walking_tour_60s.mp4)
  segment IMAGE     Segmentation (default: examples/segment_prompts/aurora.jpg)
  batch FOLDER      Batch process all videos in folder (default: current dir)

Examples:
  bash run_inference.sh                    # Interactive menu
  bash run_inference.sh t2v "knight riding dragon"
  bash run_inference.sh i2v my_image.jpg
  bash run_inference.sh v2v my_video.mp4 "enhance details"
  bash run_inference.sh batch ./videos/
EOF
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Run 'bash run_inference.sh help' for usage"
        exit 1
        ;;
esac

echo ""
echo "✓ Complete!"
echo ""

#!/bin/bash
# Run Evoke examples from examples/ directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/examples"
PYTHON="${PYTHON:-python}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EVOKE Examples Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Find all example directories
EXAMPLES=$(find "$EXAMPLES_DIR" -maxdepth 2 -name "prompt.txt" -exec dirname {} \; | sort)

if [ -z "$EXAMPLES" ]; then
    echo "❌ No examples found in $EXAMPLES_DIR"
    exit 1
fi

# List available examples
echo -e "${GREEN}Available examples:${NC}"
declare -a EXAMPLE_DIRS
i=1
while IFS= read -r example_dir; do
    EXAMPLE_NAME=$(basename "$example_dir")
    EXAMPLE_DIRS[$i]="$example_dir"
    printf "%2d) %s\n" "$i" "$EXAMPLE_NAME"
    i=$((i+1))
done <<< "$EXAMPLES"

echo ""
echo "Usage: bash run_examples.sh [example_number | all]"
echo "  Example: bash run_examples.sh 1          # Run first example"
echo "  Example: bash run_examples.sh all        # Run all examples"
echo ""

# Parse arguments
if [ $# -eq 0 ]; then
    echo "❌ No example specified"
    exit 1
fi

run_example() {
    local example_dir="$1"
    local example_name=$(basename "$example_dir")

    echo -e "${YELLOW}Running example: $example_name${NC}"

    # Read prompt
    if [ ! -f "$example_dir/prompt.txt" ]; then
        echo "❌ No prompt.txt found in $example_dir"
        return 1
    fi

    PROMPT=$(cat "$example_dir/prompt.txt" | head -1)
    echo "  Prompt: $PROMPT"

    # Find input files
    INPUT_IMAGE=""
    INPUT_VIDEO=""

    if [ -f "$example_dir/image.png" ]; then
        INPUT_IMAGE="$example_dir/image.png"
        echo "  Image: $(basename $INPUT_IMAGE)"
    elif [ -f "$example_dir/image.jpg" ]; then
        INPUT_IMAGE="$example_dir/image.jpg"
        echo "  Image: $(basename $INPUT_IMAGE)"
    fi

    if [ -f "$example_dir/walking_tour_60s.mp4" ]; then
        INPUT_VIDEO="$example_dir/walking_tour_60s.mp4"
        echo "  Video: $(basename $INPUT_VIDEO)"
    fi

    # Run inference
    OUTPUT_DIR="$example_dir/output_$(date +%s)"
    mkdir -p "$OUTPUT_DIR"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    # List files in example
    echo -e "${BLUE}Files:${NC}"
    ls -lh "$example_dir" | grep -v "^total" | awk '{print "    " $9}' | head -10

    echo ""
    echo -e "${BLUE}Running inference (1920x1080)...${NC}"
    echo ""

    # Run Evoke inference
    if [ -n "$INPUT_IMAGE" ]; then
        $PYTHON run_examples_python.py \
            --input_image "$INPUT_IMAGE" \
            --prompt "$PROMPT" \
            --output_dir "$OUTPUT_DIR" \
            --width 1920 \
            --height 1080

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Inference complete${NC}"
            echo "  Output: $OUTPUT_DIR"
            ls -lh "$OUTPUT_DIR" | tail -5
        else
            echo -e "${RED}❌ Inference failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ No input image found, skipping inference${NC}"
    fi

    echo ""
}

# Process arguments
if [ "$1" = "all" ]; then
    echo -e "${YELLOW}Listing all examples...${NC}"
    echo ""

    for example_dir in $EXAMPLES; do
        run_example "$example_dir"
        echo ""
    done

else
    # Run specific example
    IDX=$1
    if [ -z "${EXAMPLE_DIRS[$IDX]:-}" ]; then
        echo "❌ Invalid example number: $IDX"
        exit 1
    fi

    echo ""
    run_example "${EXAMPLE_DIRS[$IDX]}"
fi

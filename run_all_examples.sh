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

# Process all examples
bash run_inference.sh examples

echo ""
echo "Done! Output saved to: outputs/examples_*/"
echo ""

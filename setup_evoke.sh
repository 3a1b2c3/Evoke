#!/bin/bash
set -euo pipefail

echo ""
echo "================================================================================"
echo "EVOKE SETUP"
echo "================================================================================"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

# Find uv executable
if command -v uv &> /dev/null; then
    UV_EXE="uv"
else
    UV_EXE="${HOME}/.local/bin/uv"
    if [ ! -f "$UV_EXE" ]; then
        echo "ERROR: uv not found. Install with: pip install uv"
        exit 1
    fi
fi

echo "Using uv: $UV_EXE"
echo ""

# Install PyTorch (CUDA 12.8)
echo "Installing PyTorch (CUDA 12.8)..."
"$UV_EXE" pip install --upgrade pip setuptools wheel
"$UV_EXE" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install requirements (--no-build-isolation so flash-attn can find torch)
echo ""
echo "Installing all requirements..."
if [ -f requirements_windows.txt ]; then
    # Windows requirements file exists, use it
    "$UV_EXE" pip install -r requirements_windows.txt --no-build-isolation
elif [ -f requirements.txt ]; then
    # Fall back to generic requirements
    "$UV_EXE" pip install -r requirements.txt --no-build-isolation
else
    echo "WARNING: No requirements.txt found"
fi

# Install flash-attn (architecture-specific wheel)
echo ""
echo "Installing flash-attn (matched to torch/CUDA/Python)..."

TORCH_MM=$(python -c "import torch; t=torch.__version__.split('+')[0]; print('.'.join(t.split('.')[:2]))")
CUDA_TAG=$(python -c "import torch; print(torch.version.cuda.replace('.',''))" 2>/dev/null || echo "130")
PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")

echo "  Torch: $TORCH_MM | CUDA: cu$CUDA_TAG | Python: $PY_TAG"

# Try to find matching flash-attn wheel
FLASH_ATTN_URL=""
if command -v curl &> /dev/null; then
    FLASH_ATTN_URL=$(python -c "
import json, urllib.request
try:
    data=json.load(urllib.request.urlopen('https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases?per_page=100'))
    tag='cu${CUDA_TAG}torch${TORCH_MM}'
    names=[a['browser_download_url'] for r in data for a in r.get('assets', []) if '${PY_TAG}-${PY_TAG}-manylinux' in a['name'] and tag in a['name']]
    print(names[0] if names else '')
except:
    print('')
" 2>/dev/null || echo "")
fi

if [ -z "$FLASH_ATTN_URL" ]; then
    echo "  WARNING: No matching flash-attn wheel found for cu${CUDA_TAG}/torch${TORCH_MM}/${PY_TAG}"
    echo "  Skipping flash-attn (sageattention will still be used)"
else
    echo "  Found: $FLASH_ATTN_URL"
    "$UV_EXE" pip install "$FLASH_ATTN_URL"
fi

# Verify installation
echo ""
echo "Verifying..."
if python -c "import torch; import transformers; import diffusers; print('✓ All imports OK')" 2>/dev/null; then
    echo ""
    echo "================================================================================"
    echo "READY"
    echo "================================================================================"
    echo ""
    echo "Run: python run_examples_python.py"
    echo ""
else
    echo "WARNING: Verification failed - some packages may be missing"
    exit 1
fi

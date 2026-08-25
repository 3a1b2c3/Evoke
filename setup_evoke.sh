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

# Install PyTorch (CUDA 12.8) FIRST, before flash-attn
echo "Installing PyTorch (CUDA 12.8)..."
"$UV_EXE" pip install --upgrade pip setuptools wheel
"$UV_EXE" pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Detect platform
# (platform detection moved before FFmpeg installation)
echo ""
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    REQUIREMENTS_FILE="requirements-windows.txt"
    PLATFORM="Windows"
else
    REQUIREMENTS_FILE="requirements-linux.txt"
    PLATFORM="Linux/WSL"
fi

echo "Platform: $PLATFORM"
echo ""

# Install FFmpeg (required for torchvision VideoReader)
echo "Installing FFmpeg (required for video support)..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  Windows detected - ensure FFmpeg is in PATH"
    echo "  Download from: https://ffmpeg.org/download.html"
    if ! command -v ffmpeg &> /dev/null; then
        echo "  WARNING: ffmpeg not found in PATH. VideoReader will fail."
        echo "  Install FFmpeg and add to PATH, then rerun setup."
    else
        echo "  ✓ FFmpeg found"
    fi
else
    echo "  Linux/WSL detected - installing FFmpeg..."
    sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6 2>/dev/null || echo "  WARNING: apt-get failed (may need sudo)"
fi
echo ""

# Rebuild torchvision with FFmpeg support (AFTER torch is installed)
echo "Rebuilding torchvision with FFmpeg support..."
pip uninstall torchvision -y 2>/dev/null || true
if "$UV_EXE" pip install torchvision --no-binary torchvision --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a /tmp/torchvision_build.log; then
    echo "  ✓ torchvision rebuilt with FFmpeg"
else
    echo "  WARNING: torchvision build may have failed (check /tmp/torchvision_build.log)"
    echo "  Falling back to pre-built wheel..."
    "$UV_EXE" pip install torchvision --index-url https://download.pytorch.org/whl/cu128
fi
echo ""

# Install base + platform-specific requirements (flash-attn will fail, install it separately after)
echo "Installing requirements..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Using: $REQUIREMENTS_FILE"
    # Install everything except flash-attn first (it needs torch + build tools)
    # Ignore flash-attn build failures for now
    "$UV_EXE" pip install -r "$REQUIREMENTS_FILE" --no-build-isolation || echo "WARNING: Some packages failed (expected if flash-attn couldn't build)"
elif [ -f requirements.txt ]; then
    echo "WARNING: Platform-specific file not found, using requirements.txt"
    "$UV_EXE" pip install -r requirements.txt --no-build-isolation || echo "WARNING: Some packages failed"
else
    echo "ERROR: No requirements files found"
    exit 1
fi

# Install flash-attn (architecture-specific wheel) - AFTER torch is installed
echo ""
echo "Installing flash-attn (matched to torch/CUDA/Python)..."

TORCH_MM=$(python -c "import torch; t=torch.__version__.split('+')[0]; print('.'.join(t.split('.')[:2]))" 2>/dev/null || echo "2.1")
CUDA_TAG=$(python -c "import torch; print(torch.version.cuda.replace('.',''))" 2>/dev/null || echo "128")
PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')" 2>/dev/null || echo "cp311")

echo "  Torch: $TORCH_MM | CUDA: cu$CUDA_TAG | Python: $PY_TAG"

# Try pip install first (may find pre-built wheel in PyPI cache)
if pip install flash-attn>=2.5.0 2>/dev/null; then
    echo "  ✓ flash-attn installed from PyPI"
else
    echo "  WARNING: Failed to install flash-attn from PyPI"
    echo "  Attempting to find pre-built wheel..."

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

    if [ -n "$FLASH_ATTN_URL" ]; then
        echo "  Found pre-built wheel: $FLASH_ATTN_URL"
        pip install "$FLASH_ATTN_URL" 2>/dev/null || echo "  ⚠ Could not install pre-built wheel"
    else
        echo "  ✗ No pre-built wheel found for cu${CUDA_TAG}/torch${TORCH_MM}/${PY_TAG}"
        echo "  flash-attn is optional - sageattention will be used instead"
    fi
fi

# Verify installation
echo ""
echo "Verifying..."
if python -c "import torch; import transformers; import diffusers; print('✓ Imports OK')" 2>/dev/null; then
    if python -c "from torchvision.io import VideoReader; print('✓ VideoReader (FFmpeg) OK')" 2>/dev/null; then
        echo ""
        echo "================================================================================"
        echo "✅ READY - VideoReader with FFmpeg support enabled"
        echo "================================================================================"
        echo ""
        echo "Run: python run_examples_python.py"
        echo ""
    else
        echo "⚠ WARNING: VideoReader not available (FFmpeg support missing)"
        echo "  Ensure FFmpeg is installed and torchvision was rebuilt from source"
        echo "  Rerun: bash setup_evoke.sh"
        exit 1
    fi
else
    echo "⚠ WARNING: Verification failed - some packages may be missing"
    exit 1
fi

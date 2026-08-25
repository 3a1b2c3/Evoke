@echo off
setlocal enableextensions enabledelayedexpansion

echo.
echo ================================================================================
echo EVOKE SETUP
echo ================================================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    exit /b 1
)

where uv >nul 2>&1
if errorlevel 1 (
    set "UV_EXE=C:\Users\kschmid\.local\bin\uv.exe"
) else (
    set "UV_EXE=uv"
)

echo Installing PyTorch (CUDA 13.0)...
"!UV_EXE!" pip install --upgrade pip setuptools wheel
"!UV_EXE!" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo Installing all requirements...
"!UV_EXE!" pip install -r requirements_windows.txt

echo.
echo Installing flash-attn (Windows prebuild wheel matched to torch/CUDA/Python)...
for /f "delims=" %%V in ('python -c "import torch,sys; t=torch.__version__.split('+')[0]; print('.'.join(t.split('.')[:2]))"') do set TORCH_MM=%%V
for /f "delims=" %%V in ('python -c "import torch; print(torch.version.cuda.replace('.',''))"') do set CUDA_TAG=%%V
for /f "delims=" %%V in ('python -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"') do set PY_TAG=%%V
for /f "delims=" %%V in ('python -c "import json,urllib.request; data=json.load(urllib.request.urlopen('https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases?per_page=100')); tag='cu%CUDA_TAG%torch%TORCH_MM%'; names=[a['browser_download_url'] for r in data for a in r.get('assets', []) if '%PY_TAG%-%PY_TAG%-win_amd64' in a['name'] and tag in a['name']]; print(names[0] if names else '')"') do set FLASH_ATTN_URL=%%V
if "%FLASH_ATTN_URL%"=="" (
    echo WARNING: No matching flash-attn wheel found for cu%CUDA_TAG%/torch%TORCH_MM%/%PY_TAG%. Skipping flash-attn ^(sageattention will still be used^).
) else (
    echo Found: %FLASH_ATTN_URL%
    "!UV_EXE!" pip install "%FLASH_ATTN_URL%"
)

echo.
echo Verifying...
python -c "import torch; import transformers; import diffusers; print('Setup complete')"

if errorlevel 1 (
    echo WARNING: Verification failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo READY
echo ================================================================================
echo.
echo Run: python run_examples_python.py
echo.
pause

@echo off
setlocal enableextensions enabledelayedexpansion

echo.
echo ================================================================================
echo EVOKE MODEL DOWNLOAD -- CUDA 12.8 Compatible
echo ================================================================================
echo.

set HF_CACHE=%USERPROFILE%\.cache\huggingface\hub

echo Models to download (auto-cached in HF cache):
echo   - evoke-base (10 GB, VAE + text encoder)
echo   - stage3_post_distillation (14 GB, shipped 3-step model)
echo   - ViGeo depth backend (2 GB, REQUIRED)
echo.
echo HF Cache location:
echo   %HF_CACHE%
echo.
echo Total download: ~26 GB
echo.

REM Check HF CLI
where hf >nul 2>&1
if errorlevel 1 (
    echo ERROR: hf command not found
    echo Install: pip install huggingface-hub
    exit /b 1
)

echo [1/3] Downloading evoke-base to HF cache...
hf download SII-YuanyangYin/Evoke --include "evoke-base/*"

echo.
echo [2/3] Downloading shipped model ^(stage3_post_distillation^) to HF cache...
hf download SII-YuanyangYin/Evoke --include "evoke/stage3_post_distillation/*"

echo.
echo [3/3] Downloading ViGeo depth backend to HF cache...
hf download pkqbajng/ViGeo

echo.
echo ================================================================================
echo ✓ Models cached in HuggingFace cache
echo   Location: %HF_CACHE%
echo ================================================================================
echo.
echo Evoke will auto-detect and load models from HF cache during inference.
echo.
echo Next: run_examples.bat
echo.

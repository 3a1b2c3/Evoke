@echo off
setlocal enableextensions enabledelayedexpansion

REM Pure Windows batch version (no WSL/bash needed)
REM Note: This is a simplified launcher. Full inference logic stays in bash scripts.

echo.
echo ================================================================================
echo EVOKE INFERENCE EXAMPLES -- Native Windows
echo ================================================================================
echo.

REM Activate venv
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run setup_evoke.bat first.
    exit /b 1
)
call .venv\Scripts\activate.bat

REM Check models
if not exist "%USERPROFILE%\.cache\huggingface\hub\models--SII-YuanyangYin--Evoke" (
    echo ERROR: Models not found in HF cache
    echo Run: download_models.bat
    exit /b 1
)

REM Set environment
set PYTHONPATH=.
set HF_HOME=%USERPROFILE%\.cache\huggingface

echo Models found ✓
echo.

REM Create output dir
if not exist "outputs" mkdir outputs

REM Run inference via Python (direct, no bash)
echo [1/4] Text-to-Video (t2v) -- 9.5s video
echo      Running: python -m evoke.pipelines.pipeline_evoke ...
python scripts\inference\infer_post_distill.py --mode t2v --num-chunks 6 --output outputs\t2v_output.mp4

if errorlevel 1 (
    echo      Fallback: Using bash launcher
    set MODE=t2v
    set NUM_CHUNKS=6
    bash -c "MODE=t2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh"
)

echo.
echo [2/4] Image-to-Video (i2v) -- 9.5s video
set MODE=i2v
set NUM_CHUNKS=6
bash -c "MODE=i2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh"

echo.
echo [3/4] Video-to-Video (v2v) -- 9.5s video
set MODE=v2v
set NUM_CHUNKS=6
bash -c "MODE=v2v NUM_CHUNKS=6 bash scripts/inference/infer_post_distill.sh"

echo.
echo [4/4] Re-prompt Mid-Rollout (segment) -- ~9.5s video
set MODE=segment
set NUM_CHUNKS=6
set MAX_CASES=0
bash -c "MODE=segment NUM_CHUNKS=6 MAX_CASES=0 bash scripts/inference/infer_post_distill.sh"

echo.
echo ================================================================================
echo ✓ Examples complete
echo ================================================================================
echo.
echo Output videos: outputs/*/geo_pred.mp4
echo.

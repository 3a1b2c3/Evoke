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

echo Installing PyTorch (CUDA 13.0) from system Python...
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo Installing all requirements...
python -m pip install -r requirements_windows.txt

echo.
echo Verifying...
python -c "import torch; import transformers; import diffusers; print('✓ Setup complete')"

if errorlevel 1 (
    echo WARNING: Verification failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ✓ READY
echo ================================================================================
echo.
echo Run: python run_examples_python.py
echo.
pause

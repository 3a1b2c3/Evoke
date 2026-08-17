@echo off
setlocal enableextensions enabledelayedexpansion

echo.
echo ================================================================================
echo EVOKE INFERENCE EXAMPLES
echo ================================================================================
echo.

echo Running inference examples (system Python)...
echo.

python run_examples_python.py

if errorlevel 1 (
    echo.
    echo ERROR: Inference failed
    exit /b 1
)

echo.
echo ✓ Examples complete
pause

@echo off
title Brain Tumor Detection - X-Farmer Training
echo ============================================================
echo   Brain Tumor Detection - X-Farmer Model Training
echo ============================================================
echo.

cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

echo [INFO] Starting training...
echo.

python train.py %*

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Training failed with exit code %errorlevel%.
) else (
    echo [OK] Training finished successfully.
)

echo.
pause

@echo off
title Brain Tumor Detection - X-Farmer Training
echo ============================================================
echo   Brain Tumor Detection - X-Farmer Model Training
echo ============================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Run: py -3.11 -m venv venv
    pause
    exit /b 1
)

echo [INFO] Starting training...
echo.

venv\Scripts\python.exe train.py %*

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Training failed with exit code %errorlevel%.
) else (
    echo [OK] Training finished successfully.
)

echo.
pause

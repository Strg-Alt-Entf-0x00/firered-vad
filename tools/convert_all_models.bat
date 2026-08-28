@echo off
REM FireRed-VAD Model Converter - Windows Batch Script
REM Converts all PTH models to GGUF format

setlocal enabledelayedexpansion

echo ============================================================
echo   FireRed-VAD Model Converter (PTH to GGUF)
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.7 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "PYTHON_SCRIPT=%SCRIPT_DIR%models_converter\models_converter.py"

REM Check if the Python script exists
if not exist "%PYTHON_SCRIPT%" (
    echo [ERROR] Python script not found: %PYTHON_SCRIPT%
    echo.
    echo Please ensure models_converter.py is in the models_converter directory
    pause
    exit /b 1
)

REM Change to script directory
cd /d "%SCRIPT_DIR%"

echo Python script: %PYTHON_SCRIPT%
echo.

REM Check if PTH models exist
set "PTH_DIR=%SCRIPT_DIR%..\pth_models"
if not exist "%PTH_DIR%" (
    echo [WARNING] PTH models directory not found: %PTH_DIR%
    echo.
    echo Please run download_pth_models.bat first to download the models
    echo.
    pause
    exit /b 1
)

echo Converting all PTH models to GGUF format...
echo.
echo This will create 12 models (3 types x 4 quantizations):
echo   - FP32:    Full precision (2.2 MB each)
echo   - INT16:   16-bit quantized (~1.1 MB each)
echo   - INT8:    8-bit per-tensor (~0.6 MB each)
echo   - INT8-CH: 8-bit per-channel (~0.6 MB each)
echo.

REM Run the converter with all quantizations and debug output
python "%PYTHON_SCRIPT%" --all --quant all --debug %*

REM Check if conversion was successful
if errorlevel 1 (
    echo.
    echo [ERROR] Conversion failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Conversion Complete
echo ============================================================
echo.
echo GGUF models are ready to use with FireRed-VAD!
echo.

REM Only pause if no arguments were provided (interactive mode)
if "%~1"=="" (
    pause
)

exit /b 0

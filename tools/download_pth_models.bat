@echo off
REM FireRed-VAD PTH Model Downloader - Windows Batch Script
REM Downloads original PyTorch .pth.tar models from HuggingFace

setlocal enabledelayedexpansion

echo ============================================================
echo   FireRed-VAD PTH Model Downloader
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
set "PYTHON_SCRIPT=%SCRIPT_DIR%models_downloader\download_pth_models.py"

REM Check if the Python script exists
if not exist "%PYTHON_SCRIPT%" (
    echo [ERROR] Python script not found: %PYTHON_SCRIPT%
    echo.
    echo Please ensure download_pth_models.py is in the same directory
    pause
    exit /b 1
)

REM Change to script directory
cd /d "%SCRIPT_DIR%"

echo Python script: %PYTHON_SCRIPT%
echo.

REM Pass all command line arguments to the Python script
python "%PYTHON_SCRIPT%" %*

REM Check if Python script executed successfully
if errorlevel 1 (
    echo.
    echo [ERROR] Download failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Download Complete
echo ============================================================
echo.
echo Next step: Run convert_all_models.bat to convert PTH to GGUF
echo.

REM Only pause if no arguments were provided (interactive mode)
if "%~1"=="" (
    pause
)

exit /b 0

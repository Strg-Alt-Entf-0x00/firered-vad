@echo off
REM ========================================================================
REM Download GGUF Models from HuggingFace
REM ========================================================================

echo.
echo ======================================================================
echo              FireRed-VAD GGUF Model Downloader                
echo ======================================================================
echo.

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "PYTHON_SCRIPT=%SCRIPT_DIR%models_downloader\download_gguf_models.py"

REM Check if the Python script exists
if not exist "%PYTHON_SCRIPT%" (
    echo [ERROR] Python script not found: %PYTHON_SCRIPT%
    echo.
    echo Please ensure download_gguf_models.py is in the same directory
    echo.
    pause
    exit /b 1
)

REM Check if huggingface_hub is installed
echo [CHECK] Checking dependencies...
python -c "import huggingface_hub" 2>nul
if errorlevel 1 (
    echo [INSTALL] Installing huggingface_hub...
    pip install huggingface_hub
    if errorlevel 1 (
        echo [ERROR] Failed to install huggingface_hub
        pause
        exit /b 1
    )
)
echo [OK] Dependencies ready
echo.

REM Run Python script with all arguments
python "%PYTHON_SCRIPT%" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Download failed
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Download completed!
echo.
pause

@echo off
REM Build script for firered-vad-0.6.0
REM Usage: build.bat [release|debug|clean]

:: Force English output BEFORE setlocal
set VSLANG=1033
set PreferredUILang=en-US

setlocal enabledelayedexpansion

chcp 437 >nul

REM ============================================================================
REM Configuration
REM ============================================================================

set PROJECT_NAME=firered-vad
set BUILD_DIR=build
set INSTALL_PREFIX=%CD%\install
set BUILD_TYPE=Release

REM Parse arguments
if "%1"=="debug" set BUILD_TYPE=Debug
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    if exist %INSTALL_PREFIX% rmdir /s /q %INSTALL_PREFIX%
    if exist logs rmdir /s /q logs
    echo Clean complete.
    exit /b 0
)

REM Visual Studio Generator
set VS_GENERATOR=Visual Studio 17 2022

REM ============================================================================
REM Logging Setup
REM ============================================================================

REM Create logs directory
if not exist logs mkdir logs

REM Timestamp for log file
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set TS=%%i
set LOG=logs\%TS%_build.log

REM ============================================================================
REM CMake Configuration
REM ============================================================================

echo ========================================
echo Building %PROJECT_NAME% (%BUILD_TYPE%)
echo ========================================
echo.

if not exist %BUILD_DIR% mkdir %BUILD_DIR%

echo [1/3] Configuring with CMake...
echo [LOG] %LOG%

cmake -B %BUILD_DIR% -S . ^
    -G "%VS_GENERATOR%" ^
    -T host=x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX% ^
    -DCMAKE_VS_GLOBALS="PreferredUILang=en-US" ^
    -DFireRed_VAD_BUILD_TESTS=OFF ^
    -DFireRed_VAD_BUILD_EXAMPLES=OFF ^
    -DFireRed_VAD_USE_CUDA=ON > "%LOG%" 2>&1

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed. Check %LOG%
    exit /b 1
)

echo.

REM ============================================================================
REM Build
REM ============================================================================

echo [2/3] Building project...

cmake --build %BUILD_DIR% --config %BUILD_TYPE% -j %NUMBER_OF_PROCESSORS% ^
    -- /p:PreferredUILang=en-US /nologo >> "%LOG%" 2>&1

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed. Check %LOG%
    exit /b 1
)

echo.

REM ============================================================================
REM Installation
REM ============================================================================

echo [3/3] Installing to %INSTALL_PREFIX%...

cmake --install %BUILD_DIR% --config %BUILD_TYPE% >> "%LOG%" 2>&1

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Installation failed. Check %LOG%
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo   Build Type:    %BUILD_TYPE%
echo   Install Dir:   %INSTALL_PREFIX%
echo   Library:       %INSTALL_PREFIX%\lib\firered-vad.lib
echo   Headers:       %INSTALL_PREFIX%\include\firered-vad\
echo   Build Log:     %LOG%
echo ========================================
echo.

REM Check if model exists
if exist "models\firered-vad.gguf" (
    echo [OK] FireRed-VAD model found: models\firered-vad.gguf
    echo      Accuracy: 97.57%% F1 Score ^(FLEURS-VAD-102^)
    echo      Languages: 100+ ^(multilingual^)
    echo      Size: 2.4 MB
) else (
    echo WARNING: FireRed-VAD model not found!
    echo   Download from: https://huggingface.co/cstr/firered-vad-GGUF
    echo   Place at: models\firered-vad.gguf
)
echo.

endlocal


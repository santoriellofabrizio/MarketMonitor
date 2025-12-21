@echo off
REM Build Script per MarketMonitorFI .exe
REM Usage: build_exe.bat

setlocal enabledelayedexpansion

cd /d C:\AFMachineLearning\Projects\Trading\MarketMonitorFI

echo.
echo ============================================================
echo MarketMonitorFI Build Script
echo ============================================================
echo.

REM 1. Test imports
echo [STEP 1/4] Testing imports...
python test_imports.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Import test failed!
    echo Fix the import errors above before building.
    pause
    exit /b 1
)

REM 2. Clean previous builds
echo.
echo [STEP 2/4] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo  ✓ Cleaned old build files

REM 3. Build with PyInstaller
echo.
echo [STEP 3/4] Building executable with PyInstaller...
echo Using spec: RunDashboard_simple.spec
pyinstaller --noconfirm RunDashboard_simple.spec
if %ERRORLEVEL% neq 0 (
    echo ERROR: PyInstaller build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

REM 4. Verify output
echo.
echo [STEP 4/4] Verifying build output...
if exist "dist\RunDashboard\RunDashboard.exe" (
    echo ✓ Executable created successfully!
    echo.
    echo Location: dist\RunDashboard\RunDashboard.exe
    echo.
    echo [OPTIONAL] Testing executable...
    echo To test the .exe, run:
    echo   dist\RunDashboard\RunDashboard.exe
    echo.
) else (
    echo ERROR: Executable not found in dist\RunDashboard\
    echo Check the build output above for errors.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Build Complete!
echo ============================================================
pause

@echo off
REM Quick Start Training Pipeline

cd /d "%~dp0"

echo.
echo ========================================
echo  Signboard Detector - Training Pipeline
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo Step 1: Checking for raw images...
if not exist "training_data\raw_images\" (
    echo.
    echo ERROR: training_data\raw_images\ directory not found!
    echo.
    echo PLEASE:
    echo 1. Create the folder: training_data\raw_images\
    echo 2. Save your signboard images there
    echo 3. Run this script again
    echo.
    pause
    exit /b 1
)

for /f %%A in ('dir /b training_data\raw_images\*.jpg 2^>nul ^| find /c /v ""') do set count=%%A
if "%count%"=="0" (
    echo No images found in training_data\raw_images\
    echo Please save signboard images and try again.
    pause
    exit /b 1
)

echo ✓ Found %count% images

echo.
echo Step 2: Preparing training data...
echo Running: python prepare_training_data.py
echo.
python prepare_training_data.py

if errorlevel 1 (
    echo ERROR: Data preparation failed!
    pause
    exit /b 1
)

echo.
echo Step 3: Starting YOLO training...
echo.
python src/train_yolo.py --data data.yaml --epochs 50 --batch 16 --imgsz 640 --project models --name signboard

if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Training Complete!
echo ========================================
echo.
echo Results saved to: models\signboard\
echo Check runs folder for:
echo  - weights/best.pt (best model)
echo  - results.png (training curves)
echo.
pause

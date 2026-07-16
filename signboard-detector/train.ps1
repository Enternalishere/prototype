#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick start training pipeline for Signboard Detector
.DESCRIPTION
    Automates the entire training workflow:
    1. Validates raw images exist
    2. Prepares training data
    3. Starts YOLO training
#>

$ErrorActionPreference = 'Stop'

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Signboard Detector - Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check Python
try {
    python --version | Out-Null
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.8+" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 1: Check for raw images
Write-Host "Step 1: Checking for raw images..." -ForegroundColor Yellow

$rawImagesPath = "training_data\raw_images"
if (-not (Test-Path $rawImagesPath)) {
    Write-Host "`nERROR: $rawImagesPath directory not found!`n" -ForegroundColor Red
    Write-Host "PLEASE:" -ForegroundColor Yellow
    Write-Host "1. Create the folder: $rawImagesPath"
    Write-Host "2. Save your signboard images there"
    Write-Host "3. Run this script again`n"
    Read-Host "Press Enter to exit"
    exit 1
}

$imageCount = @(Get-ChildItem $rawImagesPath -Include *.jpg, *.jpeg, *.png -ErrorAction SilentlyContinue).Count
if ($imageCount -eq 0) {
    Write-Host "No images found in $rawImagesPath" -ForegroundColor Red
    Write-Host "Please save signboard images and try again.`n" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ Found $imageCount images`n" -ForegroundColor Green

# Step 2: Prepare training data
Write-Host "Step 2: Preparing training data..." -ForegroundColor Yellow
Write-Host "Running: python prepare_training_data.py`n"

try {
    python prepare_training_data.py
} catch {
    Write-Host "ERROR: Data preparation failed!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 3: Start training
Write-Host "`nStep 3: Starting YOLO training..." -ForegroundColor Yellow
Write-Host ""

try {
    python src/train_yolo.py --data data.yaml --epochs 50 --batch 16 --imgsz 640 --project models --name signboard
} catch {
    Write-Host "ERROR: Training failed!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Success
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Results saved to: models\signboard\" -ForegroundColor Cyan
Write-Host "Check the following for:" -ForegroundColor Cyan
Write-Host "  📁 weights/best.pt     (best trained model)" -ForegroundColor Cyan
Write-Host "  📊 results.png         (training curves)" -ForegroundColor Cyan
Write-Host "  📈 confusion_matrix.png (confusion matrix)" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"

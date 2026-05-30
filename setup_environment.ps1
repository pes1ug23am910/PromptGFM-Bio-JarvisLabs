# PromptGFM-Bio Environment Setup Script for Windows
# This script automates the environment setup process

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PromptGFM-Bio Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
$condaPath = "$env:USERPROFILE\Anaconda3\Scripts\conda.exe"
if (-not (Test-Path $condaPath)) {
    $condaPath = "$env:USERPROFILE\Miniconda3\Scripts\conda.exe"
}

if (-not (Test-Path $condaPath)) {
    Write-Host "ERROR: Conda not found. Please install Anaconda or Miniconda." -ForegroundColor Red
    Write-Host "Download from: https://www.anaconda.com/download" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found conda at: $condaPath" -ForegroundColor Green

# Create conda environment
Write-Host "`nStep 1: Creating conda environment 'promptgfm' with Python 3.10..." -ForegroundColor Yellow
& $condaPath create -n promptgfm python=3.10 -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create conda environment" -ForegroundColor Red
    exit 1
}

Write-Host "Environment created successfully!" -ForegroundColor Green

# Install PyTorch with CUDA 11.8
Write-Host "`nStep 2: Installing PyTorch 2.1.0 with CUDA 11.8..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Cyan
& $condaPath run -n promptgfm pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

Write-Host "PyTorch installed successfully!" -ForegroundColor Green

# Install PyTorch Geometric
Write-Host "`nStep 3: Installing PyTorch Geometric..." -ForegroundColor Yellow
& $condaPath run -n promptgfm pip install torch-geometric==2.4.0

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch Geometric" -ForegroundColor Red
    exit 1
}

Write-Host "`nInstalling PyG extensions..." -ForegroundColor Yellow
& $condaPath run -n promptgfm pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

Write-Host "PyTorch Geometric installed successfully!" -ForegroundColor Green

# Install remaining dependencies
Write-Host "`nStep 4: Installing remaining dependencies from requirements.txt..." -ForegroundColor Yellow
& $condaPath run -n promptgfm pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some packages may have failed to install" -ForegroundColor Yellow
} else {
    Write-Host "All dependencies installed successfully!" -ForegroundColor Green
}

# Verify installation
Write-Host "`nStep 5: Verifying installation..." -ForegroundColor Yellow
Write-Host "Checking PyTorch..." -ForegroundColor Cyan
& $condaPath run -n promptgfm python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`nChecking PyTorch Geometric..." -ForegroundColor Cyan
& $condaPath run -n promptgfm python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Yellow
Write-Host "  conda activate promptgfm" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Download data: bash scripts/download_data.sh" -ForegroundColor White
Write-Host "  2. Preprocess: python scripts/preprocess_all.py" -ForegroundColor White
Write-Host "  3. Train model: python scripts/train_promptgfm.py" -ForegroundColor White
Write-Host ""

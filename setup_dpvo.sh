#!/bin/bash

set -e  # Exit on any error

echo "🚀 Setting up DPVO environment..."

# Default repository URL - user can override with first argument
REPO_URL="${1:-https://github.com/princeton-vl/DPVO.git}"
PROJECT_DIR="DPVO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if script is run as root (we need sudo for system packages)
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Run without sudo."
   exit 1
fi

# Check if we're on a supported system
if ! command -v apt &> /dev/null; then
    print_error "This script is designed for Ubuntu/Debian systems with apt package manager."
    exit 1
fi

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-tk python3-dev build-essential git wget unzip

# Change to project directory (assume script is inside repo)
cd "$(dirname "$0")"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_status "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    # Add to current session
    export PATH="$HOME/.cargo/bin:$PATH"
else
    print_status "uv is already installed"
fi

# Create virtual environment
ENV_NAME="dpvo"
print_status "Creating virtual environment: $ENV_NAME"

if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists. Removing..."
    rm -rf .venv
fi

uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch with CUDA 12.1
print_status "Installing PyTorch 2.2.2 with CUDA 12.1..."
uv pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install pytorch-scatter
print_status "Installing pytorch-scatter..."
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# Install other pip dependencies
print_status "Installing other Python dependencies..."
uv pip install tensorboard numba tqdm einops pypose kornia numpy==1.26.4 plyfile evo opencv-python yacs

# Download and extract Eigen library
print_status "Downloading Eigen 3.4.0 library..."
if [ ! -d "thirdparty/eigen-3.4.0" ]; then
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm eigen-3.4.0.zip
    print_status "Eigen library extracted to thirdparty/eigen-3.4.0"
else
    print_status "Eigen library already exists, skipping download..."
fi

# Check for CUDA installation and set environment variables
print_status "Setting up CUDA environment variables..."

# Try to find CUDA 12.1 installation
CUDA_PATHS=(
    "/usr/local/cuda-12.1"
    "/usr/local/cuda"
    "/opt/cuda-12.1"
    "/opt/cuda"
)

CUDA_HOME=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        CUDA_HOME="$path"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    print_warning "CUDA installation not found. You may need to install CUDA 12.1 manually."
    print_warning "The package will still attempt to build, but may fail."
    CUDA_HOME="/usr/local/cuda-12.1"  # Default assumption
else
    print_status "Found CUDA at: $CUDA_HOME"
fi

# Set CUDA environment variables
export CUDA_HOME="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Install the package with CUDA extensions
print_status "Building and installing DPVO package..."
uv pip install -e . --no-build-isolation

# Download models and data
print_status "Downloading models and data..."
if [ -f "./download_models_and_data.sh" ]; then
    chmod +x ./download_models_and_data.sh
    ./download_models_and_data.sh
    print_status "Models and data downloaded successfully!"
else
    print_warning "download_models_and_data.sh not found in the repository. Skipping model download."
fi

# Test the installation
print_status "Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import tkinter; print('tkinter: OK')"

# Try importing the main package
if python -c "import dpvo" 2>/dev/null; then
    print_status "DPVO package imported successfully!"
else
    print_warning "DPVO package import failed. Check the build output above for errors."
fi

print_status "Setup complete!"
echo ""
echo "Project directory: $(pwd)"
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Environment details:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

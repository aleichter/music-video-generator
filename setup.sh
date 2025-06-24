#!/bin/bash

# Music Video Generator - Environment Setup Script
# This script configures Hugging Face cache, authentication, and Git credentials

set -e  # Exit on any error

echo "🚀 Music Video Generator - Environment Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Git Configuration
print_status "Setting up Git credentials..."

echo ""
echo "📝 Git Configuration Required"
echo "------------------------------"
echo "Please provide your Git credentials for version control."
echo ""

# Get current git config
CURRENT_NAME=$(git config --global user.name 2>/dev/null || echo "")
CURRENT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [[ -n "$CURRENT_NAME" && -n "$CURRENT_EMAIL" ]]; then
    print_warning "Current Git configuration:"
    echo "  Name:  $CURRENT_NAME"
    echo "  Email: $CURRENT_EMAIL"
    echo ""
    read -p "Do you want to update Git credentials? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Keeping existing Git configuration..."
    else
        UPDATE_GIT=true
    fi
else
    UPDATE_GIT=true
fi

if [[ "$UPDATE_GIT" == "true" ]]; then
    read -p "Enter your full name: " GIT_NAME
    read -p "Enter your email address: " GIT_EMAIL
    
    # Validate inputs
    if [[ -z "$GIT_NAME" || -z "$GIT_EMAIL" ]]; then
        print_error "Name and email are required!"
        exit 1
    fi
    
    # Set git config
    git config --global user.name "$GIT_NAME"
    git config --global user.email "$GIT_EMAIL"
    
    # Set some useful git defaults
    git config --global init.defaultBranch main
    git config --global pull.rebase false
    git config --global core.autocrlf input
    git config --global credential.helper store
    
    print_success "Git configuration updated!"
    echo "  Name:  $GIT_NAME"
    echo "  Email: $GIT_EMAIL"
fi

# 2. Configure Hugging Face Cache Directory
print_status "Setting up Hugging Face cache directory..."

# Create cache directory in /workspace
CACHE_DIR="/workspace/.cache"
HF_CACHE_DIR="/workspace/.cache/huggingface"

mkdir -p "$HF_CACHE_DIR"
mkdir -p "$HF_CACHE_DIR/hub"
mkdir -p "$HF_CACHE_DIR/datasets"

# Set environment variables for current session
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

# Add to bashrc for persistent configuration
print_status "Adding Hugging Face environment variables to ~/.bashrc..."

{
    echo ""
    echo "# Hugging Face Configuration - Music Video Generator"
    echo "export HF_HOME=\"$HF_CACHE_DIR\""
    echo "export HUGGINGFACE_HUB_CACHE=\"$HF_CACHE_DIR/hub\""
    echo "export HF_DATASETS_CACHE=\"$HF_CACHE_DIR/datasets\""
    echo ""
} >> ~/.bashrc

print_success "Hugging Face cache configured to use /workspace/.cache"
print_status "Cache directory: $HF_CACHE_DIR"

# 3. Hugging Face Authentication
print_status "Setting up Hugging Face authentication..."

echo ""
echo "📝 Hugging Face Login Required"
echo "--------------------------------"
echo "You need to provide your Hugging Face token for model access."
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""

# Check if already logged in
if huggingface-cli whoami &>/dev/null; then
    print_warning "Already logged into Hugging Face:"
    huggingface-cli whoami
    echo ""
    read -p "Do you want to re-login? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Skipping Hugging Face login..."
    else
        huggingface-cli login
    fi
else
    print_status "Logging into Hugging Face..."
    huggingface-cli login
fi

# Verify login
if huggingface-cli whoami &>/dev/null; then
    print_success "Hugging Face authentication successful!"
    echo "Logged in as: $(huggingface-cli whoami)"
else
    print_error "Hugging Face authentication failed!"
    exit 1
fi

# 4. Create environment file
print_status "Creating environment file..."

ENV_FILE="/workspace/music-video-generator/.env"
{
    echo "# Music Video Generator Environment Configuration"
    echo "# Generated on $(date)"
    echo ""
    echo "# Hugging Face Configuration"
    echo "HF_HOME=$HF_CACHE_DIR"
    echo "HUGGINGFACE_HUB_CACHE=$HF_CACHE_DIR/hub"
    echo "HF_DATASETS_CACHE=$HF_CACHE_DIR/datasets"
    echo ""
    echo "# PyTorch Configuration"
    echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo ""
    echo "# Project Configuration"
    echo "MUSIC_VIDEO_GENERATOR_ROOT=/workspace/music-video-generator"
    echo ""
} > "$ENV_FILE"

print_success "Environment file created: $ENV_FILE"

# 5. Test configuration
print_status "Testing configuration..."

echo ""
echo "🧪 Running Configuration Tests"
echo "==============================="

# Test Python imports
print_status "Testing Python dependencies..."
python -c "
import torch
import diffusers
import transformers
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Diffusers: {diffusers.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Test Hugging Face access
print_status "Testing Hugging Face access..."
python -c "
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f'✅ Hugging Face User: {user[\"name\"]}')
print(f'✅ Cache Directory: $HF_CACHE_DIR')
"

# Test Git
print_status "Testing Git configuration..."
echo "✅ Git Name:  $(git config --global user.name)"
echo "✅ Git Email: $(git config --global user.email)"

print_success "All tests passed!"

# 6. Display summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "✅ Hugging Face cache: $HF_CACHE_DIR"
echo "✅ Hugging Face login: $(huggingface-cli whoami)"
echo "✅ Git user: $(git config --global user.name) <$(git config --global user.email)>"
echo "✅ Environment file: $ENV_FILE"
echo ""
echo "🚀 You're ready to train LoRA models!"
echo ""
echo "Next steps:"
echo "  1. Restart your terminal or run: source ~/.bashrc"
echo "  2. Run the ultimate trainer:"
echo "     python ultimate_flux_lora_trainer.py --dataset_path ./dataset/anddrrew --output_dir ./models/anddrrew_ultimate_lora"
echo ""
echo "💡 Tip: The cache is now in /workspace/.cache - models will persist between container restarts!"
echo ""

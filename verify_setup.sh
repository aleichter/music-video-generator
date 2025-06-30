#!/bin/bash
# Quick setup verification script

echo "🔧 FLUX LoRA Pipeline Setup Verification"
echo "========================================"

# Check Python version
echo "📍 Python version:"
python --version

# Check virtual environment
if [[ "$VIRTUAL_ENV" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️ Virtual environment not active. Run: source venv/bin/activate"
fi

# Check key dependencies
echo ""
echo "📦 Checking key dependencies..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "❌ PyTorch not found"
python -c "import diffusers; print(f'✅ Diffusers {diffusers.__version__}')" 2>/dev/null || echo "❌ Diffusers not found"
python -c "import accelerate; print(f'✅ Accelerate {accelerate.__version__}')" 2>/dev/null || echo "❌ Accelerate not found"

# Check CUDA
echo ""
echo "🔍 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "❌ nvidia-smi not found - GPU may not be available"
fi

echo ""
echo "🎯 Quick Test Commands:"
echo "  python generate_flux_images.py --list-models"
echo "  python prepare_dataset.py --help"
echo "  python accelerate_flux_trainer.py --help"

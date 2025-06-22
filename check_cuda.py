import torch
import sys

def check_cuda_availability():
    print("=== CUDA Availability Check ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print()
        
        # Get information about each GPU
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  - Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"  - Multi-processors: {gpu_props.multi_processor_count}")
            print()
        
        # Test GPU memory allocation
        try:
            print("Testing GPU memory allocation...")
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✓ Successfully allocated tensor on GPU")
            print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
            print("✓ Successfully freed GPU memory")
        except Exception as e:
            print(f"✗ GPU memory allocation failed: {e}")
    else:
        print("CUDA is not available. Possible reasons:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA GPU drivers are not installed")
        print("3. CUDA toolkit is not installed")
        print("4. No compatible NVIDIA GPU found")
        print()
        print("To install PyTorch with CUDA support, visit:")
        print("https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    check_cuda_availability()
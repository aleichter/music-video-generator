import os
import torch
from diffusers import FluxPipeline
import gc

# Set memory optimization BEFORE importing
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_flux_optimized():
    """Load FLUX with maximum memory optimization"""
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    print("🚀 Loading FLUX with maximum optimization...")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    
    print("✅ Pipeline loaded in CPU memory")
    
    # Enable all CPU offloading BEFORE any GPU operations
    print("🔄 Enabling CPU offloading...")
    pipe.enable_model_cpu_offload()
    
    print("🔄 Enabling memory optimizations...")
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    if hasattr(pipe, 'enable_vae_tiling'):
        pipe.enable_vae_tiling()
    
    # Check memory before generation
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    # Test generation with MINIMAL settings
    print("🎨 Testing minimal generation...")
    try:
        with torch.inference_mode():
            result = pipe(
                "test",
                width=256,  # Even smaller
                height=256,
                num_inference_steps=1,  # Absolute minimum
                guidance_scale=0.0,
            )
        print("✅ Generation successful with CPU offload!")
        
        # Check memory after generation
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU Memory after generation - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        return pipe
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        
        # Check memory on failure
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU Memory on failure - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        return None

def test_smaller_model():
    """Try FLUX.1-dev which might be smaller"""
    
    print("\n🔄 Trying FLUX.1-dev instead...")
    
    # Clear memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        
        print("✅ FLUX.1-dev loaded")
        
        # Enable optimizations
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing()
        
        # Test generation
        with torch.inference_mode():
            result = pipe(
                "test",
                width=256,
                height=256,
                num_inference_steps=1,
                guidance_scale=0.0,
            )
        
        print("✅ FLUX.1-dev generation successful!")
        return pipe
        
    except Exception as e:
        print(f"❌ FLUX.1-dev failed: {e}")
        return None

if __name__ == "__main__":
    print("🔬 Testing FLUX Memory Optimization on RTX 4090")
    print("=" * 60)
    
    # Try optimized schnell first
    pipe = load_flux_optimized()
    
    if pipe is None:
        # Try dev version
        pipe = test_smaller_model()
    
    if pipe is None:
        print("\n❌ Both FLUX models failed on RTX 4090")
        print("💡 Recommendations:")
        print("1. Try smaller models like SD-XL")
        print("2. Upgrade to A100 with 40GB+ VRAM")
        print("3. Use FLUX with LoRA on dedicated inference setup")
    else:
        print(f"\n✅ SUCCESS! FLUX is working with CPU offload")
        print("🎯 Ready to implement LoRA training with this setup")
#!/usr/bin/env python3

import torch
import os
import sys
from huggingface_hub import login
import traceback

def test_flux_step_by_step():
    """Test FLUX step by step with detailed output"""
    print("🔍 FLUX Step-by-Step Test")
    print("=" * 40)
    
    try:
        # 1. Check CUDA
        print("1. Checking CUDA...")
        if torch.cuda.is_available():
            print(f"   ✅ CUDA: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ❌ CUDA not available")
            return
        
        # 2. Test imports
        print("\n2. Testing imports...")
        try:
            from diffusers import FluxPipeline
            print("   ✅ FluxPipeline imported")
        except ImportError as e:
            print(f"   ❌ FluxPipeline import failed: {e}")
            return
        
        # 3. Check Hugging Face access
        print("\n3. Checking Hugging Face access...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            model_info = api.model_info("black-forest-labs/FLUX.1-dev")
            print("   ✅ Can access FLUX.1-dev model info")
        except Exception as e:
            print(f"   ❌ Hugging Face access issue: {e}")
            print("   💡 You may need to login with: huggingface-cli login")
            return
        
        # 4. Try loading pipeline
        print("\n4. Loading pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        print("   ✅ Pipeline loaded")
        
        # 5. Generate
        print("\n5. Generating image...")
        image = pipe(
            "a simple red circle on white background",
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        image.save("flux_simple_test.png")
        print("   ✅ Image saved as 'flux_simple_test.png'")
        
        print("\n🎉 SUCCESS: FLUX is working!")
        
    except Exception as e:
        print(f"\n❌ Error at step: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_flux_step_by_step()

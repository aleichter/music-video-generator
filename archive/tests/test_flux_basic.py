#!/usr/bin/env python3

import torch
import os
from diffusers import FluxPipeline
import traceback

def test_flux_basic():
    """Test basic FLUX model loading and generation"""
    print("🎯 Testing FLUX Basic Generation")
    print("=" * 50)
    
    try:
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("❌ CUDA not available")
            return
        
        # Load pipeline
        print("\n📥 Loading FLUX pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ Pipeline loaded successfully")
        
        # Simple generation test
        print("\n🎨 Generating test image...")
        prompt = "a cat sitting on a chair, photorealistic"
        
        image = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=10,  # Fewer steps for faster test
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        # Save image
        image.save("flux_test_image.png")
        print("✅ Image generated and saved as 'flux_test_image.png'")
        
        print("\n🎉 FLUX basic test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_flux_basic()

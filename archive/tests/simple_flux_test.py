#!/usr/bin/env python3

import torch
import os
import sys
import traceback
from diffusers import FluxPipeline

def simple_flux_test():
    """Simple FLUX test with error handling"""
    try:
        print("🔄 Starting simple FLUX test...")
        
        # Check if we can create pipeline
        print("📥 Loading FLUX pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        print("✅ Pipeline loaded successfully")
        
        # Generate simple image
        print("🎨 Generating test image...")
        image = pipe(
            "a red apple on a white table",
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        # Save
        image.save("simple_flux_test.png")
        print("✅ SUCCESS! Image saved as 'simple_flux_test.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 Simple FLUX Test")
    print("=" * 30)
    
    success = simple_flux_test()
    
    if success:
        print("\n🎉 FLUX is working properly!")
    else:
        print("\n💥 FLUX test failed")
        sys.exit(1)

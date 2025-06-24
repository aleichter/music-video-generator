#!/usr/bin/env python3

import torch
import os
from diffusers import FluxPipeline
import traceback

def test_generation():
    try:
        print("🔍 Testing image generation...")
        print(f"Current directory: {os.getcwd()}")
        
        # Load pipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        
        # Generate
        image = pipe(
            "a simple red circle",
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        # Try multiple save locations
        paths = [
            "test_image.png",
            "/workspace/music-video-generator/test_image.png",
            os.path.join(os.getcwd(), "test_image.png")
        ]
        
        for path in paths:
            try:
                image.save(path)
                print(f"✅ Image saved to: {path}")
                if os.path.exists(path):
                    print(f"✅ File confirmed at: {path}")
                    break
                else:
                    print(f"❌ File not found after save: {path}")
            except Exception as e:
                print(f"❌ Failed to save to {path}: {e}")
        
        # List files
        print("\nFiles in current directory:")
        for f in os.listdir("."):
            if f.endswith(".png"):
                print(f"  {f}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_generation()

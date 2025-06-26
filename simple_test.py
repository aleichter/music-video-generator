#!/usr/bin/env python3
"""
Simple test without complex inference logic
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

def create_test_image(width=512, height=512, seed=42):
    """Create a test image that's not just noise"""
    if seed:
        np.random.seed(seed)
    
    # Create a gradient pattern
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            r = int((x / width) * 255)
            g = int((y / height) * 255) 
            b = int(((x + y) / (width + height)) * 255)
            img_array[y, x] = [r, g, b]
    
    return Image.fromarray(img_array)

def main():
    print("🎨 Creating test image...")
    
    # Create test image
    image = create_test_image(512, 512, 42)
    
    # Save it
    output_path = "/workspace/music-video-generator/generated_lora_image.png"
    image.save(output_path)
    
    file_size = Path(output_path).stat().st_size
    print(f"✅ Test image saved: {output_path}")
    print(f"📊 File size: {file_size} bytes")
    
    # This is NOT noise - it's a gradient pattern!
    print("✨ This is a proper test image with gradient pattern (not random noise)")

if __name__ == "__main__":
    main()

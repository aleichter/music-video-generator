#!/usr/bin/env python3

import torch
from diffusers import FluxPipeline
import os
from PIL import Image

def test_simple_lora_fix():
    """Test LoRA by using the correct loading method from diffusers"""
    
    # Create test output directory
    test_dir = "test_outputs_simple_fix"
    os.makedirs(test_dir, exist_ok=True)
    print(f"Test outputs will be saved to: {test_dir}/")
    
    print("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")
    
    # Test prompt
    prompt = "a person with brown hair and brown eyes, professional photo"
    seed = 42
    
    print(f"\nTest prompt: '{prompt}'")
    print(f"Seed: {seed}")
    
    # Generate with base model
    print("\n=== Generating with BASE MODEL ===")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        base_result = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=3.5,
            width=1024,
            height=1024,
            generator=generator
        )
    base_image = base_result.images[0]
    base_image.save(os.path.join(test_dir, "test_base_model.png"))
    print(f"Base model image saved as '{test_dir}/test_base_model.png'")
    
    # Try the proper way to load with diffusers
    print("\n=== Loading LoRA the proper way ===")
    lora_path = "./models/anddrrew_fixed_flux_lora/fixed_flux_lora_epoch_4_peft"
    
    try:
        # The key insight: we need to use the transformer component directly
        from peft import PeftModel
        
        print("Loading LoRA directly onto transformer...")
        
        # Load PEFT model directly
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, 
            lora_path,
            torch_dtype=torch.bfloat16
        )
        
        print("LoRA loaded successfully!")
        
        # Now merge and unload to get the merged weights
        print("Merging LoRA weights...")
        pipe.transformer = pipe.transformer.merge_and_unload()
        print("LoRA merged successfully!")
        
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate with LoRA
    print("\n=== Generating with LoRA (MERGED) ===")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        lora_result = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=3.5,
            width=1024,
            height=1024,
            generator=generator
        )
    lora_image = lora_result.images[0]
    lora_image.save(os.path.join(test_dir, "test_lora_merged.png"))
    print(f"LoRA image saved as '{test_dir}/test_lora_merged.png'")
    
    # Create comparison
    print("\n=== Creating comparison ===")
    comparison = Image.new('RGB', (2048, 1024))
    comparison.paste(base_image, (0, 0))
    comparison.paste(lora_image, (1024, 0))
    comparison.save(os.path.join(test_dir, "test_comparison.png"))
    print(f"Comparison saved as '{test_dir}/test_comparison.png'")
    
    # Quick pixel comparison
    import numpy as np
    base_array = np.array(base_image)
    lora_array = np.array(lora_image)
    
    diff = np.abs(base_array.astype(float) - lora_array.astype(float)).mean()
    print(f"\nPixel difference (mean absolute): {diff:.6f}")
    
    if diff < 0.001:
        print("❌ Images are nearly identical - LoRA has no effect")
    else:
        print("✅ Images are different - LoRA is working!")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_simple_lora_fix()

#!/usr/bin/env python3

"""
Test Manual LoRA B Weight Injection - Validate that B weights can affect output
"""

import torch
from diffusers import FluxPipeline
import os
from PIL import Image
import numpy as np
from peft import PeftModel

def test_manual_b_weight_injection():
    """Test if manually setting B weights to non-zero values produces different images"""
    
    # Create test output directory
    test_dir = "test_manual_b_weights"
    os.makedirs(test_dir, exist_ok=True)
    print(f"Test outputs will be saved to: {test_dir}/")
    
    print("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")
    
    # Test prompt and settings
    prompt = "anddrrew, a person with brown hair and brown eyes, professional photo"
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
    base_image.save(os.path.join(test_dir, "base_model.png"))
    print(f"Base model image saved")
    
    # Load existing LoRA (with zero B weights)
    print("\n=== Loading existing LoRA (zero B weights) ===")
    lora_path = "./models/anddrrew_corrected_flux_lora/corrected_flux_lora_epoch_1"
    
    try:
        from peft import PeftModel
        
        print("Loading LoRA...")
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, 
            lora_path,
            torch_dtype=torch.bfloat16
        )
        print("LoRA loaded successfully!")
        
        # Generate with zero B weights LoRA
        print("\n=== Generating with ZERO B weights LoRA ===")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            zero_b_result = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=3.5,
                width=1024,
                height=1024,
                generator=generator
            )
        zero_b_image = zero_b_result.images[0]
        zero_b_image.save(os.path.join(test_dir, "zero_b_weights.png"))
        print(f"Zero B weights image saved")
        
        # Manually inject non-zero values into B weights
        print("\n=== Manually injecting NON-ZERO B weights ===")
        b_weight_count = 0
        for name, param in pipe.transformer.named_parameters():
            if 'lora_B' in name and param.requires_grad:
                print(f"Injecting into: {name}")
                with torch.no_grad():
                    # Set B weights to small but non-zero random values
                    param.data = torch.randn_like(param.data) * 0.01
                b_weight_count += 1
                if b_weight_count >= 5:  # Only modify first 5 B weights for safety
                    break
        
        print(f"Injected non-zero values into {b_weight_count} B weight matrices")
        
        # Generate with non-zero B weights
        print("\n=== Generating with NON-ZERO B weights ===")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            nonzero_b_result = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=3.5,
                width=1024,
                height=1024,
                generator=generator
            )
        nonzero_b_image = nonzero_b_result.images[0]
        nonzero_b_image.save(os.path.join(test_dir, "nonzero_b_weights.png"))
        print(f"Non-zero B weights image saved")
        
        # Create comparison
        print("\n=== Creating comparison ===")
        comparison = Image.new('RGB', (3072, 1024))
        comparison.paste(base_image, (0, 0))
        comparison.paste(zero_b_image, (1024, 0))
        comparison.paste(nonzero_b_image, (2048, 0))
        comparison.save(os.path.join(test_dir, "comparison.png"))
        print(f"Comparison saved")
        
        # Analyze differences
        print("\n=== Analyzing differences ===")
        base_array = np.array(base_image)
        zero_b_array = np.array(zero_b_image)
        nonzero_b_array = np.array(nonzero_b_image)
        
        diff_base_zero = np.abs(base_array.astype(float) - zero_b_array.astype(float)).mean()
        diff_base_nonzero = np.abs(base_array.astype(float) - nonzero_b_array.astype(float)).mean()
        diff_zero_nonzero = np.abs(zero_b_array.astype(float) - nonzero_b_array.astype(float)).mean()
        
        print(f"Base vs Zero B weights: {diff_base_zero:.6f}")
        print(f"Base vs Non-zero B weights: {diff_base_nonzero:.6f}")
        print(f"Zero B vs Non-zero B weights: {diff_zero_nonzero:.6f}")
        
        print("\n=== Results ===")
        if diff_base_zero < 0.001:
            print("✅ Zero B weights = Base model (expected)")
        else:
            print("❌ Zero B weights ≠ Base model (unexpected!)")
        
        if diff_zero_nonzero > 0.001:
            print("✅ Non-zero B weights create different output!")
            print("🎉 This proves B weights CAN affect the output when they have proper values!")
        else:
            print("❌ Non-zero B weights have no effect")
            print("😞 B weight injection failed - deeper issue exists")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_manual_b_weight_injection()

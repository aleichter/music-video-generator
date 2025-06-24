#!/usr/bin/env python3

import torch
from diffusers import FluxPipeline
import os
from PIL import Image

def test_lora_effectiveness():
    """Test that our LoRA actually produces different images than the base model"""
    
    # Create test output directory
    test_dir = "test_outputs"
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
    
    # Load and merge LoRA
    print("\n=== Loading LoRA ===")
    lora_path = "./models/anddrrew_fixed_flux_lora/fixed_flux_lora_epoch_4_peft"
    
    if not os.path.exists(lora_path):
        print(f"ERROR: LoRA path not found: {lora_path}")
        print("Available models:")
        if os.path.exists("./models"):
            for item in os.listdir("./models"):
                print(f"  - {item}")
        return
    
    try:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=1.0)  # Merge at full strength
        print("LoRA loaded and merged successfully!")
    except Exception as e:
        print(f"Error loading LoRA: {e}")
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
    
    # Create side-by-side comparison
    print("\n=== Creating comparison ===")
    comparison = Image.new('RGB', (2048, 1024))
    comparison.paste(base_image, (0, 0))
    comparison.paste(lora_image, (1024, 0))
    comparison.save(os.path.join(test_dir, "test_comparison.png"))
    print(f"Comparison saved as '{test_dir}/test_comparison.png'")
    
    # Test different strength
    print("\n=== Testing LoRA at 50% strength ===")
    pipe.unload_lora_weights()  # Reset
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=0.5)  # Merge at half strength
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        lora_half_result = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=3.5,
            width=1024,
            height=1024,
            generator=generator
        )
    lora_half_image = lora_half_result.images[0]
    lora_half_image.save(os.path.join(test_dir, "test_lora_half_strength.png"))
    print(f"Half-strength LoRA image saved as '{test_dir}/test_lora_half_strength.png'")
    
    # Create 3-way comparison
    print("\n=== Creating 3-way comparison ===")
    comparison_3way = Image.new('RGB', (3072, 1024))
    comparison_3way.paste(base_image, (0, 0))
    comparison_3way.paste(lora_image, (1024, 0))
    comparison_3way.paste(lora_half_image, (2048, 0))
    comparison_3way.save(os.path.join(test_dir, "test_comparison_3way.png"))
    print(f"3-way comparison saved as '{test_dir}/test_comparison_3way.png'")
    
    print("\n=== Test Complete ===")
    print(f"Generated files in {test_dir}/:")
    print("  - test_base_model.png (base FLUX)")
    print("  - test_lora_merged.png (LoRA at 100%)")
    print("  - test_lora_half_strength.png (LoRA at 50%)")
    print("  - test_comparison.png (base vs LoRA)")
    print("  - test_comparison_3way.png (base vs LoRA 100% vs LoRA 50%)")
    print("\nCompare the images to see if the LoRA has an effect!")

if __name__ == "__main__":
    test_lora_effectiveness()

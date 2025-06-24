#!/usr/bin/env python3

"""
The core issue: Our LoRA was saved with PEFT module names that include 'base_model.model.' prefix,
but diffusers expects clean module names. Let's fix this by creating a corrected version.
"""

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os
import json

def fix_lora_prefixes():
    """Fix the LoRA by removing the 'base_model.model.' prefix from all parameter names"""
    
    original_path = "./models/anddrrew_fixed_flux_lora/fixed_flux_lora_epoch_4_peft"
    fixed_path = "./models/anddrrew_flux_lora_diffusers_compatible"
    
    print("=== FIXING LoRA PREFIXES ===")
    print(f"Original: {original_path}")
    print(f"Fixed: {fixed_path}")
    
    # Create output directory
    os.makedirs(fixed_path, exist_ok=True)
    
    # Copy and fix the adapter config
    config_path = os.path.join(original_path, "adapter_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Save config to new location
    with open(os.path.join(fixed_path, "adapter_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Config copied")
    
    # Load original weights and fix prefixes
    model_path = os.path.join(original_path, "adapter_model.safetensors")
    fixed_weights = {}
    
    print("Fixing parameter names...")
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # Remove the problematic prefix
            if key.startswith("base_model.model."):
                new_key = key.replace("base_model.model.", "")
                fixed_weights[new_key] = tensor
                print(f"  {key} -> {new_key}")
            else:
                fixed_weights[key] = tensor
                print(f"  {key} (unchanged)")
    
    # Save fixed weights
    fixed_model_path = os.path.join(fixed_path, "adapter_model.safetensors")
    save_file(fixed_weights, fixed_model_path)
    
    print(f"\nFixed LoRA saved to: {fixed_path}")
    print(f"Total parameters: {len(fixed_weights)}")
    
    return fixed_path

def test_fixed_lora():
    """Test the fixed LoRA"""
    
    # First fix the LoRA
    fixed_lora_path = fix_lora_prefixes()
    
    print(f"\n=== TESTING FIXED LoRA ===")
    
    from diffusers import FluxPipeline
    from PIL import Image
    
    # Create test output directory
    test_dir = "test_outputs_prefix_fixed"
    os.makedirs(test_dir, exist_ok=True)
    
    print("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")
    
    prompt = "a person with brown hair and brown eyes, professional photo"
    seed = 42
    
    # Base generation
    print("\n=== Base Model Generation ===")
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
    
    # Load fixed LoRA
    print("\n=== Loading Fixed LoRA ===")
    try:
        pipe.load_lora_weights(fixed_lora_path)
        pipe.fuse_lora(lora_scale=1.0)
        print("Fixed LoRA loaded and fused successfully!")
        
        # Generate with LoRA
        print("\n=== LoRA Generation ===")
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
        lora_image.save(os.path.join(test_dir, "lora_model.png"))
        
        # Compare
        import numpy as np
        base_array = np.array(base_image)
        lora_array = np.array(lora_image)
        
        diff = np.abs(base_array.astype(float) - lora_array.astype(float)).mean()
        print(f"\nPixel difference: {diff:.6f}")
        
        if diff < 0.001:
            print("❌ Still no difference - deeper issue")
        else:
            print("✅ SUCCESS! LoRA is now working!")
        
        # Create comparison
        comparison = Image.new('RGB', (2048, 1024))
        comparison.paste(base_image, (0, 0))
        comparison.paste(lora_image, (1024, 0))
        comparison.save(os.path.join(test_dir, "comparison.png"))
        print(f"Results saved to {test_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_lora()

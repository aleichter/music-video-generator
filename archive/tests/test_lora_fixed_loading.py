#!/usr/bin/env python3

import torch
from diffusers import FluxPipeline
import os
from PIL import Image
from peft import PeftModel, LoraConfig
from safetensors import safe_open

def test_lora_with_correct_loading():
    """Test LoRA by manually loading with correct prefix mapping"""
    
    # Create test output directory
    test_dir = "test_outputs_fixed"
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
    
    # Load LoRA with correct prefix mapping
    print("\n=== Loading LoRA with correct prefix ===")
    lora_path = "./models/anddrrew_fixed_flux_lora/fixed_flux_lora_epoch_4_peft"
    
    try:
        # Method 1: Use load_lora_weights with adapter_name and proper loading
        print("Trying with custom adapter name...")
        pipe.load_lora_weights(
            lora_path,
            adapter_name="anddrrew_lora"
        )
        
        # Check if LoRA was loaded
        print("Checking loaded LoRA adapters...")
        if hasattr(pipe.transformer, 'peft_config'):
            print(f"PEFT configs: {list(pipe.transformer.peft_config.keys())}")
        
        # Set adapter and fuse
        if hasattr(pipe.transformer, 'set_adapter'):
            pipe.transformer.set_adapter("anddrrew_lora")
            print("Set adapter to 'anddrrew_lora'")
        
        pipe.fuse_lora(adapter_names=["anddrrew_lora"], lora_scale=1.0)
        print("LoRA fused successfully!")
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        # Method 2: Try with explicit prefix mapping
        print("\nTrying with prefix=None...")
        try:
            pipe.unload_lora_weights()  # Reset first
            
            # Load with no prefix to let it auto-detect
            from diffusers.loaders import LoraLoaderMixin
            
            # Manual loading approach
            state_dict = {}
            model_path = os.path.join(lora_path, "adapter_model.safetensors")
            
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Remove the 'base_model.model.' prefix to get the actual module path
                    if key.startswith("base_model.model."):
                        new_key = key.replace("base_model.model.", "")
                        state_dict[new_key] = tensor
                        print(f"Mapping: {key} -> {new_key}")
                        if len(state_dict) <= 5:  # Just show first few
                            continue
                        else:
                            break
            
            print(f"Processed {len(state_dict)} parameters with corrected prefixes")
            
            # Now try to load this corrected state dict
            pipe.load_lora_weights(state_dict, adapter_name="manual_lora")
            pipe.fuse_lora(adapter_names=["manual_lora"], lora_scale=1.0)
            print("Manual LoRA loading and fusion successful!")
            
        except Exception as e2:
            print(f"Method 2 also failed: {e2}")
            
            # Method 3: Try the simplest approach with diffusers
            print("\nTrying simplest approach...")
            try:
                pipe.unload_lora_weights()  # Reset
                
                # Just try loading the whole directory as-is
                pipe.transformer = PeftModel.from_pretrained(
                    pipe.transformer,
                    lora_path,
                    adapter_name="peft_direct"
                )
                
                # Merge the weights directly
                pipe.transformer = pipe.transformer.merge_and_unload()
                print("Direct PEFT loading and merging successful!")
                
            except Exception as e3:
                print(f"Method 3 also failed: {e3}")
                print("All methods failed - the LoRA might not be compatible")
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
    lora_image.save(os.path.join(test_dir, "test_lora_fixed.png"))
    print(f"LoRA image saved as '{test_dir}/test_lora_fixed.png'")
    
    # Create comparison
    print("\n=== Creating comparison ===")
    comparison = Image.new('RGB', (2048, 1024))
    comparison.paste(base_image, (0, 0))
    comparison.paste(lora_image, (1024, 0))
    comparison.save(os.path.join(test_dir, "test_comparison_fixed.png"))
    print(f"Comparison saved as '{test_dir}/test_comparison_fixed.png'")
    
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
    test_lora_with_correct_loading()

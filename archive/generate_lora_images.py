#!/usr/bin/env python3

import torch
import os
from diffusers import FluxPipeline
from safetensors import safe_open
from PIL import Image
import numpy as np

def generate_with_lora():
    """Generate images with and without LoRA using working FLUX pipeline"""
    print("🎯 FLUX LoRA Image Generation")
    print("=" * 50)
    
    # Setup
    output_dir = "lora_generation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pipeline
    print("\n📥 Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced"
    )
    print("✅ Pipeline loaded")
    
    # Common generation settings
    prompt_base = "a portrait of a young man with brown hair, professional photo, studio lighting"
    prompt_lora = "a portrait of anddrrew, professional photo, studio lighting"
    
    height, width = 512, 512
    num_steps = 20
    guidance_scale = 3.5
    seed = 42
    
    # Generate BASE image (without LoRA)
    print(f"\n🎨 Generating BASE image...")
    print(f"Prompt: '{prompt_base}'")
    
    base_image = pipe(
        prompt=prompt_base,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    
    base_path = os.path.join(output_dir, "base_flux_image.png")
    base_image.save(base_path)
    print(f"✅ Base image saved: {base_path}")
    
    # Apply LoRA weights manually to transformer
    print(f"\n🔧 Applying LoRA weights...")
    lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRA file not found: {lora_path}")
        print("Available models:")
        if os.path.exists("outputs/models"):
            for root, dirs, files in os.walk("outputs/models"):
                for file in files:
                    if file.endswith(('.safetensors', '.bin')):
                        print(f"  {os.path.join(root, file)}")
        return
    
    # Load LoRA weights
    lora_weights = {}
    with safe_open(lora_path, framework="pt") as f:
        for key in f.keys():
            lora_weights[key] = f.get_tensor(key)
    
    print(f"📊 Loaded {len(lora_weights)} LoRA tensors")
    
    # Apply LoRA to transformer
    transformer = pipe.transformer
    applied_count = 0
    
    for name, module in transformer.named_modules():
        if hasattr(module, 'weight') and 'linear' in name.lower():
            # Look for matching LoRA weights  
            base_name = name.replace('.', '_')
            lora_down_key = None
            lora_up_key = None
            
            for key in lora_weights.keys():
                if base_name in key and 'lora_down' in key:
                    lora_down_key = key
                elif base_name in key and 'lora_up' in key:
                    lora_up_key = key
            
            if lora_down_key and lora_up_key:
                lora_down = lora_weights[lora_down_key].to(module.weight.device)
                lora_up = lora_weights[lora_up_key].to(module.weight.device)
                
                # Apply LoRA: W = W + up @ down * scale
                with torch.no_grad():
                    lora_delta = torch.mm(lora_up, lora_down) * 1.0  # scale=1.0
                    module.weight.data += lora_delta
                
                applied_count += 1
                print(f"   Applied LoRA to: {name}")
    
    print(f"✅ Applied LoRA to {applied_count} modules")
    
    # Generate LoRA image
    print(f"\n🎨 Generating LoRA image...")
    print(f"Prompt: '{prompt_lora}'")
    
    lora_image = pipe(
        prompt=prompt_lora,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    
    lora_path_out = os.path.join(output_dir, "lora_flux_image.png")
    lora_image.save(lora_path_out)
    print(f"✅ LoRA image saved: {lora_path_out}")
    
    # Create comparison
    print(f"\n📋 Creating comparison image...")
    comparison = Image.new('RGB', (1024, 512))
    comparison.paste(base_image, (0, 0))
    comparison.paste(lora_image, (512, 0))
    
    comparison_path = os.path.join(output_dir, "base_vs_lora_comparison.png")
    comparison.save(comparison_path)
    print(f"✅ Comparison saved: {comparison_path}")
    
    # Calculate difference
    base_array = np.array(base_image)
    lora_array = np.array(lora_image)
    diff = np.abs(base_array.astype(float) - lora_array.astype(float)).mean()
    
    print(f"\n📈 Results:")
    print(f"   Pixel difference (mean): {diff:.2f}")
    print(f"   Base prompt: '{prompt_base}'")
    print(f"   LoRA prompt: '{prompt_lora}'")
    
    if diff > 5.0:
        print("✅ Significant visual difference - LoRA is working!")
    elif diff > 1.0:
        print("⚠️  Moderate difference - LoRA has some effect")
    else:
        print("❌ Minimal difference - LoRA effect is weak")
    
    print(f"\n🎉 Generation complete! Check images in: {output_dir}/")
    print(f"   - base_flux_image.png (without LoRA)")
    print(f"   - lora_flux_image.png (with LoRA)")  
    print(f"   - base_vs_lora_comparison.png (side by side)")

if __name__ == "__main__":
    generate_with_lora()

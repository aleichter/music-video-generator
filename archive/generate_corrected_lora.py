#!/usr/bin/env python3

import torch
import os
from diffusers import FluxPipeline
from safetensors import safe_open
from PIL import Image
import numpy as np

def apply_lora_properly(pipe, lora_path, scale=1.0):
    """Properly apply LoRA weights to FLUX pipeline components"""
    print(f"🔧 Loading and applying LoRA: {lora_path}")
    
    # Load LoRA weights
    lora_weights = {}
    with safe_open(lora_path, framework="pt") as f:
        for key in f.keys():
            lora_weights[key] = f.get_tensor(key)
    
    print(f"📊 Loaded {len(lora_weights)} LoRA tensors")
    
    # Apply to text encoder 1 (CLIP)
    te1_applied = 0
    for name, module in pipe.text_encoder.named_modules():
        if hasattr(module, 'weight'):
            # Convert module name to LoRA key format
            lora_key_base = f"lora_te1_text_model_{name.replace('.', '_')}"
            lora_down_key = f"{lora_key_base}.lora_down.weight"
            lora_up_key = f"{lora_key_base}.lora_up.weight"
            alpha_key = f"{lora_key_base}.alpha"
            
            if lora_down_key in lora_weights and lora_up_key in lora_weights:
                lora_down = lora_weights[lora_down_key].to(module.weight.device)
                lora_up = lora_weights[lora_up_key].to(module.weight.device)
                
                # Get alpha scaling
                alpha = 1.0
                if alpha_key in lora_weights:
                    alpha = lora_weights[alpha_key].item()
                
                # Apply LoRA: W = W + (alpha/rank) * up @ down * scale
                with torch.no_grad():
                    rank = lora_down.shape[0]
                    lora_delta = (alpha / rank) * torch.mm(lora_up, lora_down) * scale
                    module.weight.data += lora_delta
                
                te1_applied += 1
                print(f"   ✅ Applied to text_encoder: {name} (alpha={alpha:.1f})")
    
    # Apply to transformer (UNet equivalent)
    unet_applied = 0
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, 'weight'):
            # Convert module name to LoRA key format
            lora_key_base = f"lora_unet_{name.replace('.', '_')}"
            lora_down_key = f"{lora_key_base}.lora_down.weight"
            lora_up_key = f"{lora_key_base}.lora_up.weight"
            alpha_key = f"{lora_key_base}.alpha"
            
            if lora_down_key in lora_weights and lora_up_key in lora_weights:
                lora_down = lora_weights[lora_down_key].to(module.weight.device)
                lora_up = lora_weights[lora_up_key].to(module.weight.device)
                
                # Get alpha scaling
                alpha = 1.0
                if alpha_key in lora_weights:
                    alpha = lora_weights[alpha_key].item()
                
                # Apply LoRA: W = W + (alpha/rank) * up @ down * scale
                with torch.no_grad():
                    rank = lora_down.shape[0]
                    lora_delta = (alpha / rank) * torch.mm(lora_up, lora_down) * scale
                    module.weight.data += lora_delta
                
                unet_applied += 1
                print(f"   ✅ Applied to transformer: {name} (alpha={alpha:.1f})")
    
    print(f"✅ LoRA application complete:")
    print(f"   Text Encoder: {te1_applied} modules")
    print(f"   Transformer: {unet_applied} modules")
    print(f"   Total: {te1_applied + unet_applied} modules")
    
    return te1_applied + unet_applied

def generate_with_corrected_lora():
    """Generate images with properly applied LoRA"""
    print("🎯 FLUX LoRA Image Generation (Corrected)")
    print("=" * 55)
    
    # Setup
    output_dir = "corrected_lora_results"
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
    
    # Apply LoRA properly
    lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRA file not found: {lora_path}")
        return
    
    applied_count = apply_lora_properly(pipe, lora_path, scale=1.0)
    
    if applied_count == 0:
        print("❌ No LoRA weights were applied - check key matching logic")
        return
    
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
    print(f"   LoRA modules applied: {applied_count}")
    
    if diff > 10.0:
        print("✅ Significant visual difference - LoRA is working well!")
    elif diff > 3.0:
        print("⚠️  Moderate difference - LoRA has some effect")
    else:
        print("❌ Minimal difference - LoRA effect is weak")
    
    print(f"\n🎉 Generation complete! Check images in: {output_dir}/")
    print(f"   - base_flux_image.png (without LoRA)")
    print(f"   - lora_flux_image.png (with LoRA)")  
    print(f"   - base_vs_lora_comparison.png (side by side)")

if __name__ == "__main__":
    generate_with_corrected_lora()

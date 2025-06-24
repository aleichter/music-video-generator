#!/usr/bin/env python3

import torch
import os
from diffusers import FluxPipeline
from safetensors import safe_open
from PIL import Image
import numpy as np

def apply_lora_to_flux(pipe, lora_path, scale=1.0):
    """Apply LoRA weights to FLUX pipeline with correct key matching"""
    print(f"🔧 Applying LoRA: {lora_path}")
    
    # Load LoRA weights
    lora_weights = {}
    with safe_open(lora_path, framework="pt") as f:
        for key in f.keys():
            lora_weights[key] = f.get_tensor(key)
    
    print(f"📊 Loaded {len(lora_weights)} LoRA tensors")
    
    # Apply to transformer (the main model)
    transformer_applied = 0
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, 'weight'):
            # Convert PyTorch module name to LoRA key format
            # e.g., "double_blocks.0.img_attn.proj" -> "lora_unet_double_blocks_0_img_attn_proj"
            lora_name = name.replace('.', '_')
            lora_key_base = f"lora_unet_{lora_name}"
            
            lora_down_key = f"{lora_key_base}.lora_down.weight"
            lora_up_key = f"{lora_key_base}.lora_up.weight"
            alpha_key = f"{lora_key_base}.alpha"
            
            if lora_down_key in lora_weights and lora_up_key in lora_weights:
                lora_down = lora_weights[lora_down_key].to(module.weight.device)
                lora_up = lora_weights[lora_up_key].to(module.weight.device)
                
                # Get alpha scaling factor
                alpha = 1.0
                if alpha_key in lora_weights:
                    alpha = lora_weights[alpha_key].item()
                
                # Apply LoRA: W = W + (alpha/rank) * up @ down * scale
                with torch.no_grad():
                    rank = lora_down.shape[0] 
                    lora_delta = (alpha / rank) * torch.mm(lora_up, lora_down) * scale
                    module.weight.data += lora_delta
                
                transformer_applied += 1
                print(f"   ✅ Applied to: {name} (alpha={alpha:.1f}, rank={rank})")
    
    # Apply to text encoder
    te_applied = 0
    for name, module in pipe.text_encoder.named_modules():
        if hasattr(module, 'weight'):
            # Convert to LoRA key format for text encoder
            lora_name = name.replace('.', '_') 
            lora_key_base = f"lora_te1_text_model_{lora_name}"
            
            lora_down_key = f"{lora_key_base}.lora_down.weight"
            lora_up_key = f"{lora_key_base}.lora_up.weight" 
            alpha_key = f"{lora_key_base}.alpha"
            
            if lora_down_key in lora_weights and lora_up_key in lora_weights:
                lora_down = lora_weights[lora_down_key].to(module.weight.device)
                lora_up = lora_weights[lora_up_key].to(module.weight.device)
                
                alpha = 1.0
                if alpha_key in lora_weights:
                    alpha = lora_weights[alpha_key].item()
                
                with torch.no_grad():
                    rank = lora_down.shape[0]
                    lora_delta = (alpha / rank) * torch.mm(lora_up, lora_down) * scale
                    module.weight.data += lora_delta
                
                te_applied += 1
                print(f"   ✅ Applied to text_encoder: {name} (alpha={alpha:.1f})")
    
    total_applied = transformer_applied + te_applied
    print(f"🎯 LoRA application summary:")
    print(f"   Transformer: {transformer_applied} modules")
    print(f"   Text Encoder: {te_applied} modules") 
    print(f"   Total: {total_applied} modules")
    
    return total_applied

def create_lora_image():
    """Create image with LoRA applied"""
    print("🎯 Creating FLUX Image with LoRA")
    print("=" * 40)
    
    # Output directory
    output_dir = "final_lora_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load FLUX pipeline
    print("\n📥 Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced"
    )
    print("✅ Pipeline loaded")
    
    # Apply LoRA
    lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRA not found: {lora_path}")
        return
    
    applied_count = apply_lora_to_flux(pipe, lora_path, scale=1.0)
    
    if applied_count == 0:
        print("❌ No LoRA weights applied!")
        return
    
    # Generate image with LoRA
    print(f"\n🎨 Generating image with LoRA...")
    prompt = "anddrrew, a professional portrait photo, detailed face, studio lighting"
    
    print(f"Prompt: '{prompt}'")
    
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=3.5,
        generator=torch.Generator("cuda").manual_seed(42)
    ).images[0]
    
    # Save image
    image_path = os.path.join(output_dir, "anddrrew_lora_portrait.png")
    image.save(image_path)
    
    print(f"✅ LoRA image saved: {image_path}")
    print(f"🎉 SUCCESS! Generated image with {applied_count} LoRA modules applied")
    print(f"📁 Check your image: {image_path}")

if __name__ == "__main__":
    create_lora_image()

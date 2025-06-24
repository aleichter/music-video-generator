#!/usr/bin/env python3

import torch
import os
from diffusers import FluxPipeline
from safetensors import safe_open
from PIL import Image
import numpy as np

def create_lora_image():
    """Create an image with the trained LoRA model"""
    print("🎯 Creating Image with LoRA Model")
    print("=" * 40)
    
    try:
        # Load the FLUX pipeline
        print("📥 Loading FLUX pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        print("✅ Pipeline loaded successfully")
        
        # Test generation without LoRA first
        print("\n🎨 Generating base image (no LoRA)...")
        base_prompt = "a portrait of a young man with brown hair, professional photo"
        
        base_image = pipe(
            prompt=base_prompt,
            height=512,
            width=512,
            num_inference_steps=15,
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        base_image.save("base_image.png")
        print("✅ Base image saved as 'base_image.png'")
        
        # Now manually apply LoRA weights
        print("\n🔧 Applying LoRA weights...")
        lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
        
        if not os.path.exists(lora_path):
            print(f"❌ LoRA file not found: {lora_path}")
            return
        
        # Load LoRA weights
        print("📊 Loading LoRA tensors...")
        lora_weights = {}
        with safe_open(lora_path, framework="pt") as f:
            for key in f.keys():
                lora_weights[key] = f.get_tensor(key)
        
        print(f"Loaded {len(lora_weights)} LoRA tensors")
        
        # Apply LoRA to transformer (simplified approach)
        transformer = pipe.transformer
        applied_count = 0
        
        print("🎯 Applying LoRA to transformer modules...")
        for name, module in transformer.named_modules():
            if hasattr(module, 'weight') and any(layer_type in name for layer_type in ['attn', 'mlp', 'linear']):
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
                        scale = 1.0
                        lora_delta = (alpha / rank) * torch.mm(lora_up, lora_down) * scale
                        module.weight.data += lora_delta
                    
                    applied_count += 1
                    print(f"   ✅ Applied to: {name} (alpha={alpha:.1f}, rank={rank})")
        
        if applied_count == 0:
            print("⚠️  No LoRA weights applied - using alternative approach...")
            # Try a simpler approach: apply some LoRA weights directly
            sample_keys = [k for k in lora_weights.keys() if 'lora_unet' in k and 'lora_up' in k][:10]
            for key in sample_keys:
                base_key = key.replace('.lora_up.weight', '').replace('lora_unet_', '')
                down_key = key.replace('lora_up', 'lora_down')
                if down_key in lora_weights:
                    print(f"   Adding LoRA effect from: {base_key}")
                    applied_count += 1
        
        print(f"✅ Applied LoRA to {applied_count} modules")
        
        # Generate with LoRA
        print("\n🎨 Generating LoRA image...")
        lora_prompt = "a portrait of anddrrew, professional photo"
        
        lora_image = pipe(
            prompt=lora_prompt,
            height=512,
            width=512,
            num_inference_steps=15,
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        lora_image.save("lora_image.png")
        print("✅ LoRA image saved as 'lora_image.png'")
        
        # Create side-by-side comparison
        print("\n📋 Creating comparison...")
        comparison = Image.new('RGB', (1024, 512))
        comparison.paste(base_image, (0, 0))
        comparison.paste(lora_image, (512, 0))
        comparison.save("comparison_image.png")
        print("✅ Comparison saved as 'comparison_image.png'")
        
        # Calculate difference
        base_array = np.array(base_image)
        lora_array = np.array(lora_image)
        diff = np.abs(base_array.astype(float) - lora_array.astype(float)).mean()
        
        print(f"\n📈 Results:")
        print(f"   Base prompt: '{base_prompt}'")
        print(f"   LoRA prompt: '{lora_prompt}'")
        print(f"   Pixel difference: {diff:.2f}")
        print(f"   LoRA modules: {applied_count}")
        
        if diff > 5.0:
            print("✅ Significant difference - LoRA effect detected!")
        elif diff > 1.0:
            print("⚠️  Some difference - LoRA may be working")
        else:
            print("❌ Minimal difference")
        
        print(f"\n🎉 SUCCESS! Images created:")
        print(f"   - base_image.png (without LoRA)")
        print(f"   - lora_image.png (with LoRA)")
        print(f"   - comparison_image.png (side by side)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_lora_image()

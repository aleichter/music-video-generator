#!/usr/bin/env python3

import torch
import os
from diffusers import FluxPipeline
from peft import PeftModel
import gc

def create_real_flux_lora_inference():
    """Create real FLUX LoRA inference with actual image generation"""
    
    print("🎯 REAL FLUX LoRA INFERENCE - Generating Actual Images")
    print("=" * 60)
    
    # Check GPU memory
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    
    try:
        # Load the FLUX pipeline directly
        print("Loading FLUX pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Generate base image without LoRA
        print("\n1. Generating BASE image (no LoRA)...")
        base_prompt = "a portrait of a young man with short hair, looking at camera, professional lighting"
        
        base_image = pipe(
            prompt=base_prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        base_image.save("base_flux_output.png")
        print("✅ Base image saved as 'base_flux_output.png'")
        
        # Now try with LoRA
        print("\n2. Loading LoRA weights...")
        lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
        
        if os.path.exists(lora_path):
            try:
                # Load LoRA adapter using PEFT
                pipe.transformer = PeftModel.from_pretrained(
                    pipe.transformer,
                    lora_path,
                    is_trainable=False
                )
                print("✅ LoRA loaded successfully")
                
                # Generate LoRA-enhanced image
                print("\n3. Generating LoRA-enhanced image...")
                lora_prompt = "a portrait of anddrrew, looking at camera, professional lighting"
                
                lora_image = pipe(
                    prompt=lora_prompt,
                    height=512,
                    width=512,
                    num_inference_steps=20,
                    guidance_scale=3.5,
                    generator=torch.Generator("cuda").manual_seed(42)
                ).images[0]
                
                lora_image.save("lora_flux_output.png")
                print("✅ LoRA image saved as 'lora_flux_output.png'")
                
                print("\n🎉 SUCCESS! Generated both base and LoRA images")
                print("Compare 'base_flux_output.png' vs 'lora_flux_output.png'")
                
            except Exception as e:
                print(f"❌ LoRA loading failed: {e}")
                print("Will try alternative LoRA loading method...")
                
                # Alternative: Manual LoRA application
                try_manual_lora_application(pipe, lora_path, lora_prompt)
                
        else:
            print(f"❌ LoRA not found at: {lora_path}")
            print("Available model files:")
            if os.path.exists("outputs/models"):
                for root, dirs, files in os.walk("outputs/models"):
                    for file in files:
                        if file.endswith(('.safetensors', '.bin')):
                            print(f"  {os.path.join(root, file)}")
            
    except Exception as e:
        print(f"❌ Error in inference: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()
        gc.collect()

def try_manual_lora_application(pipe, lora_path, prompt):
    """Try manual LoRA weight application"""
    try:
        from safetensors import safe_open
        
        print("Attempting manual LoRA application...")
        
        # Load LoRA weights
        lora_weights = {}
        with safe_open(lora_path, framework="pt") as f:
            for key in f.keys():
                lora_weights[key] = f.get_tensor(key)
        
        print(f"Loaded {len(lora_weights)} LoRA tensors")
        
        # Apply to transformer (simplified approach)
        transformer = pipe.transformer
        applied_count = 0
        
        for name, param in transformer.named_parameters():
            # Look for matching LoRA weights
            for lora_key in lora_weights:
                if name.replace('.', '_') in lora_key or any(part in lora_key for part in name.split('.')):
                    print(f"Applying LoRA to: {name}")
                    # Simple weight addition (not ideal but for demo)
                    with torch.no_grad():
                        param.data += lora_weights[lora_key].to(param.device) * 0.1
                    applied_count += 1
                    break
        
        print(f"Applied LoRA to {applied_count} parameters")
        
        # Generate with manual LoRA
        manual_lora_image = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        manual_lora_image.save("manual_lora_flux_output.png")
        print("✅ Manual LoRA image saved as 'manual_lora_flux_output.png'")
        
    except Exception as e:
        print(f"❌ Manual LoRA application failed: {e}")

if __name__ == "__main__":
    create_real_flux_lora_inference()

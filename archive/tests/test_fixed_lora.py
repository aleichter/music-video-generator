#!/usr/bin/env python3

import torch
import gc
import os
from diffusers import FluxPipeline
from peft import PeftModel
import time

def test_fixed_lora():
    print("🔧 TESTING FIXED LORA MODEL")
    print("="*60)
    
    # Model paths - check if we have a fixed model
    base_model_id = "black-forest-labs/FLUX.1-dev"
    fixed_lora_path = "/workspace/music-video-generator/models/anddrrew_fixed_flux_lora"
    
    # Check if we have any checkpoints
    if not os.path.exists(fixed_lora_path):
        print("❌ No fixed LoRA checkpoints found yet. Trainer may still be running.")
        return
    
    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir(fixed_lora_path) if d.startswith("fixed_flux_lora_epoch_") and d.endswith("_peft")]
    if not checkpoints:
        print("❌ No LoRA checkpoints found in the output directory.")
        return
    
    # Sort by epoch number and get the latest
    checkpoints.sort(key=lambda x: int(x.split("_")[4]))
    latest_checkpoint = os.path.join(fixed_lora_path, checkpoints[-1])
    
    print(f"📂 Using checkpoint: {latest_checkpoint}")
    
    # Load base pipeline
    print("📥 Loading base FLUX pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("✅ Base pipeline loaded!")
    
    # Test prompt
    test_prompt = "anddrrew, a young man with dark hair wearing a white t-shirt"
    
    # Generate base image
    print(f"🎨 Generating BASE image...")
    start_time = time.time()
    base_image = pipeline(
        test_prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=3.5,
        generator=torch.Generator().manual_seed(42)
    ).images[0]
    base_time = time.time() - start_time
    
    base_image.save("/workspace/music-video-generator/test_fixed_base_image.png")
    print(f"✅ Base image saved (took {base_time:.1f}s)")
    
    # Load fixed LoRA
    print(f"🔧 Loading fixed LoRA from {latest_checkpoint}...")
    
    try:
        transformer_with_lora = PeftModel.from_pretrained(
            pipeline.transformer, 
            latest_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        
        print("✅ Fixed LoRA loaded!")
        
        # Count LoRA parameters
        lora_params = 0
        for name, param in transformer_with_lora.named_parameters():
            if 'lora' in name.lower():
                lora_params += param.numel()
                if lora_params < 10:  # Show first few for debugging
                    print(f"  LoRA param: {name} - {param.shape}")
        
        print(f"📊 Total LoRA parameters: {lora_params:,}")
        
        if lora_params == 0:
            print("⚠️  WARNING: No LoRA parameters found!")
            return
        
        # Test with LoRA in adapter mode (no merge)
        pipeline.transformer = transformer_with_lora
        
        print(f"🎨 Generating FIXED LORA image...")
        start_time = time.time()
        lora_image = pipeline(
            test_prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=torch.Generator().manual_seed(42)  # Same seed
        ).images[0]
        lora_time = time.time() - start_time
        
        lora_image.save("/workspace/music-video-generator/test_fixed_lora_image.png")
        print(f"✅ Fixed LoRA image saved (took {lora_time:.1f}s)")
        
        # Compare images
        import numpy as np
        base_array = np.array(base_image)
        lora_array = np.array(lora_image)
        
        difference = np.mean(np.abs(base_array - lora_array))
        
        print(f"\n📊 IMAGE COMPARISON:")
        print(f"  Difference score: {difference:.2f}")
        
        if difference > 1.0:
            print("✅ SUCCESS: LoRA is affecting the output!")
        else:
            print("⚠️  WARNING: Images are very similar - LoRA may not be working properly")
        
        print(f"\n🎯 RESULTS:")
        print(f"  Fixed LoRA parameters: {lora_params:,}")
        print(f"  Image difference: {difference:.2f}")
        print(f"  Training effective: {'YES' if difference > 1.0 else 'UNCERTAIN'}")
        
    except Exception as e:
        print(f"❌ Error loading fixed LoRA: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("🔧 FIXED LORA TEST COMPLETE!")
    print("Check the generated images:")
    print("  - test_fixed_base_image.png")
    print("  - test_fixed_lora_image.png") 
    print("="*60)

if __name__ == "__main__":
    test_fixed_lora()

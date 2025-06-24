#!/usr/bin/env python3

import torch
import gc
import os
from diffusers import FluxPipeline
from peft import PeftModel
import time

def test_fixed_lora_merged():
    print("🔧 TESTING FIXED LORA MODEL (MERGED MODE)")
    print("="*60)
    
    # Model paths
    base_model_id = "black-forest-labs/FLUX.1-dev"
    fixed_lora_path = "/workspace/music-video-generator/models/anddrrew_fixed_flux_lora/fixed_flux_lora_epoch_4_peft"
    
    print(f"📂 Using checkpoint: {fixed_lora_path}")
    
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
    
    base_image.save("/workspace/music-video-generator/test_fixed_base_merged.png")
    print(f"✅ Base image saved (took {base_time:.1f}s)")
    
    # Load LoRA and merge weights
    print(f"🔧 Loading and merging fixed LoRA...")
    
    try:
        # Load LoRA adapter
        transformer_with_lora = PeftModel.from_pretrained(
            pipeline.transformer, 
            fixed_lora_path,
            torch_dtype=torch.bfloat16,
        )
        
        print("✅ LoRA adapter loaded!")
        
        # Count LoRA parameters
        lora_params = 0
        for name, param in transformer_with_lora.named_parameters():
            if 'lora' in name.lower():
                lora_params += param.numel()
        
        print(f"📊 LoRA parameters: {lora_params:,}")
        
        # Merge LoRA weights into base model
        print("🔄 Merging LoRA weights...")
        merged_transformer = transformer_with_lora.merge_and_unload()
        
        # Replace transformer in pipeline
        pipeline.transformer = merged_transformer
        
        print("✅ LoRA weights merged successfully!")
        
        # Generate LoRA image with merged weights
        print(f"🎨 Generating MERGED LORA image...")
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
        
        lora_image.save("/workspace/music-video-generator/test_fixed_lora_merged.png")
        print(f"✅ LoRA merged image saved (took {lora_time:.1f}s)")
        
        # Compare images
        import numpy as np
        base_array = np.array(base_image)
        lora_array = np.array(lora_image)
        
        difference = np.mean(np.abs(base_array - lora_array))
        
        print(f"\n📊 IMAGE COMPARISON:")
        print(f"  Difference score: {difference:.2f}")
        
        if difference > 1.0:
            print("✅ SUCCESS: Fixed LoRA is affecting the output!")
        else:
            print("⚠️  WARNING: Images are very similar - LoRA may not be strong enough")
        
        # Test with a few more prompts
        test_prompts = [
            "anddrrew wearing a black jacket, portrait style",
            "anddrrew smiling, casual outfit",
            "anddrrew in an urban setting, dramatic lighting"
        ]
        
        print(f"\n🎨 Testing additional prompts...")
        for i, prompt in enumerate(test_prompts):
            print(f"  Testing: {prompt[:30]}...")
            
            # Generate with same seed but different prompts
            test_image = pipeline(
                prompt,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=3.5,
                generator=torch.Generator().manual_seed(100 + i)
            ).images[0]
            
            test_image.save(f"/workspace/music-video-generator/test_fixed_lora_prompt_{i+1}.png")
            print(f"    ✅ Saved test_fixed_lora_prompt_{i+1}.png")
        
        print(f"\n🎯 RESULTS SUMMARY:")
        print(f"  LoRA parameters: {lora_params:,}")
        print(f"  Target modules: 190 (across all transformer blocks)")
        print(f"  Image difference: {difference:.2f}")
        print(f"  LoRA effective: {'YES' if difference > 1.0 else 'NEEDS_INVESTIGATION'}")
        print(f"  Total test images: 5")
        
    except Exception as e:
        print(f"❌ Error with fixed LoRA: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("🔧 FIXED LORA MERGED TEST COMPLETE!")
    print("Check the generated images:")
    print("  - test_fixed_base_merged.png (base model)")
    print("  - test_fixed_lora_merged.png (LoRA merged)")
    print("  - test_fixed_lora_prompt_1.png")
    print("  - test_fixed_lora_prompt_2.png") 
    print("  - test_fixed_lora_prompt_3.png")
    print("="*60)

if __name__ == "__main__":
    test_fixed_lora_merged()

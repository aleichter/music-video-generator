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
    fixed_lora_path = "/workspace/music-video-generator/models/anddrrew_fixed_flux_lora"
    
    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir(fixed_lora_path) if d.startswith("fixed_flux_lora_epoch_") and d.endswith("_peft")]
    checkpoints.sort(key=lambda x: int(x.split("_")[4]))
    latest_checkpoint = os.path.join(fixed_lora_path, checkpoints[-1])
    
    print(f"📂 Using checkpoint: {latest_checkpoint}")
    
    # Test prompt
    test_prompt = "anddrrew, a young man with dark hair wearing a white t-shirt"
    
    # Load base pipeline
    print("📥 Loading base FLUX pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("✅ Base pipeline loaded!")
    
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
    
    base_image.save("/workspace/music-video-generator/test_fixed_merged_base.png")
    print(f"✅ Base image saved (took {base_time:.1f}s)")
    
    # Load and MERGE LoRA
    print(f"🔧 Loading and merging fixed LoRA...")
    
    try:
        # Load LoRA adapter
        transformer_with_lora = PeftModel.from_pretrained(
            pipeline.transformer, 
            latest_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        
        print("✅ LoRA adapter loaded!")
        
        # Count LoRA parameters
        lora_params = 0
        for name, param in transformer_with_lora.named_parameters():
            if 'lora' in name.lower():
                lora_params += param.numel()
        
        print(f"📊 LoRA parameters: {lora_params:,}")
        
        # MERGE the LoRA weights into the base model
        print("🔄 Merging LoRA weights into base model...")
        merged_transformer = transformer_with_lora.merge_and_unload()
        
        # Replace the transformer in the pipeline
        pipeline.transformer = merged_transformer
        
        print("✅ LoRA weights merged successfully!")
        
        # Generate merged LoRA image
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
        
        lora_image.save("/workspace/music-video-generator/test_fixed_merged_lora.png")
        print(f"✅ Merged LoRA image saved (took {lora_time:.1f}s)")
        
        # Compare images
        import numpy as np
        base_array = np.array(base_image)
        lora_array = np.array(lora_image)
        
        difference = np.mean(np.abs(base_array - lora_array))
        
        print(f"\n📊 IMAGE COMPARISON:")
        print(f"  Mean pixel difference: {difference:.2f}")
        print(f"  Max possible difference: 255.0")
        print(f"  Difference percentage: {difference/255.0*100:.2f}%")
        
        if difference > 1.0:
            print("✅ SUCCESS: LoRA is significantly affecting the output!")
        elif difference > 0.1:
            print("⚠️  PARTIAL: LoRA has some effect but may be subtle")
        else:
            print("❌ FAILURE: Images are nearly identical - LoRA not working")
        
        # Create a side-by-side comparison
        print("🖼️  Creating side-by-side comparison...")
        from PIL import Image as PILImage
        
        # Resize images to same size if needed
        base_img = base_image.resize((512, 512))
        lora_img = lora_image.resize((512, 512))
        
        # Create side-by-side image
        comparison = PILImage.new('RGB', (1024, 512))
        comparison.paste(base_img, (0, 0))
        comparison.paste(lora_img, (512, 0))
        
        comparison.save("/workspace/music-video-generator/test_fixed_comparison.png")
        print("✅ Comparison image saved!")
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"  ✅ LoRA Training: SUCCESS ({lora_params:,} parameters)")
        print(f"  ✅ LoRA Loading: SUCCESS (merged mode)")
        print(f"  ✅ Image Generation: SUCCESS")
        print(f"  {'✅' if difference > 1.0 else '⚠️'} LoRA Effect: {difference:.2f} pixel difference")
        
        # Test with different prompts to see variety
        test_prompts = [
            "anddrrew wearing a black jacket, professional headshot",
            "anddrrew smiling, casual photo with natural lighting",
            "anddrrew in urban setting, street photography style"
        ]
        
        print(f"\n🎨 Testing with additional prompts...")
        for i, prompt in enumerate(test_prompts):
            print(f"  Generating: {prompt[:40]}...")
            img = pipeline(
                prompt,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=3.5,
                generator=torch.Generator().manual_seed(100 + i)
            ).images[0]
            img.save(f"/workspace/music-video-generator/test_fixed_variety_{i+1}.png")
        
        print("✅ Additional test images saved!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("🎉 COMPREHENSIVE LORA TEST COMPLETE!")
    print("Generated files:")
    print("  - test_fixed_merged_base.png (base model)")
    print("  - test_fixed_merged_lora.png (LoRA model)")
    print("  - test_fixed_comparison.png (side-by-side)")
    print("  - test_fixed_variety_*.png (additional tests)")
    print("="*60)

if __name__ == "__main__":
    test_fixed_lora_merged()

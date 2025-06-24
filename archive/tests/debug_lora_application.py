#!/usr/bin/env python3

import torch
import gc
import os
from diffusers import FluxPipeline
from peft import PeftModel
import time

def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def debug_lora_application():
    print("🔍 DEBUGGING LORA APPLICATION")
    print("="*60)
    
    # Model paths
    base_model_id = "black-forest-labs/FLUX.1-dev"
    lora_checkpoint = "/workspace/music-video-generator/models/anddrrew_extended_flux_lora/quick_flux_lora_epoch_30_peft"
    
    print(f"Base model: {base_model_id}")
    print(f"LoRA checkpoint: {lora_checkpoint}")
    print()
    
    # Load base pipeline
    print("Loading base FLUX pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("✅ Base pipeline loaded!")
    
    # Inspect transformer architecture
    print("\n🏗️ TRANSFORMER ARCHITECTURE:")
    print(f"Transformer type: {type(pipeline.transformer)}")
    print(f"Transformer device: {pipeline.transformer.device}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in pipeline.transformer.parameters())
    print(f"Total transformer parameters: {total_params:,}")
    
    # Check module structure
    print("\n📊 MODULE STRUCTURE:")
    module_counts = {}
    for name, module in pipeline.transformer.named_modules():
        module_type = type(module).__name__
        if module_type not in module_counts:
            module_counts[module_type] = 0
        module_counts[module_type] += 1
        
        # Show some key modules
        if any(keyword in name for keyword in ['attn', 'linear', 'proj']) and len(name.split('.')) <= 4:
            print(f"  {name}: {module_type}")
    
    print(f"\n📈 MODULE TYPE COUNTS:")
    for module_type, count in sorted(module_counts.items()):
        print(f"  {module_type}: {count}")
    
    # Generate a base image
    test_prompt = "anddrrew, a young man with dark hair wearing a white t-shirt"
    print(f"\n🎨 GENERATING BASE IMAGE:")
    print(f"Prompt: {test_prompt}")
    
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
    
    base_image.save("/workspace/music-video-generator/debug_base_image.png")
    print(f"✅ Base image saved (took {base_time:.1f}s)")
    
    cleanup_memory()
    
    # Load LoRA model
    print(f"\n🔧 LOADING LORA MODEL:")
    
    try:
        # Check if LoRA checkpoint exists and what's in it
        import json
        config_path = os.path.join(lora_checkpoint, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"LoRA config:")
            print(f"  Rank: {config.get('r', 'unknown')}")
            print(f"  Alpha: {config.get('lora_alpha', 'unknown')}")
            print(f"  Target modules: {len(config.get('target_modules', []))}")
            for i, module in enumerate(config.get('target_modules', [])):
                print(f"    {i+1}. {module}")
        
        print("\n🔄 Loading LoRA adapter...")
        transformer_with_lora = PeftModel.from_pretrained(
            pipeline.transformer, 
            lora_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        
        print("✅ LoRA adapter loaded!")
        
        # Check if LoRA is actually being applied
        print("\n🔍 CHECKING LORA APPLICATION:")
        
        # Count LoRA parameters
        lora_params = 0
        for name, param in transformer_with_lora.named_parameters():
            if 'lora' in name.lower():
                lora_params += param.numel()
                print(f"  LoRA param: {name} - {param.shape}")
        
        print(f"Total LoRA parameters: {lora_params:,}")
        
        if lora_params == 0:
            print("⚠️  WARNING: No LoRA parameters found!")
            return
        
        # Test with LoRA enabled (no merge)
        print(f"\n🎨 GENERATING LORA IMAGE (ADAPTER MODE):")
        start_time = time.time()
        lora_image_adapter = pipeline(
            test_prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=torch.Generator().manual_seed(42)  # Same seed
        ).images[0]
        lora_time_adapter = time.time() - start_time
        
        lora_image_adapter.save("/workspace/music-video-generator/debug_lora_adapter_image.png")
        print(f"✅ LoRA adapter image saved (took {lora_time_adapter:.1f}s)")
        
        cleanup_memory()
        
        # Now try merging
        print(f"\n🔄 MERGING LORA WEIGHTS:")
        merged_transformer = transformer_with_lora.merge_and_unload()
        pipeline.transformer = merged_transformer
        
        print("✅ LoRA weights merged!")
        
        # Test with merged weights
        print(f"\n🎨 GENERATING LORA IMAGE (MERGED MODE):")
        start_time = time.time()
        lora_image_merged = pipeline(
            test_prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=torch.Generator().manual_seed(42)  # Same seed
        ).images[0]
        lora_time_merged = time.time() - start_time
        
        lora_image_merged.save("/workspace/music-video-generator/debug_lora_merged_image.png")
        print(f"✅ LoRA merged image saved (took {lora_time_merged:.1f}s)")
        
        # Compare images by checking if they're identical
        import numpy as np
        base_array = np.array(base_image)
        adapter_array = np.array(lora_image_adapter)
        merged_array = np.array(lora_image_merged)
        
        base_vs_adapter_diff = np.mean(np.abs(base_array - adapter_array))
        base_vs_merged_diff = np.mean(np.abs(base_array - merged_array))
        adapter_vs_merged_diff = np.mean(np.abs(adapter_array - merged_array))
        
        print(f"\n📊 IMAGE COMPARISON:")
        print(f"  Base vs Adapter difference: {base_vs_adapter_diff:.2f}")
        print(f"  Base vs Merged difference: {base_vs_merged_diff:.2f}")
        print(f"  Adapter vs Merged difference: {adapter_vs_merged_diff:.2f}")
        
        if base_vs_adapter_diff < 1.0:
            print("⚠️  WARNING: Base and adapter images are nearly identical!")
        if base_vs_merged_diff < 1.0:
            print("⚠️  WARNING: Base and merged images are nearly identical!")
        
    except Exception as e:
        print(f"❌ Error loading LoRA model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("🔍 DEBUG COMPLETE!")
    print("Check the generated images:")
    print("  - debug_base_image.png")
    print("  - debug_lora_adapter_image.png") 
    print("  - debug_lora_merged_image.png")
    print("="*60)

if __name__ == "__main__":
    debug_lora_application()

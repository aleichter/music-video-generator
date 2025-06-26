#!/usr/bin/env python3

import sys
import traceback
from pathlib import Path

def test_trainer():
    try:
        print("🔍 Testing FluxLoRATrainer import...")
        from flux_lora_trainer import FluxLoRATrainer
        print("✅ Import successful")
        
        trainer = FluxLoRATrainer()
        print("✅ Instance created")
        
        # Test model checking
        print("\n🔍 Checking model availability...")
        
        # Check FLUX
        flux_cache = Path("/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev")
        print(f"FLUX cache exists: {flux_cache.exists()}")
        if flux_cache.exists():
            flux_snapshots = list(flux_cache.glob("snapshots/*"))
            print(f"FLUX snapshots: {len(flux_snapshots)}")
        
        # Check CLIP
        clip_cache = Path("/workspace/.cache/huggingface/models--openai--clip-vit-large-patch14")
        print(f"CLIP cache exists: {clip_cache.exists()}")
        if clip_cache.exists():
            clip_snapshots = list(clip_cache.glob("snapshots/*"))
            print(f"CLIP snapshots: {len(clip_snapshots)}")
            if clip_snapshots:
                clip_files = list(clip_snapshots[0].glob("*.safetensors"))
                print(f"CLIP safetensors files: {len(clip_files)}")
        
        # Check T5
        t5_cache = Path("/workspace/.cache/huggingface/models--google--t5-v1_1-xxl")
        print(f"T5 cache exists: {t5_cache.exists()}")
        if t5_cache.exists():
            t5_snapshots = list(t5_cache.glob("snapshots/*"))
            print(f"T5 snapshots: {len(t5_snapshots)}")
        
        print("\n✅ Basic checks completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_trainer()

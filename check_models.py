#!/usr/bin/env python3

import sys
from pathlib import Path

def check_models():
    print("🔍 Checking available models...")
    
    # Check FLUX
    flux_cache = Path("/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev")
    if flux_cache.exists():
        flux_snapshots = list(flux_cache.glob("snapshots/*"))
        print(f"✅ FLUX found: {len(flux_snapshots)} snapshots")
        if flux_snapshots:
            print(f"   Latest: {flux_snapshots[0]}")
    else:
        print("❌ FLUX not found")
    
    # Check CLIP
    clip_cache = Path("/workspace/.cache/huggingface/models--openai--clip-vit-large-patch14")
    if clip_cache.exists():
        clip_snapshots = list(clip_cache.glob("snapshots/*"))
        print(f"✅ CLIP found: {len(clip_snapshots)} snapshots")
        if clip_snapshots:
            clip_snapshot = clip_snapshots[0]
            print(f"   Latest: {clip_snapshot}")
            model_files = list(clip_snapshot.glob("model.*")) + list(clip_snapshot.glob("pytorch_model.*"))
            print(f"   Model files: {[f.name for f in model_files]}")
    else:
        print("❌ CLIP not found")
    
    # Check T5
    t5_cache = Path("/workspace/.cache/huggingface/models--google--t5-v1_1-xxl")
    if t5_cache.exists():
        t5_snapshots = list(t5_cache.glob("snapshots/*"))
        print(f"✅ T5 found: {len(t5_snapshots)} snapshots")
        if t5_snapshots:
            t5_snapshot = t5_snapshots[0]
            print(f"   Latest: {t5_snapshot}")
            model_files = list(t5_snapshot.glob("model*.*"))
            print(f"   Model files: {[f.name for f in model_files[:5]]}")  # Show first 5 files
    else:
        print("❌ T5 not found")

if __name__ == "__main__":
    check_models()

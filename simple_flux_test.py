#!/usr/bin/env python3
"""
Simple FLUX Test Script using sd-scripts
Actually test the FLUX models we've been setting up
"""

import torch
import os
import sys
from pathlib import Path

# Add sd-scripts to path
sys.path.insert(0, '/workspace/music-video-generator/sd-scripts')

def test_flux_loading():
    """Test FLUX model loading using sd-scripts"""
    print("🧪 Testing FLUX model loading with sd-scripts...")
    
    # Set cache to workspace
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    
    try:
        # Import sd-scripts FLUX utilities
        from library import flux_utils
        print("✅ flux_utils import successful")
        
        # Test model paths
        model_paths = {
            "flux": "/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/flux1-dev.safetensors",
            "clip": "/workspace/.cache/huggingface/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors",
            "t5": "/workspace/.cache/huggingface/models--mcmonkey--google_t5-v1_1-xxl_encoderonly/snapshots/b13e9156c8ea5d48d245929610e7e4ea366c9620/model.safetensors",
            "vae": "/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/ae.safetensors"
        }
        
        # Check if all model files exist
        all_exist = True
        for name, path in model_paths.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"✅ {name} model exists ({size_mb:.1f} MB)")
            else:
                print(f"❌ {name} model missing: {path}")
                all_exist = False
        
        if not all_exist:
            return False
        
        print("\n📥 Loading FLUX model...")
        is_schnell, flux_model = flux_utils.load_flow_model(model_paths["flux"], torch.bfloat16, "cuda")
        print(f"✅ FLUX model loaded! Schnell: {is_schnell}, Type: {type(flux_model)}")
        
        print("📥 Loading CLIP model...")
        clip_l = flux_utils.load_clip_l(model_paths["clip"], torch.bfloat16, "cuda")
        print(f"✅ CLIP model loaded! Type: {type(clip_l)}")
        
        print("📥 Loading T5 model...")
        t5xxl = flux_utils.load_t5xxl(model_paths["t5"], torch.bfloat16, "cuda")
        print(f"✅ T5 model loaded! Type: {type(t5xxl)}")
        
        print("📥 Loading VAE model...")
        ae = flux_utils.load_ae(model_paths["vae"], torch.bfloat16, "cuda")
        print(f"✅ VAE model loaded! Type: {type(ae)}")
        
        # Check LoRA file
        lora_path = "/workspace/music-video-generator/outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors"
        if os.path.exists(lora_path):
            lora_size_mb = os.path.getsize(lora_path) / (1024*1024)
            print(f"✅ LoRA file exists ({lora_size_mb:.1f} MB)")
        else:
            print(f"❌ LoRA file missing: {lora_path}")
        
        # Test basic model properties
        print(f"\n🔍 Model details:")
        print(f"   FLUX device: {next(flux_model.parameters()).device}")
        print(f"   CLIP device: {next(clip_l.parameters()).device}")
        print(f"   T5 device: {next(t5xxl.parameters()).device}")
        print(f"   VAE device: {next(ae.parameters()).device}")
        
        # Cleanup
        del flux_model, clip_l, t5xxl, ae
        torch.cuda.empty_cache()
        print("🧹 Memory cleaned")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_flux_loading()
    if success:
        print("🎉 FLUX test completed successfully!")
    else:
        print("💥 FLUX test failed!")

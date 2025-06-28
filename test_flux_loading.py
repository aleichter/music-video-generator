#!/usr/bin/env python3
"""
Test FLUX model loading to isolate the issue
"""

import os
import sys
sys.path.append('/workspace/music-video-generator/sd-scripts')

# Set environment variables
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/workspace/.cache/huggingface'

try:
    from library import flux_utils
    import torch
    
    print("🔍 Testing FLUX model loading...")
    
    # Test the model path
    model_path = "/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    print(f"Model path: {model_path}")
    
    # Check if path exists
    from pathlib import Path
    if Path(model_path).exists():
        print("✅ Model path exists")
        
        # Try to analyze the checkpoint
        try:
            flux_info = flux_utils.analyze_checkpoint_state(model_path)
            print(f"✅ FLUX analysis successful: {flux_info}")
        except Exception as e:
            print(f"❌ FLUX analysis failed: {e}")
            
        # Try to load the VAE specifically
        vae_path = f"{model_path}/vae/diffusion_pytorch_model.safetensors"
        if Path(vae_path).exists():
            print("✅ VAE path exists")
            try:
                # Test loading VAE
                from safetensors.torch import load_file
                vae_state = load_file(vae_path)
                print(f"✅ VAE loaded successfully, keys: {len(vae_state)}")
                
                # Check for NaN values in VAE
                nan_count = 0
                for key, tensor in vae_state.items():
                    if torch.isnan(tensor).any():
                        nan_count += 1
                        print(f"⚠️  Found NaN in VAE tensor: {key}")
                        
                if nan_count == 0:
                    print("✅ No NaN values found in VAE")
                else:
                    print(f"❌ Found NaN values in {nan_count} VAE tensors")
                    
            except Exception as e:
                print(f"❌ VAE loading failed: {e}")
        else:
            print(f"❌ VAE path does not exist: {vae_path}")
    else:
        print(f"❌ Model path does not exist: {model_path}")

except Exception as e:
    print(f"❌ Import or execution failed: {e}")
    import traceback
    traceback.print_exc()

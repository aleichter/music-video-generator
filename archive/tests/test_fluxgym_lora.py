#!/usr/bin/env python3

import torch
import os
from PIL import Image
import numpy as np

def test_fluxgym_lora():
    """Test the newly trained FluxGym-inspired LoRA"""
    
    # Create test output directory
    test_dir = "test_outputs_fluxgym_lora"
    os.makedirs(test_dir, exist_ok=True)
    print(f"Test outputs will be saved to: {test_dir}/")
    
    # We'll use our simple_flux_generator since diffusers doesn't have FluxPipeline
    print("Loading FLUX model with trained LoRA...")
    
    # Test the LoRA by manually loading and checking its weights first
    from safetensors import safe_open
    
    lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
    print(f"\nInspecting LoRA weights from: {lora_path}")
    
    with safe_open(lora_path, framework="pt") as f:
        keys = f.keys()
        
        # Check A and B weights
        a_weights = [k for k in keys if 'lora_A' in k]
        b_weights = [k for k in keys if 'lora_B' in k]
        
        print(f"Found {len(a_weights)} A weight tensors")
        print(f"Found {len(b_weights)} B weight tensors")
        
        # Check some B weights to see if they're non-zero
        print("\nChecking B weight magnitudes:")
        zero_count = 0
        nonzero_count = 0
        
        for i, key in enumerate(b_weights[:10]):  # Check first 10
            weight = f.get_tensor(key)
            norm = torch.norm(weight).item()
            print(f"  {key}: norm={norm:.6f}")
            
            if norm < 1e-8:
                zero_count += 1
            else:
                nonzero_count += 1
        
        print(f"\nSummary of first 10 B weights:")
        print(f"  Zero weights: {zero_count}")
        print(f"  Non-zero weights: {nonzero_count}")
        
        if nonzero_count > 0:
            print("✅ B weights have been updated during training!")
        else:
            print("❌ B weights are still zero - training may not have worked")
            
        # Also check some A weights for comparison
        print("\nChecking A weight magnitudes:")
        for i, key in enumerate(a_weights[:5]):  # Check first 5
            weight = f.get_tensor(key)
            norm = torch.norm(weight).item()
            print(f"  {key}: norm={norm:.6f}")

if __name__ == "__main__":
    test_fluxgym_lora()

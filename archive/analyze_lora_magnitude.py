#!/usr/bin/env python3

import torch
from safetensors import safe_open
import numpy as np

def analyze_lora_magnitude():
    """Analyze the magnitude of our LoRA weights to see if they're too small"""
    
    lora_path = "./models/anddrrew_flux_lora_diffusers_compatible/adapter_model.safetensors"
    
    print("=== ANALYZING LoRA MAGNITUDE ===")
    
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        lora_A_weights = []
        lora_B_weights = []
        
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            if 'lora_A' in key:
                lora_A_weights.append(tensor)
            elif 'lora_B' in key:
                lora_B_weights.append(tensor)
        
        print(f"Found {len(lora_A_weights)} LoRA A matrices")
        print(f"Found {len(lora_B_weights)} LoRA B matrices")
        
        # Analyze magnitudes
        for i, (A, B) in enumerate(zip(lora_A_weights[:5], lora_B_weights[:5])):
            # The actual LoRA contribution is B @ A
            lora_weight = B @ A
            
            print(f"\nLoRA pair {i+1}:")
            print(f"  A shape: {A.shape}, B shape: {B.shape}")
            print(f"  A stats: mean={A.mean():.6f}, std={A.std():.6f}, max_abs={A.abs().max():.6f}")
            print(f"  B stats: mean={B.mean():.6f}, std={B.std():.6f}, max_abs={B.abs().max():.6f}")
            print(f"  Combined LoRA weight:")
            print(f"    shape: {lora_weight.shape}")
            print(f"    mean: {lora_weight.mean():.6f}")
            print(f"    std: {lora_weight.std():.6f}")
            print(f"    max_abs: {lora_weight.abs().max():.6f}")
            
        # Check overall statistics
        all_A = torch.cat([w.flatten() for w in lora_A_weights])
        all_B = torch.cat([w.flatten() for w in lora_B_weights])
        
        print(f"\n=== OVERALL STATISTICS ===")
        print(f"All LoRA A weights:")
        print(f"  Mean: {all_A.mean():.6f}, Std: {all_A.std():.6f}")
        print(f"  Range: [{all_A.min():.6f}, {all_A.max():.6f}]")
        
        print(f"All LoRA B weights:")
        print(f"  Mean: {all_B.mean():.6f}, Std: {all_B.std():.6f}")
        print(f"  Range: [{all_B.min():.6f}, {all_B.max():.6f}]")
        
        # Check if weights are effectively zero
        A_near_zero = (all_A.abs() < 1e-6).float().mean()
        B_near_zero = (all_B.abs() < 1e-6).float().mean()
        
        print(f"\nPercentage near zero (< 1e-6):")
        print(f"  LoRA A: {A_near_zero*100:.2f}%")
        print(f"  LoRA B: {B_near_zero*100:.2f}%")
        
        if A_near_zero > 0.9 or B_near_zero > 0.9:
            print("\n❌ PROBLEM: Most LoRA weights are near zero!")
            print("This suggests the LoRA wasn't trained properly.")
        elif all_A.abs().max() < 1e-3 and all_B.abs().max() < 1e-3:
            print("\n⚠️  WARNING: LoRA weights are very small")
            print("The effect might be too subtle to see visually.")
        else:
            print("\n✅ LoRA weights look reasonable")

if __name__ == "__main__":
    analyze_lora_magnitude()

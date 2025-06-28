#!/usr/bin/env python3
"""
Simple LoRA weight checker
"""

from safetensors import safe_open
import torch
import os

def check_lora_weights(path):
    """Check if LoRA weights are valid"""
    print(f"🔍 Checking: {os.path.basename(path)}")
    
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            # Check first few weights
            keys = list(f.keys())
            sample_keys = [k for k in keys if "lora_down.weight" in k][:3]
            
            valid_weights = 0
            nan_weights = 0
            
            for key in sample_keys:
                tensor = f.get_tensor(key)
                if torch.isnan(tensor).any():
                    print(f"   ❌ {key}: Contains NaN")
                    nan_weights += 1
                else:
                    mean_val = tensor.float().mean().item()
                    print(f"   ✅ {key}: Mean = {mean_val:.6f}")
                    valid_weights += 1
            
            if nan_weights > 0:
                print(f"   ⚠️  CORRUPTED: {nan_weights} NaN weights found!")
                return False
            else:
                print(f"   ✅ VALID: All {valid_weights} weights are clean")
                return True
                
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False

# Check all available models
models_to_check = [
    "outputs/test_fp32/test_fp32.safetensors",
    "outputs/anddrrew_lora_v1/anddrrew_lora_v1.safetensors",
    "outputs/anddrrew_lora_v1/anddrrew_lora_v1-000002.safetensors", 
    "outputs/anddrrew_lora_v1/anddrrew_lora_v1-000004.safetensors"
]

print("🧪 LoRA Weight Validation")
print("=" * 40)

valid_models = []
for model_path in models_to_check:
    if os.path.exists(model_path):
        is_valid = check_lora_weights(model_path)
        if is_valid:
            valid_models.append(model_path)
    else:
        print(f"❌ Not found: {model_path}")

print(f"\n🎯 Summary:")
print(f"Valid models: {len(valid_models)}")
for model in valid_models:
    print(f"   ✅ {model}")

if valid_models:
    print(f"\n💡 Use one of the valid models instead!")
else:
    print(f"\n⚠️  All models are corrupted - need to retrain with stable settings!")

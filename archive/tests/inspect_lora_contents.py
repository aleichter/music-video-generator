#!/usr/bin/env python3

import torch
import json
import os

def inspect_lora_contents():
    """Inspect what's actually in the LoRA files"""
    
    lora_path = "./models/anddrrew_fixed_flux_lora/fixed_flux_lora_epoch_4_peft"
    
    print("=== INSPECTING LoRA CONTENTS ===")
    print(f"LoRA path: {lora_path}")
    
    # Check adapter_config.json
    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        print(f"\n=== adapter_config.json ===")
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(json.dumps(config, indent=2))
    
    # Load the actual LoRA weights
    model_path = os.path.join(lora_path, "adapter_model.safetensors")
    if os.path.exists(model_path):
        print(f"\n=== adapter_model.safetensors contents ===")
        from safetensors import safe_open
        
        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"Total parameters: {len(keys)}")
            print(f"First 20 parameter names:")
            for i, key in enumerate(keys[:20]):
                tensor = f.get_tensor(key)
                print(f"  {i+1:2d}. {key} - shape: {tensor.shape}")
            
            if len(keys) > 20:
                print(f"  ... and {len(keys) - 20} more parameters")
                
            # Check for different prefixes
            prefixes = {}
            for key in keys:
                parts = key.split('.')
                if len(parts) > 0:
                    prefix = parts[0]
                    if prefix not in prefixes:
                        prefixes[prefix] = 0
                    prefixes[prefix] += 1
            
            print(f"\n=== Parameter prefixes ===")
            for prefix, count in prefixes.items():
                print(f"  {prefix}: {count} parameters")
    
    # Also check what diffusers expects
    print(f"\n=== What diffusers expects vs what we have ===")
    print("Diffusers load_lora_weights() looks for prefixes like:")
    print("  - 'transformer' for FluxTransformer2DModel")
    print("  - 'text_encoder' for CLIPTextModel")
    print("  - etc.")
    
    print(f"\nOur LoRA seems to have prefixes: {list(prefixes.keys())}")

if __name__ == "__main__":
    inspect_lora_contents()

#!/usr/bin/env python3

import torch
from diffusers import FluxPipeline
import os
from PIL import Image
import numpy as np

def debug_lora_application():
    """Debug why LoRA isn't having any effect"""
    
    print("=== DEBUGGING LoRA APPLICATION ===")
    
    # Load pipeline
    print("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")
    
    # Check initial transformer parameters
    print("\n=== Checking Initial Transformer State ===")
    initial_params = {}
    for name, param in pipe.transformer.named_parameters():
        if 'attn' in name and len(name.split('.')) > 3:  # Get a few key attention parameters
            initial_params[name] = param.clone().detach()
            print(f"Initial {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            if len(initial_params) >= 3:  # Just check a few
                break
    
    # Load LoRA
    print("\n=== Loading LoRA ===")
    lora_path = "./models/anddrrew_fixed_flux_lora/fixed_flux_lora_epoch_4_peft"
    
    try:
        # Check what LoRA contains
        from peft import PeftModel
        print(f"Loading LoRA from: {lora_path}")
        
        # Load the LoRA adapter first to inspect it
        pipe.load_lora_weights(lora_path)
        print("LoRA loaded as adapter")
        
        # Check LoRA state
        print("\n=== Inspecting LoRA State ===")
        if hasattr(pipe.transformer, 'peft_config'):
            print(f"PEFT config found: {pipe.transformer.peft_config}")
            
        # Check which modules have LoRA
        lora_modules = []
        for name, module in pipe.transformer.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                lora_modules.append(name)
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A = module.lora_A['default']
                    lora_B = module.lora_B['default']
                    print(f"LoRA found in {name}: A={lora_A.weight.shape}, B={lora_B.weight.shape}")
                    if len(lora_modules) >= 5:  # Just show a few
                        break
        
        print(f"Total LoRA modules found: {len(lora_modules)}")
        if len(lora_modules) == 0:
            print("WARNING: No LoRA modules found!")
        
        # Check transformer parameters after loading LoRA (before fusion)
        print("\n=== Checking Transformer State After LoRA Load (Before Fusion) ===")
        for name, param in pipe.transformer.named_parameters():
            if name in initial_params:
                print(f"After load {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                diff = (param - initial_params[name]).abs().sum().item()
                print(f"  Difference from initial: {diff:.6f}")
        
        # Now fuse the LoRA
        print("\n=== Fusing LoRA ===")
        pipe.fuse_lora(lora_scale=1.0)
        print("LoRA fused")
        
        # Check transformer parameters after fusion
        print("\n=== Checking Transformer State After Fusion ===")
        for name, param in pipe.transformer.named_parameters():
            if name in initial_params:
                print(f"After fusion {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                diff = (param - initial_params[name]).abs().sum().item()
                print(f"  Difference from initial: {diff:.6f}")
                if diff == 0:
                    print(f"  WARNING: No change detected in {name}!")
        
        # Check if LoRA modules still exist after fusion
        print("\n=== Checking LoRA State After Fusion ===")
        post_fusion_lora_modules = []
        for name, module in pipe.transformer.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                post_fusion_lora_modules.append(name)
        
        print(f"LoRA modules remaining after fusion: {len(post_fusion_lora_modules)}")
        
        # Test generation to see if there's any difference
        print("\n=== Testing Generation ===")
        prompt = "a person with brown hair and brown eyes"
        generator = torch.Generator(device="cuda").manual_seed(42)
        
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=4,  # Just a few steps for quick test
                guidance_scale=3.5,
                width=512,  # Smaller for speed
                height=512,
                generator=generator
            )
        
        print("Generation completed - if we got here, the pipeline works")
        
    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lora_application()

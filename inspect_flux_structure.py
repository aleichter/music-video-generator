#!/usr/bin/env python3
"""
Inspect FLUX transformer structure in detail to understand naming conventions
"""

import torch
from flux_image_generator import FluxImageGenerator

def inspect_transformer_detailed():
    print("🔍 Detailed FLUX transformer inspection...")
    
    # Create generator and load pipeline
    generator = FluxImageGenerator()
    generator.load_pipeline()
    transformer = generator.pipeline.transformer
    
    # Look at the first transformer block in detail
    print("\n📋 First transformer block structure:")
    first_block = transformer.transformer_blocks[0]
    
    def print_detailed_structure(module, prefix="", max_depth=4, current_depth=0):
        if current_depth >= max_depth:
            return
            
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            print(f"{'  ' * current_depth}{full_name}: {type(child).__name__}")
            
            # Look for linear layers
            if hasattr(child, 'weight') and child.weight is not None:
                print(f"{'  ' * (current_depth+1)}  -> Weight: {child.weight.shape}")
                
            print_detailed_structure(child, full_name, max_depth, current_depth + 1)
    
    print_detailed_structure(first_block, "transformer_blocks.0", max_depth=3)
    
    print("\n📋 First single transformer block structure:")
    first_single = transformer.single_transformer_blocks[0]
    print_detailed_structure(first_single, "single_transformer_blocks.0", max_depth=3)
    
    # Now let's load some LoRA keys and see the mapping
    print("\n🔑 Sample LoRA keys:")
    import safetensors.torch
    lora_path = 'outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors'
    
    with safetensors.torch.safe_open(lora_path, framework='pt') as f:
        sample_keys = list(f.keys())[:20]
        for key in sample_keys:
            if 'lora_unet_' in key:
                clean_key = key.replace('lora_unet_', '')
                print(f"  {key} -> {clean_key}")

if __name__ == "__main__":
    inspect_transformer_detailed()

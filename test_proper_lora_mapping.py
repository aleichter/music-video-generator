#!/usr/bin/env python3
"""
Proper LoRA mapping for FLUX using sd-scripts to diffusers conversion
"""

import torch
import safetensors.torch
from flux_image_generator import FluxImageGenerator

def create_mapping_table():
    """Create mapping from sd-scripts naming to diffusers naming"""
    mapping = {}
    
    # Map double_blocks (transformer_blocks in diffusers)
    for i in range(19):  # FLUX has 19 transformer blocks
        # Image attention
        mapping[f'double_blocks_{i}_img_attn_qkv'] = f'transformer_blocks.{i}.attn.to_q'  # We'll handle qkv specially
        mapping[f'double_blocks_{i}_img_attn_proj'] = f'transformer_blocks.{i}.attn.to_out.0'
        
        # Text attention  
        mapping[f'double_blocks_{i}_txt_attn_qkv'] = f'transformer_blocks.{i}.attn.add_q_proj'  # We'll handle this specially too
        mapping[f'double_blocks_{i}_txt_attn_proj'] = f'transformer_blocks.{i}.attn.to_add_out'
        
        # Image MLP
        mapping[f'double_blocks_{i}_img_mlp_0'] = f'transformer_blocks.{i}.ff.net.2'
        mapping[f'double_blocks_{i}_img_mlp_2'] = f'transformer_blocks.{i}.ff.net.2'  # might be wrong, need to check
        
        # Text MLP
        mapping[f'double_blocks_{i}_txt_mlp_0'] = f'transformer_blocks.{i}.ff_context.net.2'
        mapping[f'double_blocks_{i}_txt_mlp_2'] = f'transformer_blocks.{i}.ff_context.net.2'  # might be wrong, need to check
        
        # Modulation layers
        mapping[f'double_blocks_{i}_img_mod_lin'] = f'transformer_blocks.{i}.norm1.linear'
        mapping[f'double_blocks_{i}_txt_mod_lin'] = f'transformer_blocks.{i}.norm1_context.linear'
    
    # Map single_blocks (single_transformer_blocks in diffusers)
    for i in range(38):  # FLUX has 38 single transformer blocks
        mapping[f'single_blocks_{i}_linear1'] = f'single_transformer_blocks.{i}.proj_mlp'
        mapping[f'single_blocks_{i}_linear2'] = f'single_transformer_blocks.{i}.proj_out'
        mapping[f'single_blocks_{i}_modulation_lin'] = f'single_transformer_blocks.{i}.norm.linear'
    
    return mapping

def test_proper_lora_mapping():
    print("🧪 Testing proper LoRA mapping with sd-scripts to diffusers conversion...")
    
    # Load the LoRA file manually
    lora_path = 'outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors'
    
    with safetensors.torch.safe_open(lora_path, framework='pt') as f:
        lora_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"Loaded {len(lora_dict)} LoRA weights")
    
    # Create generator
    generator = FluxImageGenerator()
    generator.load_pipeline()
    transformer = generator.pipeline.transformer
    
    # Create mapping table
    mapping = create_mapping_table()
    print(f"Created mapping table with {len(mapping)} entries")
    
    # Group LoRA weights by layer
    print("\n🔗 Mapping and applying LoRA weights...")
    lora_layers = {}
    for key, weight in lora_dict.items():
        if 'lora_unet_' in key:
            # Extract layer path and type
            clean_key = key.replace('lora_unet_', '')
            parts = clean_key.split('.')
            if len(parts) >= 3:
                layer_path = '.'.join(parts[:-2])  # Everything except lora_down/up.weight
                lora_type = parts[-2]  # lora_down or lora_up
                
                if layer_path not in lora_layers:
                    lora_layers[layer_path] = {}
                lora_layers[layer_path][lora_type] = weight
    
    print(f"Found {len(lora_layers)} LoRA layer groups")
    
    # Apply LoRA weights using proper mapping
    applied_count = 0
    
    for sd_layer_path, lora_weights in lora_layers.items():
        if 'lora_down' in lora_weights and 'lora_up' in lora_weights:
            # Check if we have a mapping for this layer
            if sd_layer_path in mapping:
                diffusers_path = mapping[sd_layer_path]
                
                try:
                    # Navigate to the target layer in diffusers model
                    target_module = transformer
                    for part in diffusers_path.split('.'):
                        target_module = getattr(target_module, part)
                    
                    # Check if it's a linear layer
                    if hasattr(target_module, 'weight'):
                        lora_down = lora_weights['lora_down']
                        lora_up = lora_weights['lora_up']
                        
                        # Apply LoRA: W = W + lora_up @ lora_down
                        lora_weight = torch.mm(lora_up, lora_down)
                        
                        # Check dimensions match
                        if lora_weight.shape == target_module.weight.shape:
                            target_module.weight.data += lora_weight
                            applied_count += 1
                            print(f"✅ Applied LoRA to {sd_layer_path} -> {diffusers_path}")
                        else:
                            print(f"❌ Shape mismatch for {sd_layer_path}: {lora_weight.shape} vs {target_module.weight.shape}")
                    else:
                        print(f"❌ Target module {diffusers_path} is not a linear layer")
                        
                except Exception as e:
                    print(f"❌ Failed to apply LoRA to {sd_layer_path}: {e}")
            else:
                print(f"⚠️ No mapping found for {sd_layer_path}")
    
    print(f"\n🎉 Successfully applied {applied_count} LoRA layers!")
    
    if applied_count > 0:
        # Test generation with LoRA applied
        print("\n🎨 Testing generation with properly mapped LoRA...")
        image = generator.generate_image(
            "anddrrew, professional portrait",
            output_path="proper_lora_test.png",
            width=512, height=512, num_inference_steps=20, seed=42
        )
        print("✅ Generation complete!")
        
        # Also test with a different prompt to see the effect
        print("\n🎨 Testing with different prompt...")
        image2 = generator.generate_image(
            "anddrrew in a business suit, office background",
            output_path="proper_lora_business.png",
            width=512, height=512, num_inference_steps=20, seed=123
        )
        print("✅ Second generation complete!")
        
        print("\n📊 Generated images:")
        print("  - proper_lora_test.png (anddrrew portrait)")
        print("  - proper_lora_business.png (anddrrew business)")
        
    else:
        print("❌ No LoRA weights were successfully applied")
        
        # Let's debug by checking what layers we actually have
        print("\n🔍 Debugging - checking available LoRA layers:")
        sample_keys = list(lora_layers.keys())[:10]
        for key in sample_keys:
            print(f"  {key}")
        
        print("\n🔍 Checking mapping table samples:")
        sample_mappings = list(mapping.items())[:10]
        for sd_key, diffusers_key in sample_mappings:
            print(f"  {sd_key} -> {diffusers_key}")

if __name__ == "__main__":
    test_proper_lora_mapping()

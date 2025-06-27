#!/usr/bin/env python3
"""
Manual LoRA loading test for FLUX
"""

import torch
import safetensors.torch
from flux_image_generator import FluxImageGenerator

def test_manual_lora_loading():
    print("🧪 Testing manual LoRA loading...")
    
    # Load the LoRA file manually
    lora_path = 'outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors'
    
    with safetensors.torch.safe_open(lora_path, framework='pt') as f:
        lora_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"Loaded {len(lora_dict)} LoRA weights")
    
    # Create generator
    generator = FluxImageGenerator()
    generator.load_pipeline()
    
    # Try to understand the FLUX transformer structure
    print("\n🔍 Inspecting FLUX transformer...")
    transformer = generator.pipeline.transformer
    
    # Check if transformer has load_attn_procs method
    if hasattr(transformer, 'load_attn_procs'):
        print("✓ Transformer has load_attn_procs method")
        
        # Convert sd-scripts keys to attention processor format
        attn_procs = {}
        
        for key, weight in lora_dict.items():
            if 'lora_unet_' in key:
                # Remove lora_unet_ prefix and convert to attention processor format
                clean_key = key.replace('lora_unet_', '')
                
                # Map to attention processor keys
                if 'attn' in clean_key and ('lora_down' in clean_key or 'lora_up' in clean_key):
                    attn_procs[clean_key] = weight
        
        print(f"Converted {len(attn_procs)} keys to attention processor format")
        
        if attn_procs:
            try:
                transformer.load_attn_procs(attn_procs)
                print("✅ Successfully loaded LoRA into transformer")
                
                # Test generation
                print("\n🎨 Testing generation with manually loaded LoRA...")
                image = generator.generate_image(
                    "anddrrew, professional portrait",
                    output_path="manual_lora_test.png",
                    width=512, height=512, num_inference_steps=20, seed=42
                )
                print("✅ Generation complete!")
                
            except Exception as e:
                print(f"❌ Failed to load attention processors: {e}")
    else:
        print("❌ Transformer doesn't have load_attn_procs method")
        
        # Try alternative approach - direct weight injection
        print("Trying direct weight injection...")
        
        # First, let's inspect the transformer structure
        print("\n🔍 Analyzing transformer structure...")
        
        def print_module_structure(module, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                print(f"{'  ' * current_depth}{full_name}: {type(child).__name__}")
                
                # Look for linear layers that might correspond to LoRA
                if hasattr(child, 'weight') and child.weight is not None:
                    print(f"{'  ' * (current_depth+1)}  -> Linear layer: {child.weight.shape}")
                
                print_module_structure(child, full_name, max_depth, current_depth + 1)
        
        print_module_structure(transformer, max_depth=2)
        
        # Now let's try to map LoRA keys to transformer layers
        print("\n🔗 Mapping LoRA keys to transformer layers...")
        
        # Group LoRA weights by layer
        lora_layers = {}
        for key, weight in lora_dict.items():
            if 'lora_unet_' in key:
                # Extract layer path and type
                clean_key = key.replace('lora_unet_', '')
                
                # Parse the key structure: e.g., "transformer_blocks.0.attn.to_q.lora_down.weight"
                parts = clean_key.split('.')
                if len(parts) >= 3:
                    layer_path = '.'.join(parts[:-2])  # Everything except lora_down/up.weight
                    lora_type = parts[-2]  # lora_down or lora_up
                    
                    if layer_path not in lora_layers:
                        lora_layers[layer_path] = {}
                    lora_layers[layer_path][lora_type] = weight
        
        print(f"Found {len(lora_layers)} LoRA layer groups:")
        for layer_path in sorted(lora_layers.keys())[:5]:  # Show first 5
            print(f"  {layer_path}: {list(lora_layers[layer_path].keys())}")
        if len(lora_layers) > 5:
            print(f"  ... and {len(lora_layers) - 5} more")
        
        # Apply LoRA weights to matching layers
        print("\n🎯 Applying LoRA weights...")
        applied_count = 0
        
        for layer_path, lora_weights in lora_layers.items():
            if 'lora_down' in lora_weights and 'lora_up' in lora_weights:
                try:
                    # Navigate to the target layer
                    target_module = transformer
                    for part in layer_path.split('.'):
                        if hasattr(target_module, part):
                            target_module = getattr(target_module, part)
                        else:
                            # Try to find the layer with a similar name
                            found = False
                            for attr_name in dir(target_module):
                                if part in attr_name and hasattr(target_module, attr_name):
                                    target_module = getattr(target_module, attr_name)
                                    found = True
                                    break
                            if not found:
                                raise AttributeError(f"Cannot find {part}")
                    
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
                            print(f"✅ Applied LoRA to {layer_path}")
                        else:
                            print(f"❌ Shape mismatch for {layer_path}: {lora_weight.shape} vs {target_module.weight.shape}")
                    
                except Exception as e:
                    print(f"❌ Failed to apply LoRA to {layer_path}: {e}")
        
        print(f"\n🎉 Successfully applied {applied_count} LoRA layers!")
        
        if applied_count > 0:
            # Test generation with LoRA applied
            print("\n🎨 Testing generation with manually injected LoRA...")
            image = generator.generate_image(
                "anddrrew, professional portrait",
                output_path="manual_lora_test.png",
                width=512, height=512, num_inference_steps=20, seed=42
            )
            print("✅ Generation complete!")
            
            # Also generate without LoRA for comparison
            print("\n🔄 Generating comparison image without LoRA...")
            # Reset the weights by subtracting the LoRA
            for layer_path, lora_weights in lora_layers.items():
                if 'lora_down' in lora_weights and 'lora_up' in lora_weights:
                    try:
                        target_module = transformer
                        for part in layer_path.split('.'):
                            if hasattr(target_module, part):
                                target_module = getattr(target_module, part)
                            else:
                                for attr_name in dir(target_module):
                                    if part in attr_name and hasattr(target_module, attr_name):
                                        target_module = getattr(target_module, attr_name)
                                        break
                        
                        if hasattr(target_module, 'weight'):
                            lora_down = lora_weights['lora_down']
                            lora_up = lora_weights['lora_up']
                            lora_weight = torch.mm(lora_up, lora_down)
                            
                            if lora_weight.shape == target_module.weight.shape:
                                target_module.weight.data -= lora_weight
                    except:
                        pass
            
            image_no_lora = generator.generate_image(
                "anddrrew, professional portrait",
                output_path="no_lora_comparison.png",
                width=512, height=512, num_inference_steps=20, seed=42
            )
            print("✅ Comparison generation complete!")
            print("\n📊 Compare the images:")
            print("  - manual_lora_test.png (with LoRA)")
            print("  - no_lora_comparison.png (without LoRA)")
        else:
            print("❌ No LoRA weights were successfully applied")

if __name__ == "__main__":
    test_manual_lora_loading()

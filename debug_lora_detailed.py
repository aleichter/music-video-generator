#!/usr/bin/env python3
"""
Detailed LoRA debugging script to identify why images are black squares
"""

import torch
import os
from pathlib import Path
from safetensors import safe_open
from flux_image_generator import FluxImageGenerator

def analyze_lora_weights(lora_path):
    """Analyze LoRA weights in detail"""
    print(f"\n🔍 Analyzing LoRA weights: {lora_path}")
    
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        print(f"📊 Total keys: {len(keys)}")
        
        # Categorize keys
        te_keys = [k for k in keys if "lora_te1" in k]
        unet_keys = [k for k in keys if "lora_unet" in k]
        alpha_keys = [k for k in keys if ".alpha" in k]
        up_keys = [k for k in keys if "lora_up" in k]
        down_keys = [k for k in keys if "lora_down" in k]
        
        print(f"   Text Encoder keys: {len(te_keys)}")
        print(f"   UNet keys: {len(unet_keys)}")
        print(f"   Alpha keys: {len(alpha_keys)}")
        print(f"   Up keys: {len(up_keys)}")
        print(f"   Down keys: {len(down_keys)}")
        
        # Sample some weights
        print(f"\n📋 Weight analysis:")
        sample_keys = keys[:10]
        for key in sample_keys:
            tensor = f.get_tensor(key)
            mean_val = tensor.float().mean().item()
            std_val = tensor.float().std().item()
            max_val = tensor.float().max().item()
            min_val = tensor.float().min().item()
            
            print(f"   {key[:60]}...")
            print(f"      Shape: {list(tensor.shape)}, dtype: {tensor.dtype}")
            print(f"      Stats: mean={mean_val:.6f}, std={std_val:.6f}")
            print(f"      Range: [{min_val:.6f}, {max_val:.6f}]")
            
            # Check for problematic values
            if torch.isnan(tensor).any():
                print(f"      ⚠️  Contains NaN!")
            elif torch.isinf(tensor).any():
                print(f"      ⚠️  Contains Inf!")
            elif abs(max_val) < 1e-8 and abs(min_val) < 1e-8:
                print(f"      ⚠️  Appears to be all zeros!")
            elif abs(max_val) > 100 or abs(min_val) < -100:
                print(f"      ⚠️  Extreme values detected!")
            else:
                print(f"      ✅ Looks normal")
        
        return {
            'total_keys': len(keys),
            'te_keys': len(te_keys),
            'unet_keys': len(unet_keys),
            'sample_keys': sample_keys
        }

def test_lora_loading_methods(lora_path):
    """Test different LoRA loading methods"""
    print(f"\n🧪 Testing LoRA loading methods...")
    
    generator = FluxImageGenerator()
    generator.load_pipeline()
    
    # Method 1: Standard diffusers loading
    print(f"\n1️⃣ Testing standard diffusers loading...")
    try:
        generator.pipeline.load_lora_weights(lora_path, adapter_name="test_adapter")
        print(f"   ✅ Standard loading successful")
        
        # Check if adapter is actually loaded
        if hasattr(generator.pipeline, 'get_active_adapters'):
            adapters = generator.pipeline.get_active_adapters()
            print(f"   Active adapters: {adapters}")
        
        # Test generation
        print(f"   🎨 Testing generation...")
        result = generator.pipeline(
            "test prompt",
            num_inference_steps=5,
            width=256,
            height=256,
            max_sequence_length=256
        )
        
        image = result.images[0]
        # Check if image is black
        import numpy as np
        img_array = np.array(image)
        if img_array.max() < 10:  # Very dark image
            print(f"   ⚠️  Generated image is nearly black!")
        else:
            print(f"   ✅ Image generation working")
        
        # Unload for next test
        generator.pipeline.unload_lora_weights()
        
    except Exception as e:
        print(f"   ❌ Standard loading failed: {e}")
    
    # Method 2: Check if LoRA actually affects the model
    print(f"\n2️⃣ Testing LoRA effect comparison...")
    try:
        # Generate without LoRA
        print(f"   Generating baseline...")
        baseline_result = generator.pipeline(
            "portrait of a person",
            num_inference_steps=5,
            width=256,
            height=256,
            max_sequence_length=256
        )
        
        # Load LoRA and generate
        print(f"   Loading LoRA and generating...")
        generator.pipeline.load_lora_weights(lora_path, adapter_name="test_adapter2")
        lora_result = generator.pipeline(
            "portrait of a person",
            num_inference_steps=5,
            width=256,
            height=256,
            max_sequence_length=256
        )
        
        # Compare images
        import numpy as np
        baseline_array = np.array(baseline_result.images[0])
        lora_array = np.array(lora_result.images[0])
        
        diff = np.abs(baseline_array.astype(float) - lora_array.astype(float)).mean()
        print(f"   Image difference: {diff:.2f}")
        
        if diff < 1.0:
            print(f"   ⚠️  Images are nearly identical - LoRA may not be working!")
        else:
            print(f"   ✅ LoRA is affecting generation")
        
        generator.pipeline.unload_lora_weights()
        
    except Exception as e:
        print(f"   ❌ Effect comparison failed: {e}")
    
    generator.cleanup()

def test_manual_lora_inspection(lora_path):
    """Manually inspect LoRA structure"""
    print(f"\n🔬 Manual LoRA inspection...")
    
    # Load the state dict
    state_dict = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    # Look for patterns
    print(f"📋 Key patterns:")
    key_patterns = {}
    for key in state_dict.keys():
        # Extract pattern
        if 'lora_te1' in key:
            category = 'text_encoder'
        elif 'lora_unet' in key:
            category = 'transformer'
        else:
            category = 'other'
        
        if category not in key_patterns:
            key_patterns[category] = []
        key_patterns[category].append(key)
    
    for category, keys in key_patterns.items():
        print(f"   {category}: {len(keys)} keys")
        if keys:
            print(f"      Sample: {keys[0]}")
    
    # Check for sd-scripts specific format issues
    print(f"\n🔍 Checking sd-scripts compatibility...")
    
    # Look for typical sd-scripts patterns
    sd_scripts_patterns = [
        'lora_te1_text_model_encoder_layers',
        'lora_unet_double_blocks',
        'lora_unet_single_blocks'
    ]
    
    found_patterns = []
    for pattern in sd_scripts_patterns:
        matching_keys = [k for k in state_dict.keys() if pattern in k]
        if matching_keys:
            found_patterns.append(pattern)
            print(f"   ✅ Found {pattern}: {len(matching_keys)} keys")
        else:
            print(f"   ❌ Missing {pattern}")
    
    if len(found_patterns) < len(sd_scripts_patterns):
        print(f"   ⚠️  LoRA may not be in expected sd-scripts format!")
    
    return state_dict

def main():
    print("🐛 Detailed LoRA Debugging")
    print("=" * 50)
    
    lora_path = "outputs/anddrrew_lora_v1/anddrrew_lora_v1.safetensors"
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRA file not found: {lora_path}")
        return
    
    # Step 1: Analyze weights
    weight_info = analyze_lora_weights(lora_path)
    
    # Step 2: Test loading methods
    test_lora_loading_methods(lora_path)
    
    # Step 3: Manual inspection
    state_dict = test_manual_lora_inspection(lora_path)
    
    print(f"\n🎯 Diagnosis Summary:")
    print(f"=" * 30)
    print(f"📊 LoRA has {weight_info['total_keys']} parameters")
    print(f"📝 Text encoder keys: {weight_info['te_keys']}")
    print(f"🧠 Transformer keys: {weight_info['unet_keys']}")
    
    # Common issues checklist
    print(f"\n🔍 Common Issues Checklist:")
    print(f"   1. LoRA format compatibility with diffusers")
    print(f"   2. Key naming convention mismatch") 
    print(f"   3. Model precision/dtype conflicts")
    print(f"   4. Adapter registration problems")
    print(f"   5. Pipeline component mismatch")
    
    print(f"\n💡 Recommendations:")
    print(f"   - Try using ComfyUI which has better sd-scripts LoRA support")
    print(f"   - Convert LoRA to diffusers format")
    print(f"   - Check if LoRA was trained with correct base model")
    print(f"   - Verify training completed without corruption")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LoRA Diagnostic Script
Test different LoRA models and settings to diagnose training issues
"""

import torch
import os
from pathlib import Path
from flux_image_generator import FluxImageGenerator

def test_lora_models():
    """Test different LoRA models and settings"""
    
    generator = FluxImageGenerator()
    
    # Test prompts
    test_prompts = [
        "anddrrew",  # Just the trigger word
        "anddrrew, portrait",
        "anddrrew, portrait of a man",
        "anddrrew, headshot photo",
        "anddrrew, close-up portrait",
    ]
    
    # LoRA models to test
    lora_models = [
        "outputs/anddrrew_lora_v1/anddrrew_lora_v1.safetensors",          # Final model
        "outputs/anddrrew_lora_v1/anddrrew_lora_v1-000004.safetensors",  # Epoch 4
        "outputs/anddrrew_lora_v1/anddrrew_lora_v1-000002.safetensors",  # Epoch 2
    ]
    
    # LoRA scales to test
    lora_scales = [0.5, 0.8, 1.0, 1.2]
    
    output_dir = Path("lora_diagnostics")
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 FLUX LoRA Diagnostic Test")
    print("=" * 50)
    
    # Test 1: Compare different checkpoints
    print("\\n📊 Test 1: Comparing different training checkpoints")
    print("-" * 40)
    
    for i, lora_path in enumerate(lora_models):
        if not os.path.exists(lora_path):
            print(f"⏭️  Skipping {lora_path} (not found)")
            continue
            
        print(f"\\n🔧 Testing: {Path(lora_path).name}")
        
        try:
            # Unload any previous LoRA
            generator.unload_lora()
            
            # Load this LoRA
            generator.load_lora(lora_path, lora_scale=1.0)
            
            # Generate test image
            test_prompt = "anddrrew, portrait of a man"
            output_path = output_dir / f"checkpoint_{i+1}_{Path(lora_path).stem}.png"
            
            image = generator.generate_image(
                test_prompt,
                output_path=output_path,
                width=512,
                height=512,
                num_inference_steps=25,
                seed=42  # Fixed seed for comparison
            )
            
            print(f"✅ Generated: {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to test {lora_path}: {e}")
    
    # Test 2: Different LoRA scales with best model
    print("\\n📊 Test 2: Testing different LoRA scales")
    print("-" * 40)
    
    best_lora = "outputs/anddrrew_lora_v1/anddrrew_lora_v1.safetensors"
    if os.path.exists(best_lora):
        for scale in lora_scales:
            print(f"\\n🔧 Testing LoRA scale: {scale}")
            
            try:
                generator.unload_lora()
                generator.load_lora(best_lora, lora_scale=scale)
                
                output_path = output_dir / f"scale_{scale:.1f}.png"
                image = generator.generate_image(
                    "anddrrew, portrait of a man",
                    output_path=output_path,
                    width=512,
                    height=512,
                    num_inference_steps=25,
                    seed=42
                )
                
                print(f"✅ Generated: {output_path}")
                
            except Exception as e:
                print(f"❌ Failed at scale {scale}: {e}")
    
    # Test 3: Different prompts
    print("\\n📊 Test 3: Testing different prompts")
    print("-" * 40)
    
    if os.path.exists(best_lora):
        generator.unload_lora()
        generator.load_lora(best_lora, lora_scale=0.8)  # Use moderate scale
        
        for i, prompt in enumerate(test_prompts):
            print(f"\\n🔧 Testing prompt: '{prompt}'")
            
            try:
                output_path = output_dir / f"prompt_{i+1}_{prompt.replace(' ', '_').replace(',', '')}.png"
                
                image = generator.generate_image(
                    prompt,
                    output_path=output_path,
                    width=512,
                    height=512,
                    num_inference_steps=25,
                    seed=42
                )
                
                print(f"✅ Generated: {output_path}")
                
            except Exception as e:
                print(f"❌ Failed for prompt '{prompt}': {e}")
    
    # Test 4: Baseline comparison (no LoRA)
    print("\\n📊 Test 4: Baseline comparison (no LoRA)")
    print("-" * 40)
    
    try:
        generator.unload_lora()
        
        baseline_path = output_dir / "baseline_no_lora.png"
        image = generator.generate_image(
            "anddrrew, portrait of a man",
            output_path=baseline_path,
            width=512,
            height=512,
            num_inference_steps=25,
            seed=42
        )
        
        print(f"✅ Baseline generated: {baseline_path}")
        
    except Exception as e:
        print(f"❌ Failed to generate baseline: {e}")
    
    # Cleanup
    generator.cleanup()
    
    print("\\n🎯 Diagnostic Summary:")
    print("-" * 40)
    print(f"📁 All test images saved to: {output_dir}")
    print("\\n🔍 Analysis Tips:")
    print("1. Compare different checkpoints - earlier ones might be better")
    print("2. Try lower LoRA scales (0.5-0.8) if effect is too strong")
    print("3. Check if trigger word 'anddrrew' is working properly")
    print("4. Compare with baseline to see LoRA effect")
    print("\\n💡 Common Issues:")
    print("- Overfitting: Try earlier checkpoint (epoch 2 or 4)")
    print("- Too strong: Reduce LoRA scale to 0.5-0.7")
    print("- Wrong features: Check training data quality")
    print("- No effect: Increase LoRA scale or check trigger word")

def analyze_training_data():
    """Analyze the training dataset"""
    print("\\n📂 Training Data Analysis")
    print("-" * 40)
    
    dataset_path = Path("dataset/anddrrew")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Count images and captions
    images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.JPEG"))
    captions = list(dataset_path.glob("*.txt"))
    
    print(f"📸 Images found: {len(images)}")
    print(f"📝 Caption files found: {len(captions)}")
    
    # Sample a few captions
    print("\\n📋 Sample captions:")
    for i, caption_file in enumerate(captions[:5]):
        try:
            with open(caption_file, 'r') as f:
                caption = f.read().strip()
            print(f"  {i+1}. {caption_file.name}: {caption}")
        except Exception as e:
            print(f"  {i+1}. {caption_file.name}: Error reading - {e}")
    
    print("\\n💡 Caption Analysis:")
    print("- Check if 'anddrrew' appears in captions")
    print("- Ensure captions describe the person accurately")
    print("- Look for consistent features across images")

if __name__ == "__main__":
    # Analyze training data first
    analyze_training_data()
    
    # Run diagnostic tests
    test_lora_models()

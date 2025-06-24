#!/usr/bin/env python3

import torch
import gc
import os
from diffusers import FluxPipeline
from peft import PeftModel
import time

def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def test_epoch_30_lora():
    print("Testing Epoch 30 LoRA Model vs Base Model")
    print("="*50)
    
    # Test prompts - using some from our training data and new ones
    test_prompts = [
        "anddrrew, a young man with dark hair wearing a white t-shirt, standing confidently in front of a brick wall",
        "anddrrew wearing a casual outfit, looking directly at the camera with natural lighting",
        "anddrrew in a black jacket, portrait style with soft lighting",
        "anddrrew smiling, wearing a white shirt in an urban setting",
        "anddrrew with a serious expression, dramatic lighting, cinematic style"
    ]
    
    # Model paths
    base_model_id = "black-forest-labs/FLUX.1-dev"
    lora_checkpoint = "/workspace/music-video-generator/models/anddrrew_extended_flux_lora/quick_flux_lora_epoch_30_peft"
    
    # Create output directories
    base_output_dir = "/workspace/music-video-generator/epoch30_base_test"
    lora_output_dir = "/workspace/music-video-generator/epoch30_lora_test"
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(lora_output_dir, exist_ok=True)
    
    print(f"Base model outputs will be saved to: {base_output_dir}")
    print(f"LoRA model outputs will be saved to: {lora_output_dir}")
    print()
    
    # Load base pipeline
    print("Loading base FLUX pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    print("Base pipeline loaded successfully!")
    print(f"GPU memory after loading base model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Generate base model images
    print("\nGenerating BASE MODEL images...")
    for i, prompt in enumerate(test_prompts):
        print(f"Generating base image {i+1}/5: {prompt[:50]}...")
        
        try:
            start_time = time.time()
            image = pipeline(
                prompt,
                height=1024,
                width=1024,
                num_inference_steps=20,
                guidance_scale=3.5,
                generator=torch.Generator().manual_seed(42 + i)
            ).images[0]
            
            generation_time = time.time() - start_time
            output_path = os.path.join(base_output_dir, f"base_image_{i+1}.png")
            image.save(output_path)
            print(f"  ✓ Saved to {output_path} (took {generation_time:.1f}s)")
            
        except Exception as e:
            print(f"  ✗ Error generating base image {i+1}: {e}")
        
        cleanup_memory()
    
    print("\nBase model generation complete!")
    
    # Now load LoRA model
    print(f"\nLoading LoRA checkpoint from: {lora_checkpoint}")
    
    try:
        # Load the LoRA adapter and merge weights
        transformer_with_lora = PeftModel.from_pretrained(
            pipeline.transformer, 
            lora_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        
        # Merge LoRA weights into base model
        print("Merging LoRA weights into base model...")
        merged_transformer = transformer_with_lora.merge_and_unload()
        
        # Replace transformer in pipeline
        pipeline.transformer = merged_transformer
        
        print("LoRA model merged successfully!")
        print(f"GPU memory after merging LoRA: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Generate LoRA model images
        print("\nGenerating LORA MODEL images...")
        for i, prompt in enumerate(test_prompts):
            print(f"Generating LoRA image {i+1}/5: {prompt[:50]}...")
            
            try:
                start_time = time.time()
                image = pipeline(
                    prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=20,
                    guidance_scale=3.5,
                    generator=torch.Generator().manual_seed(42 + i)  # Same seed for comparison
                ).images[0]
                
                generation_time = time.time() - start_time
                output_path = os.path.join(lora_output_dir, f"lora_image_{i+1}.png")
                image.save(output_path)
                print(f"  ✓ Saved to {output_path} (took {generation_time:.1f}s)")
                
            except Exception as e:
                print(f"  ✗ Error generating LoRA image {i+1}: {e}")
            
            cleanup_memory()
        
        print("\nLoRA model generation complete!")
        
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        return
    
    print("\n" + "="*50)
    print("EPOCH 30 TEST COMPLETE!")
    print(f"Base model images: {base_output_dir}")
    print(f"LoRA model images: {lora_output_dir}")
    print("Compare the images to see the differences!")
    print("="*50)

if __name__ == "__main__":
    test_epoch_30_lora()

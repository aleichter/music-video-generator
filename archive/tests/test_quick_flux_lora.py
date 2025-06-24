#!/usr/bin/env python3

import torch
import logging
from pathlib import Path
from diffusers import FluxPipeline
from peft import PeftModel
import argparse
from PIL import Image
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_quick_flux_lora(lora_path, output_dir="test_outputs", device="cuda"):
    """Test the quick FLUX LoRA by generating images"""
    
    logger.info("🧪 Testing Quick FLUX LoRA...")
    logger.info(f"📂 LoRA path: {lora_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load base FLUX pipeline
    logger.info("📥 Loading base FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    
    # Move to device
    pipe = pipe.to(device)
    logger.info("✅ Base pipeline loaded")
    
    # Load LoRA weights
    logger.info("🎯 Loading LoRA weights...")
    try:
        # Try loading as PEFT model first
        peft_path = str(lora_path).replace('.pt', '_peft')
        if Path(peft_path).exists():
            logger.info(f"📂 Loading PEFT model from: {peft_path}")
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, peft_path)
        else:
            # Fallback to PyTorch checkpoint
            logger.info(f"📂 Loading PyTorch checkpoint from: {lora_path}")
            checkpoint = torch.load(lora_path, map_location=device)
            pipe.transformer.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("✅ LoRA weights loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load LoRA: {e}")
        return False
    
    # Test prompts
    test_prompts = [
        "anddrrew, portrait",
        "anddrrew, professional photo",
        "anddrrew, wearing glasses",
        "anddrrew, smiling",
        "anddrrew, close-up face",
        "portrait of anddrrew",
    ]
    
    logger.info(f"🖼️ Generating {len(test_prompts)} test images...")
    
    # Generate images
    for i, prompt in enumerate(test_prompts):
        try:
            logger.info(f"🎨 Generating image {i+1}/{len(test_prompts)}: '{prompt}'")
            
            start_time = time.time()
            
            with torch.inference_mode():
                images = pipe(
                    prompt=prompt,
                    width=512,
                    height=512,
                    num_inference_steps=4,  # Fast generation
                    guidance_scale=0.0,     # FLUX.1-schnell doesn't use guidance
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                ).images
            
            gen_time = time.time() - start_time
            
            # Save image
            filename = f"test_{i+1:02d}_{prompt.replace(' ', '_').replace(',', '')}.png"
            image_path = output_dir / filename
            images[0].save(image_path)
            
            logger.info(f"✅ Saved: {image_path} (took {gen_time:.1f}s)")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate image {i+1}: {e}")
            continue
    
    logger.info("🎉 Testing completed!")
    logger.info(f"📁 Check results in: {output_dir}")
    
    return True

def compare_with_base_model(output_dir="comparison_outputs", device="cuda"):
    """Generate comparison images with base model (no LoRA)"""
    
    logger.info("🔍 Generating comparison images with base model...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load base FLUX pipeline (no LoRA)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    
    # Test same prompts
    test_prompts = [
        "anddrrew, portrait",
        "anddrrew, professional photo",
        "portrait of anddrrew",
    ]
    
    for i, prompt in enumerate(test_prompts):
        try:
            logger.info(f"🎨 Base model generating: '{prompt}'")
            
            with torch.inference_mode():
                images = pipe(
                    prompt=prompt,
                    width=512,
                    height=512,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                ).images
            
            filename = f"base_{i+1:02d}_{prompt.replace(' ', '_').replace(',', '')}.png"
            image_path = output_dir / filename
            images[0].save(image_path)
            
            logger.info(f"✅ Base model saved: {image_path}")
            
        except Exception as e:
            logger.error(f"❌ Base model generation failed: {e}")
            continue
    
    logger.info("🔍 Base model comparison completed!")

def main():
    parser = argparse.ArgumentParser(description="Test Quick FLUX LoRA")
    parser.add_argument("--lora_path", default="models/anddrrew_quick_flux_lora/quick_flux_lora_epoch_4.pt", 
                       help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", default="quick_lora_test_outputs", help="Output directory")
    parser.add_argument("--compare_base", action="store_true", help="Also generate base model comparisons")
    
    args = parser.parse_args()
    
    try:
        # Test the LoRA model
        success = test_quick_flux_lora(args.lora_path, args.output_dir)
        
        if success and args.compare_base:
            # Generate base model comparisons
            compare_with_base_model("base_model_outputs")
        
        if success:
            logger.info("🎊 All tests completed successfully!")
            logger.info("📊 Now compare the LoRA vs base model outputs to see the difference")
        else:
            logger.error("❌ Testing failed")
            
    except Exception as e:
        logger.error(f"❌ Test script failed: {e}")
        raise

if __name__ == "__main__":
    main()

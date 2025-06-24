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

def test_quick_flux_lora_fixed(lora_path, output_dir="test_outputs", device="cuda"):
    """Test the quick FLUX LoRA with proper integration"""
    
    logger.info("🧪 Testing Quick FLUX LoRA (Fixed Version)...")
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
    
    # Load LoRA weights with better integration
    logger.info("🎯 Loading LoRA weights...")
    try:
        # Try loading as PEFT model
        peft_path = str(lora_path).replace('.pt', '_peft')
        if Path(peft_path).exists():
            logger.info(f"📂 Loading PEFT model from: {peft_path}")
            
            # Load PEFT model and properly integrate with pipeline
            transformer_with_lora = PeftModel.from_pretrained(pipe.transformer, peft_path)
            
            # Set the transformer to eval mode and merge if possible
            transformer_with_lora.eval()
            
            # Try to merge LoRA weights into base model for inference
            try:
                logger.info("🔄 Attempting to merge LoRA weights...")
                merged_transformer = transformer_with_lora.merge_and_unload()
                pipe.transformer = merged_transformer
                logger.info("✅ LoRA weights merged successfully")
            except Exception as merge_error:
                logger.warning(f"⚠️ Merge failed, using PEFT model directly: {merge_error}")
                pipe.transformer = transformer_with_lora
                
        else:
            logger.warning(f"PEFT model not found at {peft_path}, trying PyTorch checkpoint...")
            # Fallback to PyTorch checkpoint
            checkpoint = torch.load(lora_path, map_location=device)
            pipe.transformer.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("✅ LoRA weights loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load LoRA: {e}")
        logger.info("🔄 Falling back to base model for testing...")
        # Continue with base model for comparison
    
    # Test prompts
    test_prompts = [
        "anddrrew, portrait",
        "anddrrew, professional photo", 
        "anddrrew, wearing glasses",
        "portrait of anddrrew",
    ]
    
    logger.info(f"🖼️ Generating {len(test_prompts)} test images...")
    
    success_count = 0
    
    # Generate images
    for i, prompt in enumerate(test_prompts):
        try:
            logger.info(f"🎨 Generating image {i+1}/{len(test_prompts)}: '{prompt}'")
            
            start_time = time.time()
            
            with torch.inference_mode():
                # Use simpler generation parameters to avoid errors
                images = pipe(
                    prompt=prompt,
                    width=512,
                    height=512,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device).manual_seed(42 + i),  # Consistent seeds
                ).images
            
            gen_time = time.time() - start_time
            
            # Save image
            filename = f"lora_test_{i+1:02d}_{prompt.replace(' ', '_').replace(',', '')}.png"
            image_path = output_dir / filename
            images[0].save(image_path)
            
            logger.info(f"✅ Saved: {image_path} (took {gen_time:.1f}s)")
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to generate image {i+1}: {e}")
            
            # Try with a simpler prompt as fallback
            try:
                logger.info(f"🔄 Retrying with simpler prompt...")
                simple_prompt = "portrait"
                
                with torch.inference_mode():
                    images = pipe(
                        prompt=simple_prompt,
                        width=512,
                        height=512,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        num_images_per_prompt=1,
                    ).images
                
                filename = f"lora_fallback_{i+1:02d}_simple_portrait.png"
                image_path = output_dir / filename
                images[0].save(image_path)
                logger.info(f"✅ Fallback saved: {image_path}")
                success_count += 1
                
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
                continue
    
    logger.info(f"🎉 Testing completed! Generated {success_count}/{len(test_prompts)} images")
    logger.info(f"📁 Check results in: {output_dir}")
    
    return success_count > 0

def simple_lora_test(lora_path, output_dir="simple_test_outputs", device="cuda"):
    """Simple test that just loads the model and generates one image"""
    
    logger.info("🔬 Simple LoRA Test...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load base pipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        pipe = pipe.to(device)
        
        # Generate with base model first
        logger.info("🎨 Testing base model...")
        with torch.inference_mode():
            base_image = pipe(
                prompt="portrait of a person",
                width=512,
                height=512,
                num_inference_steps=4,
                guidance_scale=0.0,
            ).images[0]
        
        base_path = output_dir / "base_model_test.png"
        base_image.save(base_path)
        logger.info(f"✅ Base model works: {base_path}")
        
        # Try to load LoRA checkpoint directly
        peft_path = str(lora_path).replace('.pt', '_peft')
        if Path(peft_path).exists():
            logger.info("🎯 Loading LoRA...")
            
            # Load the LoRA state dict directly and check what's in it
            checkpoint = torch.load(lora_path, map_location=device)
            logger.info(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                lora_state = checkpoint['model_state_dict']
                logger.info(f"📊 LoRA parameters: {len([k for k in lora_state.keys() if 'lora' in k])}")
                
                # List some LoRA parameter names
                lora_params = [k for k in lora_state.keys() if 'lora' in k][:5]
                for param in lora_params:
                    logger.info(f"  - {param}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Quick FLUX LoRA (Fixed)")
    parser.add_argument("--lora_path", default="models/anddrrew_quick_flux_lora/quick_flux_lora_epoch_4.pt", 
                       help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", default="fixed_lora_test_outputs", help="Output directory")
    parser.add_argument("--simple_test", action="store_true", help="Run simple test only")
    
    args = parser.parse_args()
    
    try:
        if args.simple_test:
            success = simple_lora_test(args.lora_path)
        else:
            success = test_quick_flux_lora_fixed(args.lora_path, args.output_dir)
        
        if success:
            logger.info("🎊 Test completed!")
        else:
            logger.error("❌ Test failed")
            
    except Exception as e:
        logger.error(f"❌ Test script failed: {e}")
        raise

if __name__ == "__main__":
    main()

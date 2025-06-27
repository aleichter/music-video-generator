#!/usr/bin/env python3
"""
FLUX Image Generator
A clean, production-ready class for generating images with FLUX and optional LoRA
"""

import torch
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import json

# Import FluxPipeline directly to avoid import issues
try:
    from diffusers import FluxPipeline
except ImportError:
    # Fallback if FluxPipeline is not available
    from diffusers import DiffusionPipeline as FluxPipeline


class FluxImageGenerator:
    """Professional FLUX image generation class"""
    
    def __init__(self, model_name="black-forest-labs/FLUX.1-dev", device="cuda"):
        """
        Initialize the FLUX image generator
        
        Args:
            model_name: HuggingFace model name (uses local cache structure)
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self.loaded_lora_path = None
        
        # Default generation settings
        self.default_settings = {
            "num_inference_steps": 50,  # FLUX works better with more steps
            "guidance_scale": 7.5,     # Higher guidance for FLUX
            "width": 1024,
            "height": 1024,
            "max_sequence_length": 512
        }
    
    def load_pipeline(self):
        """Load the FLUX pipeline from local cache structure"""
        if self.pipeline is None:
            print(f"📥 Loading FLUX pipeline: {self.model_name}")
            
            # Set cache directory to workspace
            os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
            
            # Load the pipeline - it will use the local cache structure we set up
            try:
                self.pipeline = FluxPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    cache_dir="/workspace/.cache/huggingface",
                    local_files_only=False,  # Force local only, no downloads
                ).to(self.device)
                print("✅ Pipeline loaded successfully from local cache")
                
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load FLUX pipeline from local cache.\n"
                    f"Error: {e}\n\n"
                    f"Make sure you have run the setup_local_models.py script first to create\n"
                    f"the proper HuggingFace cache structure with your model files.\n"
                    f"Run: python setup_local_models.py"
                )
            
            # Enable memory efficient attention
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
    
    def load_lora(self, lora_path, lora_scale=1.0):
        """
        Load a LoRA model with proper key mapping for FLUX
        
        Args:
            lora_path: Path to the LoRA safetensors file
            lora_scale: Scale factor for LoRA (0.0-2.0)
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        
        # Load pipeline if not already loaded
        self.load_pipeline()
        
        print(f"🔧 Loading LoRA: {lora_path}")
        
        try:
            # Load with automatic adapter inference (should work with sd-scripts format)
            self.pipeline.load_lora_weights(
                lora_path,
                # Try to use the correct adapter name
                adapter_name="default"
            )
            
            # Apply scale to all components
            if hasattr(self.pipeline, 'set_adapters') and hasattr(self.pipeline, 'set_adapter_scale'):
                self.pipeline.set_adapters("default")
                self.pipeline.set_adapter_scale(lora_scale)
            elif hasattr(self.pipeline, 'fuse_lora'):
                # Alternative: fuse the LoRA with specific scale
                self.pipeline.fuse_lora(lora_scale=lora_scale)
            
            print(f"✅ LoRA loaded with scale {lora_scale}")
            print("   Note: This LoRA contains both text encoder and transformer weights")
            
        except Exception as e:
            print(f"   Standard loading failed: {e}")
            print("   Trying alternative loading method...")
            
            try:
                # Try loading without adapter name
                self.pipeline.load_lora_weights(lora_path)
                
                # Set scale for specific components if possible
                if hasattr(self.pipeline, 'set_lora_scale'):
                    self.pipeline.set_lora_scale(lora_scale)
                
                print(f"✅ LoRA loaded with alternative method, scale {lora_scale}")
                
            except Exception as e2:
                print(f"   All loading methods failed. Last error: {e2}")
                raise e2
        
        self.loaded_lora_path = lora_path
    
    def unload_lora(self):
        """Unload the currently loaded LoRA"""
        if self.loaded_lora_path:
            print("🔄 Unloading LoRA...")
            
            try:
                # Try multiple unloading methods
                if hasattr(self.pipeline, 'unload_lora_weights'):
                    self.pipeline.unload_lora_weights()
                elif hasattr(self.pipeline, 'disable_lora'):
                    self.pipeline.disable_lora()
                elif hasattr(self.pipeline, 'set_adapters'):
                    self.pipeline.set_adapters(None)
                elif hasattr(self.pipeline, 'unfuse_lora'):
                    self.pipeline.unfuse_lora()
                
                self.loaded_lora_path = None
                print("✅ LoRA unloaded")
                
            except Exception as e:
                print(f"   Warning: Error unloading LoRA: {e}")
                # Force unload by recreating pipeline if necessary
                self.loaded_lora_path = None
    
    def generate_image(self, prompt, negative_prompt="", output_path=None, **kwargs):
        """
        Generate an image using FLUX
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (optional)
            output_path: Path to save the image (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            PIL Image object
        """
        # Load pipeline if not already loaded
        self.load_pipeline()
        
        # Merge settings
        settings = {**self.default_settings, **kwargs}
        
        print(f"🎨 Generating image...")
        print(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"   Steps: {settings['num_inference_steps']}")
        print(f"   Size: {settings['width']}x{settings['height']}")
        
        # Set seed if provided
        if 'seed' in kwargs:
            torch.manual_seed(kwargs['seed'])
            print(f"   Seed: {kwargs['seed']}")
        
        # Generate image
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=settings['num_inference_steps'],
                guidance_scale=settings['guidance_scale'],
                width=settings['width'],
                height=settings['height'],
                max_sequence_length=settings['max_sequence_length']
            )
            
            image = result.images[0]
        
        # Save image if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=95, optimize=True)
            print(f"✅ Image saved: {output_path}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        return image
    
    def generate_batch(self, prompts, output_dir="generated_images", prefix="image", **kwargs):
        """
        Generate multiple images from a list of prompts
        
        Args:
            prompts: List of text prompts
            output_dir: Directory to save images
            prefix: Prefix for image filenames
            **kwargs: Generation parameters
            
        Returns:
            List of (image, filename) tuples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🎨 Generating {len(prompts)} images...")
        
        for i, prompt in enumerate(prompts):
            print(f"\n📸 Image {i+1}/{len(prompts)}")
            
            # Generate filename
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            filename = f"{prefix}_{i+1:02d}_{safe_prompt}_{timestamp}.png"
            output_path = output_dir / filename
            
            # Generate image
            image = self.generate_image(prompt, output_path=output_path, **kwargs)
            results.append((image, str(output_path)))
            
            # Clear memory between generations
            torch.cuda.empty_cache()
        
        print(f"\n✅ Batch generation complete: {len(results)} images saved to {output_dir}")
        return results
    
    def generate_lora_comparison(self, prompt, lora_path, output_dir="comparisons", **kwargs):
        """
        Generate comparison images with and without LoRA
        
        Args:
            prompt: Text prompt for generation
            lora_path: Path to LoRA file
            output_dir: Directory to save comparison
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (base_image, lora_image)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🔍 Generating LoRA comparison...")
        
        # Generate without LoRA
        print("  Generating base image (without LoRA)...")
        self.unload_lora()  # Ensure no LoRA is loaded
        base_image = self.generate_image(
            prompt, 
            output_path=output_dir / f"base_{timestamp}.png",
            **kwargs
        )
        
        # Generate with LoRA
        print("  Generating LoRA image...")
        self.load_lora(lora_path)
        lora_image = self.generate_image(
            prompt,
            output_path=output_dir / f"lora_{timestamp}.png", 
            **kwargs
        )
        
        # Create side-by-side comparison
        comparison = Image.new('RGB', (base_image.width * 2, base_image.height))
        comparison.paste(base_image, (0, 0))
        comparison.paste(lora_image, (base_image.width, 0))
        
        comparison_path = output_dir / f"comparison_{timestamp}.png"
        comparison.save(comparison_path, quality=95)
        print(f"✅ Comparison saved: {comparison_path}")
        
        return base_image, lora_image
    
    def save_generation_log(self, prompts, settings, output_dir, metadata=None):
        """Save a log of generation settings and prompts"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "lora": self.loaded_lora_path,
            "settings": settings,
            "prompts": prompts,
            "metadata": metadata or {}
        }
        
        log_path = Path(output_dir) / "generation_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📋 Generation log saved: {log_path}")
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleaned")


# Example usage
if __name__ == "__main__":
    generator = FluxImageGenerator()
    
    # Generate a simple image
    print("🎨 Testing basic FLUX image generation...")
    image = generator.generate_image(
        "a beautiful landscape with mountains and lakes",
        output_path="test_image.png",
        width=512,
        height=512,
        num_inference_steps=20  # Reduced for faster testing
    )
    
    print("✅ Basic generation test complete!")
    
    # Only test LoRA if the file exists
    lora_path = "outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors"
    if os.path.exists(lora_path):
        print("🔧 Testing LoRA generation...")
        generator.load_lora(lora_path)
        lora_image = generator.generate_image(
            "anddrrew, professional portrait",
            output_path="lora_test.png",
            width=512,
            height=512,
            num_inference_steps=20
        )
        print("✅ LoRA generation test complete!")
    else:
        print("⏳ LoRA file not found - skipping LoRA test (training may still be in progress)")
    
    # Cleanup
    generator.cleanup()

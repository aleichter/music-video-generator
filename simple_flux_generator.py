import os
import torch
from diffusers import FluxPipeline
from PIL import Image
import argparse
from pathlib import Path
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFluxGenerator:
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell", device="cuda", torch_dtype=torch.float16, 
                 lora_path=None, lora_scale=1.0):
        """
        Initialize Flux generator with optional LoRA support
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.lora_loaded = False
        
        logger.info(f"Loading Flux pipeline: {model_name}")
        if lora_path:
            logger.info(f"LoRA will be loaded from: {lora_path}")
            logger.info(f"LoRA scale: {lora_scale}")
        
        # Clear GPU memory first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Check available memory
            free_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"Total GPU memory: {free_memory / 1024**3:.1f} GB")
        
        try:
            # Load with memory optimizations - no device_map for Flux
            self.pipe = FluxPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            
            # Move to device after loading
            self.pipe = self.pipe.to(device)
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            logger.info("Trying with CPU offload enabled...")
            
            # Fallback: try with CPU offload
            try:
                self.pipe = FluxPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                
                # Enable CPU offload for memory efficiency
                if hasattr(self.pipe, 'enable_model_cpu_offload'):
                    self.pipe.enable_model_cpu_offload()
                    logger.info("Enabled CPU offload for memory efficiency")
                else:
                    self.pipe = self.pipe.to(device)
                    
            except Exception as e2:
                logger.error(f"Fallback loading also failed: {e2}")
                raise
        
        # Enable additional memory optimizations
        try:
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        try:
            if hasattr(self.pipe, 'enable_vae_slicing'):
                self.pipe.enable_vae_slicing()
                logger.info("Enabled VAE slicing")
        except Exception as e:
            logger.warning(f"Could not enable VAE slicing: {e}")
        
        try:
            if hasattr(self.pipe, 'enable_vae_tiling'):
                self.pipe.enable_vae_tiling()
                logger.info("Enabled VAE tiling")
        except Exception as e:
            logger.warning(f"Could not enable VAE tiling: {e}")
        
        # Load LoRA if provided
        if lora_path and Path(lora_path).exists():
            self.load_lora(lora_path)
        elif lora_path:
            logger.warning(f"LoRA path provided but file not found: {lora_path}")
        
        logger.info("Flux generator initialized successfully")
    
    def load_lora(self, lora_path):
        """Load LoRA weights into the Flux pipeline"""
        try:
            logger.info(f"Attempting to load LoRA from: {lora_path}")
            
            # Load the checkpoint
            try:
                checkpoint = torch.load(lora_path, map_location='cpu', weights_only=False)
                logger.info("LoRA checkpoint loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LoRA checkpoint: {e}")
                return
            
            # Extract LoRA state dict
            if 'lora_state_dict' in checkpoint:
                lora_state_dict = checkpoint['lora_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                logger.info(f"Found LoRA from epoch {epoch} with {len(lora_state_dict)} parameters")
            else:
                lora_state_dict = checkpoint
                logger.info(f"Loaded LoRA state dict with {len(lora_state_dict)} parameters")
            
            if not lora_state_dict:
                logger.warning("LoRA state dict is empty!")
                return
            
            # Show what LoRA parameters we have
            logger.info("LoRA parameters found:")
            for name, param in list(lora_state_dict.items())[:5]:  # Show first 5
                logger.info(f"  {name}: {param.shape}")
            if len(lora_state_dict) > 5:
                logger.info(f"  ... and {len(lora_state_dict) - 5} more")
            
            # Try to load LoRA into Flux transformer
            if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
                success = self.load_lora_into_transformer(lora_state_dict)
                if success:
                    self.lora_loaded = True
                    logger.info("LoRA loaded successfully into Flux transformer")
                else:
                    logger.warning("LoRA loading failed - continuing without LoRA")
            else:
                logger.warning("No transformer found in pipeline - LoRA loading skipped")
            
        except Exception as e:
            logger.error(f"LoRA loading failed: {e}")
            logger.info("Continuing without LoRA - base model will work normally")
    
    def load_lora_into_transformer(self, lora_state_dict):
        """Try to load LoRA into the Flux transformer"""
        try:
            transformer = self.pipe.transformer
            model_state = transformer.state_dict()
            
            # Get target device and dtype
            target_device = next(transformer.parameters()).device
            target_dtype = next(transformer.parameters()).dtype
            
            logger.info(f"Transformer has {len(model_state)} parameters")
            logger.info(f"Target device: {target_device}, dtype: {target_dtype}")
            
            # Show some transformer parameter names
            logger.info("Transformer parameters (first 5):")
            for name in list(model_state.keys())[:5]:
                logger.info(f"  {name}: {model_state[name].shape}")
            
            # Method 1: Try direct PEFT/LoRA loading
            if hasattr(transformer, 'load_adapter') or hasattr(transformer, 'add_adapter'):
                try:
                    # Convert LoRA to target device/dtype
                    converted_lora = {}
                    for name, param in lora_state_dict.items():
                        converted_lora[name] = param.to(device=target_device, dtype=target_dtype)
                    
                    if hasattr(transformer, 'load_adapter'):
                        transformer.load_adapter(converted_lora)
                        logger.info("Loaded LoRA via load_adapter method")
                        return True
                    elif hasattr(transformer, 'add_adapter'):
                        transformer.add_adapter(converted_lora)
                        logger.info("Loaded LoRA via add_adapter method")
                        return True
                        
                except Exception as e:
                    logger.warning(f"PEFT adapter loading failed: {e}")
            
            # Method 2: Try direct parameter matching and injection
            loaded_count = 0
            for lora_name, lora_param in lora_state_dict.items():
                # Try to find matching transformer parameters
                potential_matches = []
                
                # Clean up the LoRA parameter name to match transformer naming
                clean_name = lora_name.replace('base_model.model.', '').replace('.default', '')
                
                # Look for potential matches in transformer
                for model_name in model_state.keys():
                    if any(part in model_name.lower() for part in clean_name.lower().split('.')):
                        potential_matches.append(model_name)
                
                if potential_matches:
                    logger.info(f"Potential matches for {lora_name}: {potential_matches[:3]}")
                
                # Try exact name match first
                if clean_name in model_state:
                    try:
                        target_param = model_state[clean_name]
                        if target_param.shape == lora_param.shape:
                            # Add LoRA to existing parameter (this is a simplified approach)
                            lora_converted = lora_param.to(device=target_device, dtype=target_dtype)
                            with torch.no_grad():
                                # Scale the LoRA effect
                                target_param.add_(lora_converted * self.lora_scale * 0.1)  # Small scale
                            loaded_count += 1
                            logger.info(f"Injected LoRA into {clean_name}")
                        else:
                            logger.warning(f"Shape mismatch for {clean_name}: {lora_param.shape} vs {target_param.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to inject LoRA into {clean_name}: {e}")
            
            if loaded_count > 0:
                logger.info(f"Successfully injected {loaded_count} LoRA parameters")
                return True
            else:
                logger.warning("No LoRA parameters could be matched to transformer parameters")
                
                # This is expected since our minimal LoRA trainer used a different architecture
                logger.info("This is normal - the LoRA was trained on a different model architecture")
                logger.info("For real Flux LoRA, you need to train on the actual Flux transformer layers")
                return False
                
        except Exception as e:
            logger.error(f"Transformer LoRA loading failed: {e}")
            return False
    
    def generate_image(self, prompt, width=512, height=512, num_inference_steps=4, 
                      guidance_scale=0.0, num_images=1, seed=None, output_dir="./outputs"):
        """
        Generate images using Flux Schnell with optional LoRA
        """
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        logger.info(f"Generating {num_images} image(s) with prompt: '{prompt}'")
        logger.info(f"Resolution: {width}x{height}, Steps: {num_inference_steps}")
        if self.lora_loaded:
            logger.info(f"Using LoRA with scale: {self.lora_scale}")
        else:
            logger.info("Using base model (no LoRA)")
        
        # Monitor memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        try:
            # Clear memory before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate images with minimal settings for Flux Schnell
            with torch.inference_mode():
                logger.info("Starting generation...")
                
                result = self.pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,  # Schnell uses 0.0
                    num_images_per_prompt=num_images,
                    max_sequence_length=256,  # Limit sequence length
                )
                
                images = result.images
                logger.info("Generation completed")
            
            # Log memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Peak GPU memory used: {peak_memory:.1f} GB")
            
            # Save images
            saved_paths = []
            for i, image in enumerate(images):
                # Create filename
                safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_prompt = safe_prompt.replace(' ', '_')
                
                filename = f"{safe_prompt}_{i+1:03d}"
                if seed is not None:
                    filename += f"_seed{seed}"
                if self.lora_loaded and self.lora_path:
                    lora_name = Path(self.lora_path).stem
                    filename += f"_lora_{lora_name}_scale{self.lora_scale}"
                filename += ".png"
                
                filepath = output_dir / filename
                image.save(filepath)
                saved_paths.append(filepath)
                logger.info(f"Saved image: {filepath}")
            
            logger.info(f"Generated {len(images)} image(s) successfully")
            return images, saved_paths
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory: {e}")
            logger.info("Trying with smaller resolution and fewer steps...")
            
            # Emergency fallback with minimal settings
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with torch.inference_mode():
                    result = self.pipe(
                        prompt=prompt,
                        width=256,  # Much smaller
                        height=256,
                        num_inference_steps=1,  # Minimum steps
                        guidance_scale=0.0,
                        num_images_per_prompt=1,  # Just one image
                        max_sequence_length=128,
                    )
                    images = result.images
                
                # Save the emergency image
                filepath = output_dir / "emergency_generation.png"
                images[0].save(filepath)
                logger.info(f"Emergency low-res generation saved: {filepath}")
                return images, [filepath]
                
            except Exception as e2:
                logger.error(f"Even emergency generation failed: {e2}")
                raise
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    def generate_comparison(self, prompt, num_images=4, **kwargs):
        """Generate comparison images to test LoRA effect"""
        logger.info("Generating comparison images...")
        
        # Generate with current settings (with or without LoRA)
        images, paths = self.generate_image(prompt, num_images=num_images, **kwargs)
        
        # Create a comparison grid
        if len(images) > 1:
            grid_path = Path(kwargs.get('output_dir', './outputs')) / f"comparison_grid_{'lora' if self.lora_loaded else 'base'}.png"
            self.create_comparison_grid(images, grid_path)
        
        return images, paths
    
    def create_comparison_grid(self, images, output_path, grid_size=None):
        """Create a grid comparison of generated images"""
        if not images:
            logger.warning("No images provided for grid creation")
            return None
        
        # Determine grid size
        if grid_size is None:
            cols = int(len(images) ** 0.5)
            rows = (len(images) + cols - 1) // cols
        else:
            cols, rows = grid_size
        
        # Get image dimensions
        img_width, img_height = images[0].size
        
        # Create grid image
        grid_width = cols * img_width
        grid_height = rows * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Place images in grid
        for i, img in enumerate(images):
            if i >= cols * rows:
                break
            
            row = i // cols
            col = i % cols
            
            x = col * img_width
            y = row * img_height
            
            grid_image.paste(img, (x, y))
        
        # Save grid
        grid_image.save(output_path)
        logger.info(f"Saved comparison grid: {output_path}")
        
        return grid_image

def main():
    parser = argparse.ArgumentParser(description="Simple Flux Image Generator with LoRA Support")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-schnell", help="Model name")
    parser.add_argument("--lora_path", help="Path to LoRA checkpoint file")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale factor (0.0-2.0)")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Model dtype")
    parser.add_argument("--create_grid", action="store_true", help="Create comparison grid if multiple images")
    parser.add_argument("--comparison", action="store_true", help="Generate comparison images")
    
    args = parser.parse_args()
    
    # Convert dtype
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    try:
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory = gpu_props.total_memory / 1024**3
            logger.info(f"GPU: {gpu_props.name}")
            logger.info(f"GPU memory: {gpu_memory:.1f} GB")
            
            # Clear memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Check if we have enough memory for Flux
            if gpu_memory < 12:
                logger.warning(f"GPU has only {gpu_memory:.1f} GB memory. Flux may not work well.")
                logger.info("Consider using a smaller model or reduce resolution/steps")
        
        # Initialize generator
        generator = SimpleFluxGenerator(
            model_name=args.model_name,
            device=args.device,
            torch_dtype=torch_dtype,
            lora_path=args.lora_path,
            lora_scale=args.lora_scale,
        )
        
        # Generate images
        if args.comparison:
            images, paths = generator.generate_comparison(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                num_images=args.num_images,
                seed=args.seed,
                output_dir=args.output_dir
            )
        else:
            images, paths = generator.generate_image(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                num_images=args.num_images,
                seed=args.seed,
                output_dir=args.output_dir
            )
        
        # Create grid if requested and multiple images
        if args.create_grid and len(images) > 1:
            grid_path = Path(args.output_dir) / "comparison_grid.png"
            generator.create_comparison_grid(images, grid_path)
        
        logger.info("Generation completed successfully!")
        
        # Print generated file paths
        print("\nGenerated images:")
        for path in paths:
            print(f"  {path}")
        
        # Print LoRA status
        if args.lora_path:
            if generator.lora_loaded:
                print(f"\n✅ LoRA loaded successfully from: {args.lora_path}")
            else:
                print(f"\n⚠️  LoRA loading failed, using base model only")
                print("Note: The LoRA may be incompatible with Flux architecture")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        
        # Suggest alternatives
        logger.info("\nIf this continues to fail, try:")
        logger.info("1. Reduce resolution: --width 256 --height 256")
        logger.info("2. Reduce steps: --num_inference_steps 1")
        logger.info("3. Use a different model like Stable Diffusion XL")
        
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
FLUX LoRA Image Generator
Automatically discovers trained LoRA models and generates images with custom prompts
"""

import os
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image
import gc


class FluxLoRAGenerator:
    """Generate images using trained FLUX LoRA models"""
    
    def __init__(self, outputs_dir="outputs", cache_dir="/workspace/.cache/huggingface"):
        """Initialize the FLUX LoRA generator"""
        self.outputs_dir = Path(outputs_dir)
        self.cache_dir = cache_dir
        self.pipeline = None
        self.current_lora = None
        
        # Set environment
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HUB_CACHE'] = cache_dir
        
    def discover_lora_models(self):
        """Discover available LoRA models with all epochs in outputs directory"""
        if not self.outputs_dir.exists():
            print(f"❌ Outputs directory not found: {self.outputs_dir}")
            return {}
        
        models = {}
        print(f"🔍 Discovering LoRA models in: {self.outputs_dir}")
        
        for model_dir in self.outputs_dir.iterdir():
            if model_dir.is_dir() and model_dir.name != "generated_images":
                # Look for .safetensors files
                safetensors_files = list(model_dir.glob("*.safetensors"))
                
                if safetensors_files:
                    # Try to load training info
                    info_file = model_dir / "training_info.json"
                    info = {}
                    if info_file.exists():
                        try:
                            with open(info_file, 'r') as f:
                                info = json.load(f)
                        except Exception as e:
                            print(f"⚠️ Could not read training info for {model_dir.name}: {e}")
                    
                    # Organize files by type (final model vs epochs)
                    epochs = {}
                    final_model = None
                    
                    for file in safetensors_files:
                        filename = file.name
                        if filename.endswith("-000004.safetensors") or filename.endswith("-000008.safetensors") or filename.endswith("-000012.safetensors") or "-" in filename and filename.count("-") >= 1:
                            # This is likely an epoch checkpoint
                            # Extract epoch number from filename like "modelname-000004.safetensors"
                            parts = filename.replace(".safetensors", "").split("-")
                            if len(parts) >= 2 and parts[-1].isdigit():
                                epoch_num = int(parts[-1])
                                epochs[epoch_num] = str(file)
                        else:
                            # This is likely the final model
                            final_model = str(file)
                    
                    # If we have a final model, add it as the latest epoch
                    if final_model:
                        epochs["final"] = final_model
                    
                    if epochs:
                        models[model_dir.name] = {
                            "epochs": epochs,
                            "trigger_word": info.get("trigger_word", ""),
                            "training_time": info.get("training_time_minutes", "unknown"),
                            "timestamp": info.get("timestamp", "unknown"),
                            "method": info.get("method", "unknown"),
                            "default_path": final_model or list(epochs.values())[-1]
                        }
                        
                        print(f"  ✅ Found: {model_dir.name}")
                        print(f"     Epochs: {list(epochs.keys())}")
                        print(f"     Trigger: {info.get('trigger_word', 'N/A')}")
        
        if not models:
            print("❌ No LoRA models found in outputs directory")
            
        return models
    
    def list_models(self):
        """List all available LoRA models"""
        models = self.discover_lora_models()
        
        if not models:
            return
            
        print(f"\n📋 Available LoRA Models ({len(models)} found):")
        print("=" * 60)
        
        for name, info in models.items():
            print(f"🎭 Model: {name}")
            print(f"   Trigger Word: {info['trigger_word'] or 'N/A'}")
            print(f"   Available Epochs: {list(info['epochs'].keys())}")
            print(f"   Training Time: {info['training_time']} minutes")
            print(f"   Method: {info['method']}")
            print(f"   Created: {info['timestamp']}")
            print()
    
    def list_epochs(self, model_name):
        """List all available epochs for a specific model"""
        models = self.discover_lora_models()
        
        if model_name not in models:
            print(f"❌ Model '{model_name}' not found. Available models: {list(models.keys())}")
            return
        
        model_info = models[model_name]
        epochs = model_info['epochs']
        
        print(f"\n📋 Available Epochs for Model '{model_name}':")
        print("=" * 50)
        print(f"🎭 Trigger Word: {model_info['trigger_word']}")
        print(f"📊 Training Time: {model_info['training_time']} minutes")
        print()
        
        # Sort epochs for better display
        sorted_epochs = []
        for epoch in epochs.keys():
            if epoch == "final":
                sorted_epochs.append(("final", epochs[epoch]))
            else:
                sorted_epochs.append((epoch, epochs[epoch]))
        
        # Sort by epoch number, with final last
        sorted_epochs.sort(key=lambda x: (x[0] == "final", x[0] if x[0] != "final" else float('inf')))
        
        for epoch, path in sorted_epochs:
            if epoch == "final":
                print(f"🏁 Epoch: final (latest)")
            else:
                print(f"📈 Epoch: {epoch}")
            print(f"   Path: {path}")
            print()
    
    def load_pipeline(self, device="auto"):
        """Load the FLUX pipeline"""
        if self.pipeline is not None:
            return
            
        print("🚀 Loading FLUX pipeline...")
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"   Device: {device}")
        
        try:
            # Load FLUX pipeline with proper device mapping
            if device == "cuda":
                # Use balanced device mapping for FLUX on CUDA
                self.pipeline = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    torch_dtype=torch.bfloat16,
                    cache_dir=self.cache_dir,
                    device_map="balanced"
                )
            else:
                # For CPU, load without device mapping
                self.pipeline = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    torch_dtype=torch.float32,
                    cache_dir=self.cache_dir,
                    device_map=None
                )
                self.pipeline = self.pipeline.to(device)
                
            print("✅ FLUX pipeline loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load FLUX pipeline: {e}")
            raise
    
    def load_lora(self, model_name, epoch=None):
        """Load a specific LoRA model, optionally specifying an epoch"""
        models = self.discover_lora_models()
        
        if model_name not in models:
            raise ValueError(f"LoRA model '{model_name}' not found. Available: {list(models.keys())}")
        
        model_info = models[model_name]
        epochs = model_info["epochs"]
        
        # Determine which epoch to use
        if epoch is None:
            # Use default (final or latest available)
            epoch_key = "final" if "final" in epochs else max(epochs.keys())
            lora_path = epochs[epoch_key]
        else:
            # User specified an epoch
            if epoch == "final":
                if "final" not in epochs:
                    raise ValueError(f"Model '{model_name}' does not have a final epoch. Available: {list(epochs.keys())}")
                lora_path = epochs["final"]
                epoch_key = "final"
            else:
                # Convert epoch to int if it's a string
                try:
                    epoch_num = int(epoch)
                    if epoch_num not in epochs:
                        raise ValueError(f"Epoch {epoch_num} not found for model '{model_name}'. Available epochs: {list(epochs.keys())}")
                    lora_path = epochs[epoch_num]
                    epoch_key = epoch_num
                except ValueError:
                    raise ValueError(f"Invalid epoch '{epoch}'. Must be an integer or 'final'. Available: {list(epochs.keys())}")
        
        print(f"🎭 Loading LoRA: {model_name}")
        print(f"   Epoch: {epoch_key}")
        print(f"   Path: {lora_path}")
        print(f"   Trigger: {model_info['trigger_word']}")
        
        try:
            # Unload previous LoRA if any
            if self.current_lora:
                self.pipeline.unload_lora_weights()
                
            # Load new LoRA
            self.pipeline.load_lora_weights(lora_path)
            self.current_lora = f"{model_name}@{epoch_key}"
            
            print(f"✅ LoRA '{model_name}' epoch {epoch_key} loaded successfully")
            
            # Return model info with epoch details
            model_info_with_epoch = model_info.copy()
            model_info_with_epoch["used_epoch"] = epoch_key
            model_info_with_epoch["used_model_path"] = lora_path
            return model_info_with_epoch
            
        except Exception as e:
            print(f"❌ Failed to load LoRA '{model_name}' epoch {epoch_key}: {e}")
            raise
    
    def generate_image(self, prompt, model_name=None, epoch=None, output_path=None, **kwargs):
        """Generate an image using the loaded LoRA model"""
        
        # Load pipeline if not already loaded
        if self.pipeline is None:
            self.load_pipeline()
        
        # Load LoRA if specified and different from current
        model_info = None
        used_epoch = None
        used_model_path = None
        
        current_model_epoch = f"{model_name}@{epoch}" if epoch else model_name
        if model_name and current_model_epoch != self.current_lora:
            model_info = self.load_lora(model_name, epoch)
            used_epoch = model_info.get("used_epoch")
            used_model_path = model_info.get("used_model_path")
        elif model_name:
            models = self.discover_lora_models()
            model_info = models.get(model_name, {})
            
            # If using already loaded LoRA, determine epoch from current_lora
            if self.current_lora and "@" in self.current_lora:
                _, loaded_epoch = self.current_lora.split("@", 1)
                used_epoch = loaded_epoch
                # Find the path for this epoch
                epochs = model_info.get("epochs", {})
                if loaded_epoch == "final":
                    used_model_path = epochs.get("final")
                else:
                    try:
                        epoch_num = int(loaded_epoch)
                        used_model_path = epochs.get(epoch_num)
                    except ValueError:
                        pass
        
        # Handle seed generation - use random if not specified
        seed = kwargs.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Generation parameters
        generation_params = {
            "prompt": prompt,
            "height": kwargs.get("height", 1024),
            "width": kwargs.get("width", 1024),
            "num_inference_steps": kwargs.get("steps", 20),
            "guidance_scale": kwargs.get("guidance_scale", 3.5),
            "num_images_per_prompt": kwargs.get("num_images", 1),
            "generator": torch.Generator().manual_seed(seed)
        }
        
        print(f"🎨 Generating image...")
        print(f"   Model: {model_name or 'Base FLUX'}")
        print(f"   Prompt: {prompt}")
        print(f"   Size: {generation_params['width']}x{generation_params['height']}")
        print(f"   Steps: {generation_params['num_inference_steps']}")
        print(f"   Guidance: {generation_params['guidance_scale']}")
        print(f"   Seed: {seed}")  
        
        try:
            # Generate image
            with torch.no_grad():
                result = self.pipeline(**generation_params)
                image = result.images[0]
            
            # Save image in organized directory structure
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_suffix = f"_{model_name}" if model_name else "_base"
                # Create organized path in outputs/generated_images/
                generated_dir = self.outputs_dir / "generated_images"
                output_path = generated_dir / f"generated{model_suffix}_{timestamp}.png"
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            image.save(output_path)
            
            print(f"✅ Image generated successfully!")
            print(f"   Saved: {output_path}")
            
            # Save generation info
            info = {
                "prompt": prompt,
                "model": model_name,
                "epoch": used_epoch,
                "model_path": used_model_path,
                "trigger_word": model_info.get("trigger_word", "") if model_info else "",
                "parameters": generation_params,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "output_path": str(output_path)
            }
            
            info_path = output_path.with_suffix('.json')
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
            raise
    
    def validate_and_truncate_prompt(self, prompt, max_tokens=75, truncate=True):
        """
        Validate prompt length and optionally truncate if too long
        CLIP has a limit of 77 tokens, we use 75 to be safe
        """
        # Simple token estimation (not perfect but close enough)
        # Real tokenization would require loading the tokenizer
        words = prompt.split()
        estimated_tokens = len(words) + len([c for c in prompt if c in '.,!?;:"()[]'])
        
        if estimated_tokens <= max_tokens:
            return prompt, False  # No truncation needed
        
        print(f"⚠️ Prompt is too long (~{estimated_tokens} tokens, max {max_tokens})")
        
        if not truncate:
            print("❌ Prompt truncation disabled. Please shorten your prompt.")
            return None, True
        
        # Truncate by words, keeping the most important parts
        target_words = int(len(words) * (max_tokens / estimated_tokens * 0.9))  # 90% safety margin
        truncated_words = words[:target_words]
        truncated_prompt = ' '.join(truncated_words)
        
        print(f"✂️ Prompt truncated to ~{len(truncated_words)} words:")
        print(f"   Original: {prompt[:100]}...")
        print(f"   Truncated: {truncated_prompt[:100]}...")
        
        return truncated_prompt, True
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            self.current_lora = None
        
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Generate images with FLUX LoRA models")
    
    # Model selection
    parser.add_argument("--model", type=str,
                       help="LoRA model name to use (will be discovered from outputs directory)")
    parser.add_argument("--epoch", type=str,
                       help="Specific epoch to use (e.g., 4, 8, 12, or 'final'). If not specified, uses final/latest.")
    parser.add_argument("--list-models", action="store_true",
                       help="List all available LoRA models and exit")
    parser.add_argument("--list-epochs", type=str, metavar="MODEL_NAME",
                       help="List all available epochs for a specific model and exit")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=False,
                       help="Text prompt for image generation")
    parser.add_argument("--prompt-file", type=str,
                       help="Path to text file containing the prompt (alternative to --prompt)")
    parser.add_argument("--no-truncate", action="store_true",
                       help="Disable automatic prompt truncation (will fail if prompt is too long)")
    parser.add_argument("--max-tokens", type=int, default=75,
                       help="Maximum tokens for prompt (default: 75, CLIP limit is 77)")
    parser.add_argument("--output", type=str,
                       help="Output image path (auto-generated if not specified)")
    
    # Image parameters
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height (default: 1024)")
    parser.add_argument("--steps", type=int, default=40,
                       help="Number of inference steps (default: 20)")
    parser.add_argument("--guidance-scale", type=float, default=3.5,
                       help="Guidance scale (default: 3.5)")
    parser.add_argument("--seed", type=int,
                       help="Random seed (if not specified, a random seed will be used)")
    parser.add_argument("--num-images", type=int, default=1,
                       help="Number of images to generate (default: 1)")
    
    # System parameters
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                       help="Directory containing LoRA models")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = FluxLoRAGenerator(outputs_dir=args.outputs_dir)
    
    try:
        if args.list_models:
            generator.list_models()
            return
        
        if args.list_epochs:
            generator.list_epochs(args.list_epochs)
            return
        
        # Handle prompt input (from argument or file)
        prompt = None
        if args.prompt:
            prompt = args.prompt
        elif args.prompt_file:
            try:
                with open(args.prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                print(f"📝 Loaded prompt from file: {args.prompt_file}")
            except FileNotFoundError:
                print(f"❌ Prompt file not found: {args.prompt_file}")
                return
            except Exception as e:
                print(f"❌ Error reading prompt file: {e}")
                return
        
        if not prompt:
            print("❌ Prompt is required for image generation")
            print("Use --prompt 'your text' or --prompt-file path/to/file.txt")
            print("Use --list-models to see available models")
            print("Use --list-epochs MODEL_NAME to see available epochs")
            return
        
        # Validate and optionally truncate prompt
        validated_prompt, was_truncated = generator.validate_and_truncate_prompt(
            prompt, 
            max_tokens=args.max_tokens, 
            truncate=not args.no_truncate
        )
        
        if validated_prompt is None:
            print("❌ Cannot proceed with prompt that is too long")
            return
        
        prompt = validated_prompt
        
        # Load pipeline
        generator.load_pipeline(device=args.device)
        
        # Generate image
        output_path = generator.generate_image(
            prompt=prompt,
            model_name=args.model,
            epoch=args.epoch,
            output_path=args.output,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            num_images=args.num_images
        )
        
        print(f"🎉 Generation complete! Image saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()

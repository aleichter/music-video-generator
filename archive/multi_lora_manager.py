#!/usr/bin/env python3

import torch
import os
from diffusers import FluxPipeline
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

class MultiLoRAManager:
    """
    Manages multiple LoRA models for FLUX using the merge approach.
    Supports dynamic combinations of character and style LoRAs.
    """
    
    def __init__(self, base_model_path: str = "black-forest-labs/FLUX.1-dev"):
        self.base_model_path = base_model_path
        self.pipe = None
        self.current_loras = []  # Track what's currently loaded
        self.lora_registry = {}  # Registry of available LoRAs
        
    def initialize_pipeline(self, device: str = "cuda"):
        """Initialize the FLUX pipeline"""
        print("Loading FLUX pipeline...")
        self.pipe = FluxPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(device)
        print("Pipeline loaded successfully!")
        
    def register_lora(self, name: str, path: str, category: str = "general", description: str = ""):
        """Register a LoRA for easy access"""
        self.lora_registry[name] = {
            "path": path,
            "category": category,
            "description": description
        }
        
    def list_loras(self, category: Optional[str] = None) -> Dict:
        """List available LoRAs, optionally filtered by category"""
        if category:
            return {k: v for k, v in self.lora_registry.items() if v["category"] == category}
        return self.lora_registry
        
    def reset_to_base(self):
        """Reset pipeline to base model (remove all LoRAs)"""
        if self.pipe and self.current_loras:
            print("Resetting to base model...")
            self.pipe.unload_lora_weights()
            self.current_loras = []
            print("Reset complete!")
            
    def apply_lora_combination(self, lora_configs: List[Dict]):
        """
        Apply a combination of LoRAs with specified strengths.
        
        Args:
            lora_configs: List of dicts with keys: 'name', 'strength' (optional, default 1.0)
        """
        if not self.pipe:
            raise ValueError("Pipeline not initialized. Call initialize_pipeline() first.")
            
        # Reset to base model first
        self.reset_to_base()
        
        print(f"Applying {len(lora_configs)} LoRAs...")
        
        for config in lora_configs:
            lora_name = config["name"]
            strength = config.get("strength", 1.0)
            
            if lora_name not in self.lora_registry:
                raise ValueError(f"LoRA '{lora_name}' not found in registry")
                
            lora_path = self.lora_registry[lora_name]["path"]
            print(f"  Loading {lora_name} (strength: {strength})...")
            
            # Load and merge the LoRA
            self.pipe.load_lora_weights(lora_path)
            self.pipe.fuse_lora(lora_scale=strength)
            
            self.current_loras.append({"name": lora_name, "strength": strength})
            
        print("LoRA combination applied successfully!")
        
    def generate_image(self, 
                      prompt: str, 
                      num_inference_steps: int = 20,
                      guidance_scale: float = 3.5,
                      width: int = 1024,
                      height: int = 1024,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate image with current LoRA combination"""
        if not self.pipe:
            raise ValueError("Pipeline not initialized. Call initialize_pipeline() first.")
            
        print(f"Generating image with prompt: '{prompt}'")
        if self.current_loras:
            print(f"Active LoRAs: {[f'{l['name']}({l['strength']})' for l in self.current_loras]}")
        else:
            print("Using base model (no LoRAs)")
            
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
            
        return result.images[0]
        
    def save_lora_config(self, config_name: str, lora_configs: List[Dict], output_dir: str = "lora_configs"):
        """Save a LoRA combination configuration for reuse"""
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f"{config_name}.json")
        
        config_data = {
            "name": config_name,
            "loras": lora_configs,
            "description": f"LoRA combination: {', '.join([c['name'] for c in lora_configs])}"
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        print(f"Configuration saved to {config_path}")
        
    def load_lora_config(self, config_name: str, config_dir: str = "lora_configs") -> List[Dict]:
        """Load a saved LoRA combination configuration"""
        config_path = os.path.join(config_dir, f"{config_name}.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration '{config_name}' not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        return config_data["loras"]


def demo_multi_lora_workflow():
    """Demonstrate the multi-LoRA workflow"""
    
    # Create output directory
    output_dir = "multi_lora_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}/")
    
    # Initialize manager
    manager = MultiLoRAManager()
    manager.initialize_pipeline()
    
    # Register your LoRAs (update these paths to your actual LoRAs)
    manager.register_lora(
        name="anddrrew_character",
        path="./models/anddrrew_fixed_flux_lora",
        category="character",
        description="Andrew character LoRA"
    )
    
    # You can add more LoRAs as you create them:
    # manager.register_lora(
    #     name="anime_style",
    #     path="./models/anime_style_lora",
    #     category="style",
    #     description="Anime art style"
    # )
    # 
    # manager.register_lora(
    #     name="cyberpunk_style", 
    #     path="./models/cyberpunk_style_lora",
    #     category="style",
    #     description="Cyberpunk aesthetic"
    # )
    
    print("Available LoRAs:")
    for name, info in manager.list_loras().items():
        print(f"  {name}: {info['description']} ({info['category']})")
    
    # Test 1: Base model (no LoRAs)
    print("\n=== Test 1: Base Model ===")
    manager.reset_to_base()
    base_image = manager.generate_image(
        "a person standing in a futuristic city",
        generator=torch.Generator().manual_seed(42)
    )
    base_image.save(os.path.join(output_dir, "output_base_model.png"))
    
    # Test 2: Single LoRA
    print("\n=== Test 2: Character LoRA Only ===")
    manager.apply_lora_combination([
        {"name": "anddrrew_character", "strength": 1.0}
    ])
    char_image = manager.generate_image(
        "a person standing in a futuristic city",
        generator=torch.Generator().manual_seed(42)
    )
    char_image.save(os.path.join(output_dir, "output_character_lora.png"))
    
    # Test 3: Multiple LoRAs (when you have style LoRAs)
    # print("\n=== Test 3: Character + Style LoRAs ===")
    # manager.apply_lora_combination([
    #     {"name": "anddrrew_character", "strength": 1.0},
    #     {"name": "anime_style", "strength": 0.8}
    # ])
    # combo_image = manager.generate_image(
    #     "a person standing in a futuristic city",
    #     generator=torch.Generator().manual_seed(42)
    # )
    # combo_image.save(os.path.join(output_dir, "output_character_plus_style.png"))
    
    # Save configuration for reuse
    manager.save_lora_config("character_only", [
        {"name": "anddrrew_character", "strength": 1.0}
    ])
    
    print(f"\nDemo complete! Check the generated images in {output_dir}/.")


if __name__ == "__main__":
    demo_multi_lora_workflow()

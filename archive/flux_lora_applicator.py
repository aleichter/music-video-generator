#!/usr/bin/env python3
"""
FLUX LoRA Loader Utility
Loads and applies Kohya-format LoRA weights to FLUX models
"""

import torch
from safetensors import safe_open

class FluxLoRAApplicator:
    """Apply trained LoRA weights to FLUX models"""
    
    def __init__(self, lora_path, scale=1.0):
        self.lora_path = lora_path
        self.scale = scale
        self.weights = self._load_weights()
        
    def _load_weights(self):
        """Load LoRA weights from safetensors file"""
        weights = {}
        with safe_open(self.lora_path, framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        print(f"Loaded {len(weights)} LoRA parameters from {self.lora_path}")
        return weights
    
    def apply_to_module(self, module, module_name, prefix=""):
        """Apply LoRA weights to a specific module"""
        if not hasattr(module, 'weight'):
            return False
            
        # Construct possible LoRA key names
        base_name = f"{prefix}{module_name}" if prefix else module_name
        down_key = f"{base_name}.lora_down.weight"
        up_key = f"{base_name}.lora_up.weight"
        alpha_key = f"{base_name}.alpha"
        
        # Find matching keys
        down_weight = None
        up_weight = None
        alpha = 1.0
        
        for key in self.weights.keys():
            if down_key in key:
                down_weight = self.weights[key]
            elif up_key in key:
                up_weight = self.weights[key]
            elif alpha_key in key:
                alpha = self.weights[key].item()
        
        if down_weight is not None and up_weight is not None:
            # Apply LoRA: W = W + alpha * up @ down * scale
            device = module.weight.device
            down_weight = down_weight.to(device)
            up_weight = up_weight.to(device)
            
            lora_delta = alpha * torch.mm(up_weight, down_weight) * self.scale
            module.weight.data += lora_delta
            
            print(f"Applied LoRA to {module_name} (alpha={alpha:.2f}, scale={self.scale})")
            return True
        
        return False
    
    def apply_to_transformer(self, transformer):
        """Apply LoRA to FLUX transformer"""
        applied_count = 0
        
        for name, module in transformer.named_modules():
            if self.apply_to_module(module, name, "lora_unet_"):
                applied_count += 1
        
        print(f"Applied LoRA to {applied_count} transformer modules")
        return transformer
    
    def apply_to_text_encoder(self, text_encoder):
        """Apply LoRA to text encoder"""
        applied_count = 0
        
        for name, module in text_encoder.named_modules():
            if self.apply_to_module(module, name, "lora_te1_"):
                applied_count += 1
        
        print(f"Applied LoRA to {applied_count} text encoder modules")
        return text_encoder

# Example usage:
def apply_lora_to_flux_pipeline(pipe, lora_path, scale=1.0):
    """Apply LoRA to a FLUX pipeline"""
    applicator = FluxLoRAApplicator(lora_path, scale)
    
    # Apply to transformer (main model)
    pipe.transformer = applicator.apply_to_transformer(pipe.transformer)
    
    # Apply to text encoder if present
    if hasattr(pipe, 'text_encoder'):
        pipe.text_encoder = applicator.apply_to_text_encoder(pipe.text_encoder)
    
    print("LoRA application complete!")
    return pipe

if __name__ == "__main__":
    # Test the loader
    lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
    applicator = FluxLoRAApplicator(lora_path)
    print("LoRA loader test successful!")

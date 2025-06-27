#!/usr/bin/env python3
"""
Production FLUX LoRA Image Generator
Successfully loads and applies LoRA weights from sd-scripts format to diffusers FLUX pipeline.
"""

import torch
import safetensors.torch
from flux_image_generator import FluxImageGenerator

class FluxLoRAGenerator:
    def __init__(self):
        self.generator = FluxImageGenerator()
        self.lora_loaded = False
        self.applied_layers = 0
        
    def create_mapping_table(self):
        """Create mapping from sd-scripts naming to diffusers naming"""
        mapping = {}
        
        # Map double_blocks (transformer_blocks in diffusers)
        for i in range(19):  # FLUX has 19 transformer blocks
            # Attention projections
            mapping[f'double_blocks_{i}_img_attn_proj'] = f'transformer_blocks.{i}.attn.to_out.0'
            mapping[f'double_blocks_{i}_txt_attn_proj'] = f'transformer_blocks.{i}.attn.to_add_out'
            
            # MLP layers
            mapping[f'double_blocks_{i}_img_mlp_2'] = f'transformer_blocks.{i}.ff.net.2'
            mapping[f'double_blocks_{i}_txt_mlp_2'] = f'transformer_blocks.{i}.ff_context.net.2'
            
            # Modulation layers
            mapping[f'double_blocks_{i}_img_mod_lin'] = f'transformer_blocks.{i}.norm1.linear'
            mapping[f'double_blocks_{i}_txt_mod_lin'] = f'transformer_blocks.{i}.norm1_context.linear'
        
        # Map single_blocks (single_transformer_blocks in diffusers)
        for i in range(38):  # FLUX has 38 single transformer blocks
            mapping[f'single_blocks_{i}_linear2'] = f'single_transformer_blocks.{i}.proj_out'
            mapping[f'single_blocks_{i}_modulation_lin'] = f'single_transformer_blocks.{i}.norm.linear'
        
        return mapping

    def load_lora(self, lora_path, verbose=True):
        """Load and apply LoRA weights to the FLUX transformer"""
        if verbose:
            print(f"🔄 Loading LoRA from: {lora_path}")

        # Load LoRA weights
        with safetensors.torch.safe_open(lora_path, framework='pt') as f:
            lora_dict = {k: f.get_tensor(k) for k in f.keys()}

        if verbose:
            print(f"📦 Loaded {len(lora_dict)} LoRA weights")

        # Ensure pipeline is loaded and valid
        if not hasattr(self.generator, 'pipeline') or self.generator.pipeline is None:
            if verbose:
                print("ℹ️ Pipeline not loaded yet. Loading now...")
            self.generator.load_pipeline()
        if not hasattr(self.generator, 'pipeline') or self.generator.pipeline is None:
            raise RuntimeError("❌ Could not load pipeline. Cannot apply LoRA.")

        transformer = getattr(self.generator.pipeline, 'transformer', None)
        if transformer is None:
            raise RuntimeError("❌ Pipeline loaded but transformer is missing. Cannot apply LoRA.")

        mapping = self.create_mapping_table()

        # Group LoRA weights by layer
        lora_layers = {}
        for key, weight in lora_dict.items():
            if 'lora_unet_' in key:
                clean_key = key.replace('lora_unet_', '')
                parts = clean_key.split('.')
                if len(parts) >= 3:
                    layer_path = '.'.join(parts[:-2])
                    lora_type = parts[-2]
                    if layer_path not in lora_layers:
                        lora_layers[layer_path] = {}
                    lora_layers[layer_path][lora_type] = weight

        # Apply LoRA weights using proper mapping
        applied_count = 0
        failed_count = 0

        for sd_layer_path, lora_weights in lora_layers.items():
            if 'lora_down' in lora_weights and 'lora_up' in lora_weights:
                if sd_layer_path in mapping:
                    diffusers_path = mapping[sd_layer_path]
                    try:
                        # Navigate to target layer
                        target_module = transformer
                        for part in diffusers_path.split('.'):
                            target_module = getattr(target_module, part)
                        if hasattr(target_module, 'weight'):
                            lora_down = lora_weights['lora_down']
                            lora_up = lora_weights['lora_up']
                            lora_weight = torch.mm(lora_up, lora_down)
                            if lora_weight.shape == target_module.weight.shape:
                                target_module.weight.data += lora_weight
                                applied_count += 1
                            else:
                                failed_count += 1
                                if verbose:
                                    print(f"⚠️ Shape mismatch: {sd_layer_path}")
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        failed_count += 1
                        if verbose:
                            print(f"❌ Error applying {sd_layer_path}: {e}")
                else:
                    failed_count += 1
        
        self.applied_layers = applied_count
        self.lora_loaded = True
        
        if verbose:
            print(f"✅ LoRA loaded successfully!")
            print(f"   Applied: {applied_count} layers")
            print(f"   Failed: {failed_count} layers")
            print(f"   Success rate: {applied_count/(applied_count+failed_count)*100:.1f}%")
        
        return applied_count, failed_count

    def generate_image(self, prompt, output_path=None, **kwargs):
        """Generate image with LoRA applied"""
        if not self.lora_loaded:
            print("⚠️ Warning: No LoRA loaded. Generating with base model.")
        
        return self.generator.generate_image(prompt, output_path, **kwargs)

    def generate_comparison(self, prompt, lora_path, base_name="comparison", **kwargs):
        """Generate side-by-side comparison with and without LoRA"""
        print(f"🎨 Generating comparison for: {prompt}")
        
        # Generate with LoRA
        self.load_lora(lora_path, verbose=False)
        lora_image = self.generate_image(
            prompt, 
            output_path=f"{base_name}_with_lora.png", 
            **kwargs
        )
        
        # Reset to generate without LoRA (reload fresh pipeline)
        self.generator = FluxImageGenerator()
        self.lora_loaded = False
        baseline_image = self.generate_image(
            prompt,
            output_path=f"{base_name}_baseline.png",
            **kwargs
        )
        
        print(f"📊 Comparison generated:")
        print(f"   With LoRA: {base_name}_with_lora.png")
        print(f"   Baseline: {base_name}_baseline.png")
        
        return lora_image, baseline_image

def main():
    """Demo the LoRA generator"""
    generator = FluxLoRAGenerator()
    
    # Load LoRA
    lora_path = 'outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors'
    generator.load_lora(lora_path)
    
    # Generate test images
    test_prompts = [
        "anddrrew, professional headshot, studio lighting",
        "anddrrew in casual clothing, outdoor setting",
        "anddrrew smiling, close-up portrait"
    ]
    
    print("\n🎨 Generating test images with LoRA...")
    for i, prompt in enumerate(test_prompts):
        output_path = f"lora_demo_{i+1}.png"
        generator.generate_image(
            prompt,
            output_path=output_path,
            width=512, 
            height=512, 
            num_inference_steps=20,
            seed=42 + i
        )
        print(f"✅ Generated: {output_path}")
    
    print("\n🎉 LoRA demo complete!")

if __name__ == "__main__":
    main()

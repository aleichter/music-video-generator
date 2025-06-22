import torch
import gc
from pathlib import Path
from diffusers import FluxPipeline
import argparse
from PIL import Image

class LoRATestLoader:
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell"):
        """Load Flux pipeline for testing LoRA"""
        self.model_name = model_name
        
        print(f"Loading Flux pipeline: {model_name}")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load pipeline
        self.pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        
        # Enable optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.enable_vae_tiling()
        
        print("✅ Flux pipeline loaded successfully")
        
        # Store original weights for comparison
        self.original_weights = {}
        self.lora_layers = []
    
    def load_lora_weights(self, checkpoint_path):
        """Load and apply LoRA weights to the transformer"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading LoRA checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        
        print(f"Found {len(lora_state_dict)} LoRA parameters")
        
        # Apply LoRA weights to transformer
        transformer = self.pipe.transformer
        
        # Target the same layers we trained
        target_patterns = [
            'transformer_blocks.0.attn.to_q',
            'transformer_blocks.0.attn.to_v',
        ]
        
        layer_idx = 0
        for pattern in target_patterns:
            for name, module in transformer.named_modules():
                if hasattr(module, 'weight') and pattern in name:
                    print(f"Applying LoRA to: {name}")
                    
                    # Get LoRA weights
                    lora_A_key = f"lora_layer_{layer_idx}.lora_A"
                    lora_B_key = f"lora_layer_{layer_idx}.lora_B"
                    
                    if lora_A_key in lora_state_dict and lora_B_key in lora_state_dict:
                        lora_A = lora_state_dict[lora_A_key].to(module.weight.device, dtype=module.weight.dtype)
                        lora_B = lora_state_dict[lora_B_key].to(module.weight.device, dtype=module.weight.dtype)
                        
                        # Store original weights
                        if name not in self.original_weights:
                            self.original_weights[name] = module.weight.data.clone()
                        
                        # Apply LoRA: W = W_original + (B @ A) * scaling
                        scaling = 16 / 4  # alpha / rank = 4.0
                        lora_weight = (lora_B @ lora_A) * scaling
                        
                        # Add LoRA to original weights
                        module.weight.data = self.original_weights[name] + lora_weight
                        
                        print(f"  Applied LoRA layer {layer_idx}")
                        print(f"  LoRA A range: [{lora_A.min():.6f}, {lora_A.max():.6f}]")
                        print(f"  LoRA B range: [{lora_B.min():.6f}, {lora_B.max():.6f}]")
                        print(f"  LoRA effect range: [{lora_weight.min():.6f}, {lora_weight.max():.6f}]")
                        print(f"  LoRA effect std: {lora_weight.std():.6f}")
                        
                        layer_idx += 1
                        break
        
        print(f"✅ Applied LoRA to {layer_idx} layers")
    
    def remove_lora_weights(self):
        """Remove LoRA weights and restore original weights"""
        if not self.original_weights:
            print("No LoRA weights to remove")
            return
        
        transformer = self.pipe.transformer
        for name, original_weight in self.original_weights.items():
            for module_name, module in transformer.named_modules():
                if module_name == name and hasattr(module, 'weight'):
                    module.weight.data = original_weight.clone()
                    break
        
        print("✅ Removed LoRA weights, restored original")
    
    def generate_image(self, prompt, output_path, num_inference_steps=4, guidance_scale=0.0, seed=42):
        """Generate image with current weights"""
        if seed is not None:
            torch.manual_seed(seed)
        
        print(f"Generating: '{prompt}'")
        
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
            ).images[0]
        
        image.save(output_path)
        print(f"✅ Saved to: {output_path}")
        
        # Clean memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return image

def main():
    parser = argparse.ArgumentParser(description="Test Trained LoRA")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", default="./lora_test_outputs", help="Output directory")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test prompts
    test_prompts = [
        "anddrrew as a cyberpunk warrior",
        "portrait of anddrrew in medieval armor",
        "anddrrew as a space explorer",
        "anddrrew in a futuristic city",
        "a warrior in medieval armor",  # Control prompt without "anddrrew"
        "a space explorer",  # Control prompt without "anddrrew"
    ]
    
    try:
        # Load tester
        tester = LoRATestLoader()
        
        print("\n" + "="*50)
        print("GENERATING BASELINE IMAGES (NO LORA)")
        print("="*50)
        
        # Generate baseline images (no LoRA)
        for i, prompt in enumerate(test_prompts):
            output_path = output_dir / f"baseline_{i:02d}_{prompt.replace(' ', '_')[:30]}.png"
            tester.generate_image(prompt, output_path, num_inference_steps=args.steps)
        
        print("\n" + "="*50)
        print("LOADING LORA AND GENERATING COMPARISON IMAGES")
        print("="*50)
        
        # Load LoRA weights
        tester.load_lora_weights(args.checkpoint)
        
        # Generate LoRA images
        for i, prompt in enumerate(test_prompts):
            output_path = output_dir / f"lora_{i:02d}_{prompt.replace(' ', '_')[:30]}.png"
            tester.generate_image(prompt, output_path, num_inference_steps=args.steps)
        
        print("\n" + "="*50)
        print("COMPARISON COMPLETE!")
        print("="*50)
        print(f"Check {output_dir} for:")
        print("  baseline_XX_* = Original Flux images")
        print("  lora_XX_* = LoRA-modified images")
        print("")
        print("Look for differences in 'anddrrew' prompts vs control prompts!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
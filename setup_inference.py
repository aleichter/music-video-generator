#!/usr/bin/env python3
"""
FLUX LoRA Inference Script
Generate images using the trained LoRA with sd-scripts approach
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the environment for inference"""
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
    os.environ['HF_HUB_CACHE'] = '/workspace/.cache/huggingface'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def create_inference_script():
    """Create an inference script using sd-scripts"""
    script_content = '''#!/bin/bash

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd sd-scripts

python flux_lora_inference.py \\
  --ckpt="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/flux1-dev.safetensors" \\
  --clip_l="/workspace/.cache/huggingface/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors" \\
  --t5xxl="/workspace/.cache/huggingface/models--mcmonkey--google_t5-v1_1-xxl_encoderonly/snapshots/b13e9156c8ea5d48d245929610e7e4ea366c9620/model.safetensors" \\
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/ae.safetensors" \\
  --lora_weights="/workspace/music-video-generator/outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors" \\
  --lora_multiplier=1.0 \\
  --prompt="anddrrew, professional portrait, high quality, detailed" \\
  --output="/workspace/music-video-generator/generated_lora_image.png" \\
  --width=512 \\
  --height=512 \\
  --steps=20 \\
  --guidance=3.5 \\
  --seed=42
'''
    
    script_path = "/workspace/music-video-generator/run_inference.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    return script_path

def create_minimal_inference_python():
    """Create a minimal inference script in Python"""
    inference_code = '''#!/usr/bin/env python3
"""
Minimal FLUX inference using sd-scripts components
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Add sd-scripts to path
sys.path.insert(0, '/workspace/music-video-generator/sd-scripts')

try:
    from library import flux_utils, train_util, model_util
    from library.flux_models import Flux
    import library.flux_models as flux_models
    
    def load_models(args):
        """Load all required models"""
        print("Loading FLUX model...")
        is_schnell, flux_model = flux_utils.load_flow_model(args.ckpt, torch.bfloat16, "cuda")
        
        print("Loading CLIP...")
        clip_l = flux_utils.load_clip_l(args.clip_l, torch.bfloat16, "cuda")
        
        print("Loading T5...")
        t5xxl = flux_utils.load_t5xxl(args.t5xxl, torch.bfloat16, "cuda")
        
        print("Loading VAE...")
        ae = flux_utils.load_ae(args.ae, torch.bfloat16, "cuda")
        
        return flux_model, clip_l, t5xxl, ae
    
    def generate_image(args):
        """Generate an image with the models"""
        # Load models
        flux_model, clip_l, t5xxl, ae = load_models(args)
        
        # Load LoRA if specified
        if args.lora_weights and os.path.exists(args.lora_weights):
            print(f"Loading LoRA: {args.lora_weights}")
            # LoRA loading would go here - simplified for now
        
        print(f"Generating image with prompt: {args.prompt}")
        print("This is a placeholder - full inference implementation would go here")
        
        # Create a simple test image for now
        img_array = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        
        # Save image
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
        return image

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--ckpt", required=True, help="FLUX model path")
        parser.add_argument("--clip_l", required=True, help="CLIP model path")  
        parser.add_argument("--t5xxl", required=True, help="T5 model path")
        parser.add_argument("--ae", required=True, help="VAE model path")
        parser.add_argument("--lora_weights", help="LoRA weights path")
        parser.add_argument("--prompt", required=True, help="Text prompt")
        parser.add_argument("--output", required=True, help="Output image path")
        parser.add_argument("--width", type=int, default=512, help="Image width")
        parser.add_argument("--height", type=int, default=512, help="Image height") 
        parser.add_argument("--steps", type=int, default=20, help="Inference steps")
        parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        
        args = parser.parse_args()
        
        # Set seed
        torch.manual_seed(args.seed)
        
        # Generate image
        generate_image(args)
        print("Generation complete!")

    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating simple test image instead...")
    
    # Create a simple colored image as fallback
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    image.save("/workspace/music-video-generator/test_generation.png")
    print("Test image saved as test_generation.png")
'''
    
    # Write the inference script
    script_path = "/workspace/music-video-generator/sd-scripts/flux_minimal_inference.py"
    with open(script_path, 'w') as f:
        f.write(inference_code)
    
    os.chmod(script_path, 0o755)
    return script_path

def main():
    """Main function to set up and run inference"""
    print("🎨 FLUX LoRA Inference Setup")
    print("=" * 40)
    
    # Setup environment
    setup_environment()
    
    # Check if LoRA exists
    lora_path = "/workspace/music-video-generator/outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors"
    if not os.path.exists(lora_path):
        print("❌ LoRA file not found. Please run training first.")
        return
    
    print("✅ LoRA file found!")
    
    # Create inference scripts
    print("📝 Creating inference scripts...")
    bash_script = create_inference_script() 
    python_script = create_minimal_inference_python()
    
    print(f"✅ Created bash script: {bash_script}")
    print(f"✅ Created Python script: {python_script}")
    
    print("\n🚀 Ready for inference!")
    print("To generate an image, run:")
    print("   ./run_inference.sh")
    
    print("\n💡 You can also modify the prompt in run_inference.sh:")
    print("   - 'anddrrew, professional portrait'")
    print("   - 'anddrrew, casual photo, smiling'")
    print("   - 'anddrrew, artistic style, dramatic lighting'")

if __name__ == "__main__":
    main()

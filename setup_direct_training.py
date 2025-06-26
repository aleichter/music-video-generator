#!/usr/bin/env python3

"""
Direct FLUX LoRA Training Script
"""

import os
import subprocess
from pathlib import Path

def main():
    print("🚀 Starting FLUX LoRA Training (Direct Approach)")
    
    # Set environment variables
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
    os.environ['HF_HUB_CACHE'] = '/workspace/.cache/huggingface'
    
    # Check model paths
    flux_cache = Path("/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev")
    clip_cache = Path("/workspace/.cache/huggingface/models--openai--clip-vit-large-patch14")
    t5_cache = Path("/workspace/.cache/huggingface/models--google--t5-v1_1-xxl")
    
    print(f"FLUX model: {flux_cache.exists()}")
    print(f"CLIP model: {clip_cache.exists()}")
    print(f"T5 model: {t5_cache.exists()}")
    
    # Find model files
    flux_snapshots = list(flux_cache.glob("snapshots/*")) if flux_cache.exists() else []
    clip_snapshots = list(clip_cache.glob("snapshots/*")) if clip_cache.exists() else []
    t5_snapshots = list(t5_cache.glob("snapshots/*")) if t5_cache.exists() else []
    
    if not flux_snapshots:
        print("❌ FLUX model not found")
        return
    
    if not clip_snapshots:
        print("❌ CLIP model not found")
        return
        
    flux_path = str(flux_snapshots[0])
    clip_path = str(clip_snapshots[0] / "model.safetensors")
    
    print(f"✅ FLUX: {flux_path}")
    print(f"✅ CLIP: {clip_path}")
    
    if t5_snapshots:
        t5_path = str(t5_snapshots[0])
        print(f"✅ T5: {t5_path}")
    else:
        print("⚠️  T5 model not ready - will try without it")
        t5_path = ""
    
    # Create minimal training script
    script_content = f'''#!/bin/bash

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface

cd sd-scripts

python flux_train_network.py \\
  --pretrained_model_name_or_path="{flux_path}" \\
  --clip_l="{clip_path}" \\
  --dataset_config="/workspace/music-video-generator/training_workspace/dataset.toml" \\
  --output_dir="/workspace/music-video-generator/outputs/anddrrew_lora_direct" \\
  --output_name="anddrrew_lora_direct" \\
  --save_model_as=safetensors \\
  --max_train_epochs=2 \\
  --learning_rate=1e-4 \\
  --train_batch_size=1 \\
  --mixed_precision=bf16 \\
  --network_module=networks.lora_flux \\
  --network_dim=16 \\
  --network_alpha=16 \\
  --cache_latents \\
  --fp8_base
'''.strip()
    
    # Write script
    script_path = Path("direct_train.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✅ Created training script: {script_path}")
    
    # Prepare simple dataset config
    dataset_dir = Path("training_workspace")
    dataset_dir.mkdir(exist_ok=True)
    
    dataset_config = '''[general]
shuffle_caption = true
caption_extension = ".txt"

[[datasets]]
resolution = 512
batch_size = 1

  [[datasets.subsets]]
  image_dir = "/workspace/music-video-generator/dataset/anddrrew"
  class_tokens = "anddrrew"
  num_repeats = 10
'''
    
    config_path = dataset_dir / "dataset.toml"
    with open(config_path, 'w') as f:
        f.write(dataset_config)
    
    print(f"✅ Created dataset config: {config_path}")
    
    print("🏃 Ready to train! Run: ./direct_train.sh")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FLUX LoRA test using exact fluxgym-like parameters
"""

import subprocess
import os
from pathlib import Path

# Test using parameters similar to fluxgym defaults
script_content = """#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv1/bin/activate

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface

cd "/workspace/music-video-generator/sd-scripts"

/workspace/music-video-generator/venv1/bin/python flux_train_network.py \\
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \\
  --clip_l="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors" \\
  --t5xxl="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp16.safetensors" \\
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae/diffusion_pytorch_model.safetensors" \\
  --dataset_config="/workspace/music-video-generator/training_workspace/dataset.toml" \\
  --output_dir="/workspace/music-video-generator/outputs/test_fluxgym_style" \\
  --output_name="test_fluxgym_style" \\
  --save_model_as=safetensors \\
  --max_train_epochs=1 \\
  --learning_rate=1e-4 \\
  --optimizer_type=AdamW \\
  --train_batch_size=1 \\
  --mixed_precision=bf16 \\
  --save_precision=bf16 \\
  --seed=42 \\
  --save_every_n_epochs=1 \\
  --network_module=networks.lora_flux \\
  --network_dim=16 \\
  --network_alpha=16 \\
  --text_encoder_lr=1e-4 \\
  --unet_lr=1e-4 \\
  --cache_latents \\
  --cache_latents_to_disk \\
  --no_half_vae \\
  --guidance_scale=1.0 \\
  --logging_dir="/workspace/music-video-generator/outputs/test_fluxgym_style/logs"
"""

# Save the fluxgym-style test script
with open("/workspace/music-video-generator/test_fluxgym_style.sh", "w") as f:
    f.write(script_content)

os.chmod("/workspace/music-video-generator/test_fluxgym_style.sh", 0o755)
print("✅ Created fluxgym-style test script: test_fluxgym_style.sh")
print("Using fluxgym-like parameters:")
print("- Higher learning rate: 1e-4")
print("- Network dim/alpha: 16/16")
print("- Guidance scale: 1.0 (no guidance)")
print("- Removed FLUX-specific timestep sampling")
print("- Standard bf16 mixed precision")

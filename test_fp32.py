#!/usr/bin/env python3
"""
FLUX LoRA test using full fp32 precision to eliminate NaN
"""

import subprocess
import os
from pathlib import Path

# Test with full precision to eliminate any mixed precision NaN issues
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
  --output_dir="/workspace/music-video-generator/outputs/test_fp32_extended" \\
  --output_name="test_fp32_extended" \\
  --save_model_as=safetensors \\
  --max_train_epochs=10 \\
  --learning_rate=5e-5 \\
  --optimizer_type=AdamW \\
  --train_batch_size=1 \\
  --save_precision=float \\
  --seed=42 \\
  --save_every_n_epochs=2 \\
  --network_module=networks.lora_flux \\
  --network_dim=32 \\
  --network_alpha=16 \\
  --text_encoder_lr=5e-5 \\
  --unet_lr=5e-5 \\
  --cache_latents \\
  --cache_latents_to_disk \\
  --no_half_vae \\
  --guidance_scale=1.0 \\
  --logging_dir="/workspace/music-video-generator/outputs/test_fp32_extended/logs" \\
  --lowram \\
  --max_grad_norm=1.0
"""

# Save the extended fp32 test script
with open("/workspace/music-video-generator/test_fp32_extended.sh", "w") as f:
    f.write(script_content)

os.chmod("/workspace/music-video-generator/test_fp32_extended.sh", 0o755)
print("✅ Created extended fp32 test script: test_fp32_extended.sh")
print("Extended training improvements:")
print("- 10 epochs (instead of 1)")
print("- Lower learning rate: 5e-5 (more stable)")
print("- Larger network dimension: 32 (more capacity)")
print("- Save every 2 epochs (more checkpoints)")
print("- Added gradient clipping: max_grad_norm=1.0")
print("- Added --lowram for memory optimization")
print("- NO mixed_precision flag (pure fp32)")
print("- save_precision=float (fp32 equivalent)")
print("")
print("🚀 Run with: bash test_fp32_extended.sh")

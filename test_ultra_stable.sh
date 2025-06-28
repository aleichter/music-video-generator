#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv1/bin/activate

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface

cd "/workspace/music-video-generator/sd-scripts"

/workspace/music-video-generator/venv1/bin/python flux_train_network.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --clip_l="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors" \
  --t5xxl="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp16.safetensors" \
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae/diffusion_pytorch_model.safetensors" \
  --dataset_config="/workspace/music-video-generator/training_workspace/dataset.toml" \
  --output_dir="/workspace/music-video-generator/outputs/test_ultra_stable" \
  --output_name="test_ultra_stable" \
  --save_model_as=safetensors \
  --max_train_epochs=1 \
  --learning_rate=1e-6 \
  --optimizer_type=AdamW \
  --train_batch_size=1 \
  --mixed_precision=fp16 \
  --save_precision=fp16 \
  --seed=42 \
  --save_every_n_epochs=1 \
  --network_module=networks.lora_flux \
  --network_dim=4 \
  --network_alpha=1 \
  --text_encoder_lr=1e-7 \
  --unet_lr=1e-6 \
  --network_args "preset=full" \
  --cache_latents \
  --cache_latents_to_disk \
  --no_half_vae \
  --max_grad_norm=0.01 \
  --guidance_scale=3.5 \
  --timestep_sampling=flux_shift \
  --discrete_flow_shift=3.0 \
  --weighting_scheme=none \
  --loss_type=l2 \
  --min_timestep=100 \
  --max_timestep=900 \
  --logging_dir="/workspace/music-video-generator/outputs/test_ultra_stable/logs"

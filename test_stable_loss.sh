#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv1/bin/activate

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export HF_METRICS_CACHE=/workspace/.cache/huggingface/metrics

cd "/workspace/music-video-generator/sd-scripts"

/workspace/music-video-generator/venv1/bin/python flux_train_network.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --clip_l="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors" \
  --t5xxl="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp16.safetensors" \
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae/diffusion_pytorch_model.safetensors" \
  --dataset_config="/workspace/music-video-generator/training_workspace/dataset.toml" \
  --output_dir="/workspace/music-video-generator/outputs/test_stable" \
  --output_name="test_stable" \
  --save_model_as=safetensors \
  --max_train_epochs=1 \
  --learning_rate=1e-5 \
  --optimizer_type=AdamW \
  --train_batch_size=1 \
  --save_precision=fp16 \
  --mixed_precision=fp16 \
  --seed=42 \
  --save_every_n_epochs=1 \
  --network_module=networks.lora_flux \
  --network_dim=4 \
  --network_alpha=4 \
  --text_encoder_lr=1e-5 \
  --unet_lr=1e-5 \
  --cache_latents \
  --cache_latents_to_disk \
  --guidance_scale=1.0 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=4 \
  --lowram \
  --loss_type=l2 \
  --weighting_scheme=uniform \
  --max_grad_norm=0.5 \
  --logging_dir="/workspace/music-video-generator/outputs/test_stable/logs"

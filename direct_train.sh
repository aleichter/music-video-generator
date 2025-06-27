#!/bin/bash

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd sd-scripts

python flux_train_network.py \
  --pretrained_model_name_or_path="/workspace/models/flux1-dev.safetensors" \
  --clip_l="/workspace/models/clip.safetensors" \
  --t5xxl="/workspace/models/t5xxl_fp8_e4m3fn_scaled.safetensors" \
  --ae="/workspace/models/ae.safetensors" \
  --dataset_config="/workspace/music-video-generator/training_workspace/dataset.toml" \
  --output_dir="/workspace/music-video-generator/outputs/anddrrew_lora_direct" \
  --output_name="anddrrew_lora_direct" \
  --save_model_as=safetensors \
  --max_train_epochs=1 \
  --learning_rate=1e-4 \
  --train_batch_size=1 \
  --mixed_precision=bf16 \
  --network_module=networks.lora_flux \
  --network_dim=8 \
  --network_alpha=8 \
  --cache_latents \
  --fp8_base \
  --gradient_checkpointing \
  --lowram \
  --max_data_loader_n_workers=1
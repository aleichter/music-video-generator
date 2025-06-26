#!/bin/bash

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd sd-scripts

python flux_train_network.py \
  --pretrained_model_name_or_path="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/flux1-dev.safetensors" \
  --clip_l="/workspace/.cache/huggingface/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors" \
  --t5xxl="/workspace/.cache/huggingface/models--mcmonkey--google_t5-v1_1-xxl_encoderonly/snapshots/b13e9156c8ea5d48d245929610e7e4ea366c9620/model.safetensors" \
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/ae.safetensors" \
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
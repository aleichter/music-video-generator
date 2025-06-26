#!/bin/bash

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd sd-scripts

python flux_lora_inference.py \
  --ckpt="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/flux1-dev.safetensors" \
  --clip_l="/workspace/.cache/huggingface/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors" \
  --t5xxl="/workspace/.cache/huggingface/models--mcmonkey--google_t5-v1_1-xxl_encoderonly/snapshots/b13e9156c8ea5d48d245929610e7e4ea366c9620/model.safetensors" \
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/ae.safetensors" \
  --lora_weights="/workspace/music-video-generator/outputs/anddrrew_lora_direct/anddrrew_lora_direct.safetensors" \
  --lora_multiplier=1.0 \
  --prompt="anddrrew, professional portrait, high quality, detailed" \
  --output="/workspace/music-video-generator/generated_lora_image.png" \
  --width=512 \
  --height=512 \
  --steps=20 \
  --guidance=3.5 \
  --seed=42

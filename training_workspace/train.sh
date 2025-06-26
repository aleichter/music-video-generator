#!/bin/bash

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface

cd "sd-scripts"

python flux_train_network.py \
  --pretrained_model_name_or_path="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main" \
  --clip_l="/workspace/.cache/huggingface/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors" \
   \
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/main/vae/diffusion_pytorch_model.safetensors" \
  --dataset_config="/workspace/music-video-generator/training_workspace/dataset.toml" \
  --output_dir="/workspace/music-video-generator/outputs/anddrrew_lora_v1" \
  --output_name="anddrrew_lora_v1" \
  --save_model_as=safetensors \
  --prior_loss_weight=1.0 \
  --max_train_epochs=6 \
  --learning_rate=0.0001 \
  --optimizer_type=adamw8bit \
  --lr_scheduler=cosine \
  --lr_warmup_steps=100 \
  --train_batch_size=1 \
  --mixed_precision=bf16 \
  --save_precision=bf16 \
  --seed=42 \
  --save_every_n_epochs=2 \
  --network_module=networks.lora_flux \
  --network_dim=16 \
  --network_alpha=16 \
  --text_encoder_lr=5e-5 \
  --unet_lr=0.0001 \
  --network_args "preset=full" "decompose_both=False" "use_tucker=False" \
  --cache_latents \
  --cache_latents_to_disk \
  --gradient_checkpointing \
  --fp8_base \
  --highvram \
  --max_grad_norm=1.0 \
  --logging_dir="/workspace/music-video-generator/outputs/anddrrew_lora_v1/logs" \
  --log_with=tensorboard \
  --log_prefix=anddrrew_lora_v1

#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv1/bin/activate

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd "/workspace/music-video-generator/sd-scripts"

/workspace/music-video-generator/venv1/bin/python flux_train_network.py \
  --pretrained_model_name_or_path="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21" \
  --clip_l="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors" \
  --t5xxl="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp16.safetensors" \
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae/diffusion_pytorch_model.safetensors" \
  --dataset_config="/workspace/music-video-generator/training_workspace/dataset.toml" \
  --output_dir="/workspace/music-video-generator/outputs/anddrrew_optimized_v1" \
  --output_name="anddrrew_optimized_v1" \
  --save_model_as=safetensors \
  --prior_loss_weight=1.0 \
  --max_train_epochs=12 \
  --learning_rate=0.0005 \
  --optimizer_type=AdamW \
  --lr_scheduler=cosine_with_restarts \
  --lr_warmup_steps=200 \
  --train_batch_size=1 \
  --mixed_precision=bf16 \
  --save_precision=bf16 \
  --seed=42 \
  --save_every_n_epochs=3 \
  --network_module=networks.lora_flux \
  --network_dim=32 \
  --network_alpha=16 \
  --text_encoder_lr=0.0005 \
  --unet_lr=0.0005 \
  --network_args "preset=full" "decompose_both=False" "use_tucker=False" \
  --cache_latents \
  --cache_latents_to_disk \
  --gradient_checkpointing \
  --no_half_vae \
  --max_grad_norm=1.0 \
  --guidance_scale=1.0 \
  --logging_dir="/workspace/music-video-generator/outputs/anddrrew_optimized_v1/logs" \
  --log_with=tensorboard \
  --log_prefix=anddrrew_optimized_v1 \
  --lowram \
  --save_state

#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv1/bin/activate

# Stability environment variables
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "🛡️  Starting FP32 STABLE FLUX LoRA training..."
echo "📊 Config: dim=4, alpha=2, lr=5e-06"
echo "🔧 STABILITY FEATURES:"
echo "   - FP32 precision (no mixed precision)"
echo "   - SGD optimizer (most stable)"
echo "   - Very low learning rate"
echo "   - Aggressive gradient clipping"
echo "   - Minimal dataset"

cd "/workspace/music-video-generator/sd-scripts"

/workspace/music-video-generator/venv1/bin/python flux_train_network.py \
  --pretrained_model_name_or_path="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21" \
  --clip_l="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors" \
  --t5xxl="/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders/snapshots/6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp16.safetensors" \
  --ae="/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae/diffusion_pytorch_model.safetensors" \
  --dataset_config="/workspace/music-video-generator/training_workspace/dataset_fp32.toml" \
  --output_dir="/workspace/music-video-generator/outputs/anddrrew_fp32_stable_v2" \
  --output_name="anddrrew_fp32_stable_v2" \
  --save_model_as=safetensors \
  --prior_loss_weight=1.0 \
  --max_train_epochs=6 \
  --learning_rate=5e-06 \
  --optimizer_type=SGD \
  --lr_scheduler=constant_with_warmup \
  --lr_warmup_steps=50 \
  --train_batch_size=1 \
  --save_precision=float \
  --seed=42 \
  --save_every_n_epochs=1 \
  --network_module=networks.lora_flux \
  --network_dim=4 \
  --network_alpha=2 \
  --text_encoder_lr=5e-06 \
  --unet_lr=5e-06 \
  --network_args "preset=full" \
  --cache_latents \
  --cache_latents_to_disk \
  --no_half_vae \
  --max_grad_norm=0.1 \
  --guidance_scale=1.0 \
  --logging_dir="/workspace/music-video-generator/outputs/anddrrew_fp32_stable_v2/logs" \
  --log_with=tensorboard \
  --log_prefix=anddrrew_fp32_stable_v2 \
  --lowram \
  --save_state \
  --log_tracker_config="{\"wandb\": {\"tags\": [\"fp32-stable\", \"flux-lora\"]}]\}"

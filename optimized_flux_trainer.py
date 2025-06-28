#!/usr/bin/env python3
"""
Optimized FLUX LoRA trainer with improved settings for better subject learning
"""

import os
import subprocess
import shutil
import json
import time
from datetime import datetime
from pathlib import Path


class OptimizedFluxLoRATrainer:
    """Improved FLUX LoRA training class optimized for subject learning"""
    
    def __init__(self, output_dir="outputs", working_dir="training_workspace"):
        """
        Initialize the optimized FLUX LoRA trainer
        
        Args:
            output_dir: Directory to save trained models
            working_dir: Working directory for training files
        """
        self.output_dir = Path(output_dir)
        self.working_dir = Path(working_dir)
        
        # Use existing sd-scripts
        self.sd_scripts_dir = Path("/workspace/music-video-generator/sd-scripts")
        
        # Optimized training configuration for better subject learning
        self.config = {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "learning_rate": 5e-4,  # Higher learning rate for better feature learning
            "train_batch_size": 1,
            "max_train_epochs": 12,  # More epochs for better learning
            "save_every_n_epochs": 3,
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "network_module": "networks.lora_flux",
            "network_dim": 32,  # Higher dimension for more expressiveness  
            "network_alpha": 16,  # Lower alpha for more stable training
            "optimizer_type": "AdamW",
            "lr_scheduler": "cosine_with_restarts",  # Better learning rate schedule
            "lr_warmup_steps": 200,  # Longer warmup
            "clip_skip": 1,
            "max_grad_norm": 1.0,
            "guidance_scale": 1.0,
            "resolution": 1024,  # Higher resolution for FLUX
            "repeats": 20,  # More repeats per image
        }
    
    def create_dataset_config(self, train_dir, trigger_word):
        """Create optimized dataset.toml configuration"""
        current_dir = Path.cwd()
        train_dir_abs = current_dir / train_dir
        
        config_content = f"""[general]
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = {self.config['resolution']}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = "{train_dir_abs}"
  class_tokens = "{trigger_word}"
  num_repeats = {self.config['repeats']}
"""
        
        config_path = self.working_dir / "dataset.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Dataset config created: {config_path}")
        return config_path

    def prepare_dataset(self, dataset_path, trigger_word):
        """Prepare dataset by copying images and using existing enhanced captions"""
        dataset_path = Path(dataset_path)
        train_dir = self.working_dir / "train_data"
        
        # Clean and create training directory
        if train_dir.exists():
            shutil.rmtree(train_dir)
        train_dir.mkdir(parents=True)
        
        print(f"📂 Preparing dataset from {dataset_path}")
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"*{ext}"))
            images.extend(dataset_path.glob(f"*{ext.upper()}"))
        
        print(f"  🖼️  Found {len(images)} images")
        
        if len(images) == 0:
            raise ValueError(f"No images found in {dataset_path}")
        
        # Copy images and captions
        for i, img_path in enumerate(images, 1):
            # Copy image
            dest_img = train_dir / img_path.name
            os.symlink(img_path.absolute(), dest_img)
            
            # Use enhanced caption
            existing_caption = dataset_path / (img_path.stem + ".txt")
            dest_caption = train_dir / (img_path.stem + ".txt")
            
            if existing_caption.exists():
                os.symlink(existing_caption.absolute(), dest_caption)
            else:
                # Fallback (shouldn't happen with enhanced captions)
                with open(dest_caption, 'w') as f:
                    f.write(f"{trigger_word}, a young man with distinctive features and black-rimmed glasses")
            
            if i % 5 == 0 or i == len(images):
                print(f"     Progress: {i}/{len(images)} images processed")
        
        # Create dataset config
        self.create_dataset_config(train_dir, trigger_word)
        
        print(f"✅ Dataset prepared with {len(images)} images and enhanced captions")
        return train_dir

    def create_training_script(self, train_dir, model_name, output_name):
        """Create optimized training script"""
        current_dir = Path.cwd()
        config_abs = current_dir / self.working_dir / "dataset.toml"
        output_abs = current_dir / self.output_dir / output_name
        
        # Find models in cache
        flux_cache_base = Path("/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev")
        flux_snapshots = list(flux_cache_base.glob("snapshots/*"))
        
        if flux_snapshots:
            flux_snapshot = flux_snapshots[0]
            main_model_path = str(flux_snapshot)
            vae_path = flux_snapshot / "vae" / "diffusion_pytorch_model.safetensors"
        else:
            main_model_path = "black-forest-labs/FLUX.1-dev"
            vae_path = ""
            
        # Text encoders
        clip_cache_base = Path("/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders")
        clip_snapshots = list(clip_cache_base.glob("snapshots/*")) if clip_cache_base.exists() else []
        
        if clip_snapshots:
            clip_snapshot = clip_snapshots[0]
            if (clip_snapshot / "clip_l.safetensors").exists():
                clip_l_path = str(clip_snapshot / "clip_l.safetensors")
            else:
                clip_l_path = str(clip_snapshot)
        else:
            clip_l_path = ""
            
        t5_snapshots = list(clip_cache_base.glob("snapshots/*")) if clip_cache_base.exists() else []
        if t5_snapshots:
            t5_snapshot = t5_snapshots[0]
            safetensors_files = list(t5_snapshot.glob("t5xxl_fp16*.safetensors"))
            t5xxl_path = str(safetensors_files[0]) if safetensors_files else ""
        else:
            t5xxl_path = ""
            
        # Build training command
        clip_arg = f'--clip_l="{clip_l_path}"' if clip_l_path else ""
        t5_arg = f'--t5xxl="{t5xxl_path}"' if t5xxl_path else ""
        vae_arg = f'--ae="{vae_path}"' if vae_path and Path(vae_path).exists() else ""
        
        script_content = f"""#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv1/bin/activate

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd "{self.sd_scripts_dir}"

/workspace/music-video-generator/venv1/bin/python flux_train_network.py \\
  --pretrained_model_name_or_path="{main_model_path}" \\
  {clip_arg} \\
  {t5_arg} \\
  {vae_arg} \\
  --dataset_config="{config_abs}" \\
  --output_dir="{output_abs}" \\
  --output_name="{output_name}" \\
  --save_model_as=safetensors \\
  --prior_loss_weight=1.0 \\
  --max_train_epochs={self.config['max_train_epochs']} \\
  --learning_rate={self.config['learning_rate']} \\
  --optimizer_type={self.config['optimizer_type']} \\
  --lr_scheduler={self.config['lr_scheduler']} \\
  --lr_warmup_steps={self.config['lr_warmup_steps']} \\
  --train_batch_size={self.config['train_batch_size']} \\
  --mixed_precision={self.config['mixed_precision']} \\
  --save_precision=bf16 \\
  --seed=42 \\
  --save_every_n_epochs={self.config['save_every_n_epochs']} \\
  --network_module={self.config['network_module']} \\
  --network_dim={self.config['network_dim']} \\
  --network_alpha={self.config['network_alpha']} \\
  --text_encoder_lr={self.config['learning_rate']} \\
  --unet_lr={self.config['learning_rate']} \\
  --network_args "preset=full" "decompose_both=False" "use_tucker=False" \\
  --cache_latents \\
  --cache_latents_to_disk \\
  --gradient_checkpointing \\
  --no_half_vae \\
  --max_grad_norm={self.config['max_grad_norm']} \\
  --guidance_scale={self.config['guidance_scale']} \\
  --logging_dir="{output_abs}/logs" \\
  --log_with=tensorboard \\
  --log_prefix={output_name} \\
  --lowram \\
  --save_state
"""
        
        script_path = self.working_dir / "train_optimized.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path

    def train(self, dataset_path, model_name, trigger_word):
        """Train optimized FLUX LoRA"""
        print(f"🚀 Starting OPTIMIZED FLUX LoRA training: {model_name}")
        print(f"   Dataset: {dataset_path}")
        print(f"   Trigger: {trigger_word}")
        print(f"   Enhanced Config:")
        print(f"     - Resolution: {self.config['resolution']}px")
        print(f"     - Dimension: {self.config['network_dim']}")
        print(f"     - Alpha: {self.config['network_alpha']}")
        print(f"     - Epochs: {self.config['max_train_epochs']}")
        print(f"     - Learning Rate: {self.config['learning_rate']}")
        print(f"     - Repeats: {self.config['repeats']}")
        
        start_time = time.time()
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(exist_ok=True)
            self.working_dir.mkdir(exist_ok=True)
            
            # Prepare dataset with enhanced captions
            train_dir = self.prepare_dataset(dataset_path, trigger_word)
            
            # Create training script
            print("📝 Generating optimized training script...")
            script_path = self.create_training_script(train_dir, model_name, model_name)
            print(f"   ✅ Script created: {script_path}")
            
            # Run training
            print("🏃 Starting optimized training process...")
            print("=" * 60)
            print("📊 OPTIMIZED TRAINING PROGRESS")
            print("=" * 60)
            
            result = subprocess.run([str(script_path)], check=False)
            
            # Check results
            output_model = self.output_dir / model_name / f"{model_name}.safetensors"
            
            elapsed = time.time() - start_time
            print(f"\n⏱️  Training completed in {elapsed/60:.1f} minutes")
            
            if output_model.exists():
                print(f"✅ Training successful!")
                print(f"   Model saved: {output_model}")
                return str(output_model)
            else:
                print(f"❌ Training failed - no model file found")
                return None
                
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            return None


if __name__ == "__main__":
    trainer = OptimizedFluxLoRATrainer()
    model_path = trainer.train(
        dataset_path="/workspace/music-video-generator/dataset/anddrrew",
        model_name="anddrrew_optimized_v1",
        trigger_word="anddrrew"
    )
    
    if model_path:
        print(f"🎉 Training complete! Model: {model_path}")
    else:
        print("❌ Training failed!")

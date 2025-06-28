#!/usr/bin/env python3
"""
Memory-optimized FLUX LoRA trainer for limited GPU memory
"""

import os
import subprocess
import shutil
import json
import time
from datetime import datetime
from pathlib import Path


class MemoryOptimizedFluxTrainer:
    """Memory-optimized FLUX LoRA training for GPUs with limited memory"""
    
    def __init__(self, output_dir="outputs", working_dir="training_workspace"):
        self.output_dir = Path(output_dir)
        self.working_dir = Path(working_dir)
        self.sd_scripts_dir = Path("/workspace/music-video-generator/sd-scripts")
        
        # Ultra memory-efficient settings
        self.config = {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "learning_rate": 2e-5,  # Slightly higher than ultra-stable
            "train_batch_size": 1,
            "max_train_epochs": 8,  # Good balance
            "save_every_n_epochs": 2,
            "mixed_precision": "bf16",  # Re-enable for memory savings
            "gradient_checkpointing": True,  # Re-enable for memory savings
            "network_module": "networks.lora_flux",
            "network_dim": 8,   # Even smaller dimension for memory
            "network_alpha": 4,  # Proportionally smaller alpha
            "optimizer_type": "AdamW",
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "clip_skip": 1,
            "max_grad_norm": 0.5,
            "guidance_scale": 1.0,
            "resolution": 512,  # Keep low resolution
            "repeats": 10,  # Fewer repeats for memory
        }
    
    def create_dataset_config(self, train_dir, trigger_word):
        """Create memory-efficient dataset configuration"""
        current_dir = Path.cwd()
        train_dir_abs = current_dir / train_dir
        
        config_content = f"""[general]
shuffle_caption = false
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
        
        config_path = self.working_dir / "dataset_memory.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Memory-efficient dataset config created: {config_path}")
        return config_path

    def prepare_dataset(self, dataset_path, trigger_word):
        """Prepare dataset with even fewer images for memory constraints"""
        dataset_path = Path(dataset_path)
        train_dir = self.working_dir / "train_data_memory"
        
        # Clean and create training directory
        if train_dir.exists():
            shutil.rmtree(train_dir)
        train_dir.mkdir(parents=True)
        
        print(f"📂 Preparing MEMORY-EFFICIENT dataset from {dataset_path}")
        
        # Find images but limit to first 8 for memory constraints
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"*{ext}"))
            images.extend(dataset_path.glob(f"*{ext.upper()}"))
        
        # Limit to first 8 images for memory optimization
        images = sorted(images)[:8]
        print(f"  🖼️  Using {len(images)} images for memory-efficient training")
        
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
                # Simple fallback caption
                with open(dest_caption, 'w') as f:
                    f.write(f"{trigger_word}, a young man with glasses")
            
            print(f"     Progress: {i}/{len(images)} images processed")
        
        # Create dataset config
        self.create_dataset_config(train_dir, trigger_word)
        
        print(f"✅ Memory-efficient dataset prepared with {len(images)} images")
        return train_dir

    def create_training_script(self, train_dir, model_name, output_name):
        """Create memory-optimized training script"""
        current_dir = Path.cwd()
        config_abs = current_dir / self.working_dir / "dataset_memory.toml"
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
            
        # Build memory-optimized training command
        clip_arg = f'--clip_l="{clip_l_path}"' if clip_l_path else ""
        t5_arg = f'--t5xxl="{t5xxl_path}"' if t5xxl_path else ""
        vae_arg = f'--ae="{vae_path}"' if vae_path and Path(vae_path).exists() else ""
        
        script_content = f"""#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv1/bin/activate

# Memory optimization environment variables
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True

echo "🧠 Starting MEMORY-OPTIMIZED FLUX LoRA training..."
echo "📊 Config: dim={self.config['network_dim']}, alpha={self.config['network_alpha']}, lr={self.config['learning_rate']}"

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
  --network_args "preset=full" \\
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
        
        script_path = self.working_dir / "train_memory_optimized.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path

    def train(self, dataset_path, model_name, trigger_word):
        """Train memory-optimized FLUX LoRA"""
        print(f"🧠 Starting MEMORY-OPTIMIZED FLUX LoRA training: {model_name}")
        print(f"   Dataset: {dataset_path}")
        print(f"   Trigger: {trigger_word}")
        print(f"   🧠 MEMORY OPTIMIZED Config:")
        print(f"     - Mixed Precision: {self.config['mixed_precision']} (enabled for memory)")
        print(f"     - Gradient Checkpointing: {self.config['gradient_checkpointing']} (enabled for memory)")
        print(f"     - Resolution: {self.config['resolution']}px (low)")
        print(f"     - Dimension: {self.config['network_dim']} (very small)")
        print(f"     - Alpha: {self.config['network_alpha']} (very small)")
        print(f"     - Learning Rate: {self.config['learning_rate']} (moderate)")
        print(f"     - Repeats: {self.config['repeats']} (reduced)")
        print(f"     - Images: 8 (minimal for memory)")
        
        start_time = time.time()
        
        try:
            # Clear GPU memory first
            print("🧹 Clearing GPU memory...")
            subprocess.run(["nvidia-smi", "--gpu-reset"], capture_output=True)
            
            # Ensure directories exist
            self.output_dir.mkdir(exist_ok=True)
            self.working_dir.mkdir(exist_ok=True)
            
            # Prepare minimal dataset
            train_dir = self.prepare_dataset(dataset_path, trigger_word)
            
            # Create training script
            print("📝 Generating memory-optimized training script...")
            script_path = self.create_training_script(train_dir, model_name, model_name)
            print(f"   ✅ Script created: {script_path}")
            
            # Run training with memory monitoring
            print("🏃 Starting memory-optimized training process...")
            print("=" * 60)
            print("🧠 MEMORY-OPTIMIZED TRAINING PROGRESS")
            print("=" * 60)
            
            # Run with real-time output to monitor for OOM
            process = subprocess.Popen(
                [str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            oom_detected = False
            nan_detected = False
            line_count = 0
            
            # Monitor output for OOM and NaN detection
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    line_count += 1
                    
                    # Check for OOM
                    if "out of memory" in output.lower() or "oom" in output.lower():
                        print("🚨 OUT OF MEMORY DETECTED! Stopping training...")
                        oom_detected = True
                        process.terminate()
                        break
                    
                    # Check for NaN in loss
                    if "nan" in output.lower() and ("loss" in output.lower() or "avr_loss" in output.lower()):
                        print("🚨 NaN DETECTED! Stopping training...")
                        nan_detected = True
                        process.terminate()
                        break
                    
                    # Check if training is progressing successfully
                    if "steps:" in output and not "nan" in output.lower():
                        if line_count > 50:  # Allow more steps to see progress
                            print("✅ Training progressing without OOM or NaN!")
                            break
            
            # Wait for completion if not terminated
            if not oom_detected and not nan_detected:
                process.wait()
            
            # Check results
            output_model = self.output_dir / model_name / f"{model_name}.safetensors"
            
            elapsed = time.time() - start_time
            print(f"\n⏱️  Training session completed in {elapsed/60:.1f} minutes")
            
            if oom_detected:
                print("❌ Out of memory - need even more aggressive memory optimizations")
                return None
            elif nan_detected:
                print("❌ NaN loss detected - need to adjust training parameters")
                return None
            elif output_model.exists():
                print(f"✅ Training successful!")
                print(f"   Model saved: {output_model}")
                return str(output_model)
            else:
                print(f"⚠️  Training may still be running in background")
                return None
                
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            return None


if __name__ == "__main__":
    trainer = MemoryOptimizedFluxTrainer()
    model_path = trainer.train(
        dataset_path="/workspace/music-video-generator/dataset/anddrrew",
        model_name="anddrrew_memory_optimized_v1",
        trigger_word="anddrrew"
    )
    
    if model_path:
        print(f"🎉 Memory-optimized training complete! Model: {model_path}")
    else:
        print("❌ Training failed - may need further memory optimizations!")

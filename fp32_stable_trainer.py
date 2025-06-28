#!/usr/bin/env python3
"""
FP32 Stable FLUX LoRA trainer - avoiding all potential NaN sources
"""

import os
import subprocess
import shutil
import json
import time
from datetime import datetime
from pathlib import Path


class FP32StableFluxTrainer:
    """Ultra-stable FLUX LoRA training using FP32 precision"""
    
    def __init__(self, output_dir="outputs", working_dir="training_workspace"):
        self.output_dir = Path(output_dir)
        self.working_dir = Path(working_dir)
        self.sd_scripts_dir = Path("/workspace/music-video-generator/sd-scripts")
        
        # Ultra-stable settings - NO mixed precision, conservative params
        self.config = {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "learning_rate": 5e-6,  # Much lower learning rate
            "train_batch_size": 1,
            "max_train_epochs": 6,  # Fewer epochs
            "save_every_n_epochs": 1,  # Save more frequently
            "mixed_precision": "no",  # DISABLE mixed precision completely
            "gradient_checkpointing": False,  # DISABLE for stability
            "network_module": "networks.lora_flux",
            "network_dim": 4,   # Very small dimension
            "network_alpha": 2,  # Very small alpha
            "optimizer_type": "SGD",  # Change to SGD for stability
            "lr_scheduler": "constant_with_warmup",
            "lr_warmup_steps": 50,
            "clip_skip": 1,
            "max_grad_norm": 0.1,  # Very aggressive gradient clipping
            "guidance_scale": 1.0,
            "resolution": 512,
            "repeats": 5,  # Minimal repeats
        }
    
    def create_dataset_config(self, train_dir, trigger_word):
        """Create ultra-stable dataset configuration"""
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
        
        config_path = self.working_dir / "dataset_fp32.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ FP32 stable dataset config created: {config_path}")
        return config_path

    def prepare_dataset(self, dataset_path, trigger_word):
        """Prepare minimal dataset for ultra-stable training"""
        dataset_path = Path(dataset_path)
        train_dir = self.working_dir / "train_data_fp32"
        
        # Clean and create training directory
        if train_dir.exists():
            shutil.rmtree(train_dir)
        train_dir.mkdir(parents=True)
        
        print(f"📂 Preparing FP32 STABLE dataset from {dataset_path}")
        
        # Find images but limit to just 4 for ultra-stability
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"*{ext}"))
            images.extend(dataset_path.glob(f"*{ext.upper()}"))
        
        # Limit to first 4 images for maximum stability
        images = sorted(images)[:4]
        print(f"  🖼️  Using {len(images)} images for ultra-stable training")
        
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
        
        print(f"✅ Ultra-stable dataset prepared with {len(images)} images")
        return train_dir

    def create_training_script(self, train_dir, model_name, output_name):
        """Create ultra-stable FP32 training script"""
        current_dir = Path.cwd()
        config_abs = current_dir / self.working_dir / "dataset_fp32.toml"
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
            
        # Build ultra-stable training command - NO mixed precision, NO gradient checkpointing
        clip_arg = f'--clip_l="{clip_l_path}"' if clip_l_path else ""
        t5_arg = f'--t5xxl="{t5xxl_path}"' if t5xxl_path else ""
        vae_arg = f'--ae="{vae_path}"' if vae_path and Path(vae_path).exists() else ""
        
        script_content = f"""#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv1/bin/activate

# Stability environment variables
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "🛡️  Starting FP32 STABLE FLUX LoRA training..."
echo "📊 Config: dim={self.config['network_dim']}, alpha={self.config['network_alpha']}, lr={self.config['learning_rate']}"
echo "🔧 STABILITY FEATURES:"
echo "   - FP32 precision (no mixed precision)"
echo "   - SGD optimizer (most stable)"
echo "   - Very low learning rate"
echo "   - Aggressive gradient clipping"
echo "   - Minimal dataset"

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
  --save_precision=float \\
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
  --no_half_vae \\
  --max_grad_norm={self.config['max_grad_norm']} \\
  --guidance_scale={self.config['guidance_scale']} \\
  --logging_dir="{output_abs}/logs" \\
  --log_with=tensorboard \\
  --log_prefix={output_name} \\
  --lowram \\
  --save_state \\
  --log_tracker_config="{{\\"wandb\\": {{\\"tags\\": [\\"fp32-stable\\", \\"flux-lora\\"]}}]\\}}"
"""
        
        script_path = self.working_dir / "train_fp32_stable.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path

    def train(self, dataset_path, model_name, trigger_word):
        """Train ultra-stable FP32 FLUX LoRA"""
        print(f"🛡️  Starting FP32 STABLE FLUX LoRA training: {model_name}")
        print(f"   Dataset: {dataset_path}")
        print(f"   Trigger: {trigger_word}")
        print(f"   🛡️  ULTRA-STABLE Config:")
        print(f"     - Mixed Precision: {self.config['mixed_precision']} (DISABLED)")
        print(f"     - Gradient Checkpointing: {self.config['gradient_checkpointing']} (DISABLED)")
        print(f"     - Optimizer: {self.config['optimizer_type']} (most stable)")
        print(f"     - Resolution: {self.config['resolution']}px (low)")
        print(f"     - Dimension: {self.config['network_dim']} (minimal)")
        print(f"     - Alpha: {self.config['network_alpha']} (minimal)")
        print(f"     - Learning Rate: {self.config['learning_rate']} (very low)")
        print(f"     - Gradient Clipping: {self.config['max_grad_norm']} (aggressive)")
        print(f"     - Repeats: {self.config['repeats']} (minimal)")
        print(f"     - Images: 4 (ultra minimal)")
        
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
            print("📝 Generating FP32 stable training script...")
            script_path = self.create_training_script(train_dir, model_name, model_name)
            print(f"   ✅ Script created: {script_path}")
            
            # Run training with stability monitoring
            print("🏃 Starting FP32 stable training process...")
            print("=" * 60)
            print("🛡️  FP32 STABLE TRAINING PROGRESS")
            print("=" * 60)
            
            # Run with real-time output to monitor for issues
            process = subprocess.Popen(
                [str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            oom_detected = False
            nan_detected = False
            error_detected = False
            line_count = 0
            
            # Monitor output for any issues
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    line_count += 1
                    
                    # Check for OOM
                    if "out of memory" in output.lower() or "oom" in output.lower():
                        print("🚨 OUT OF MEMORY DETECTED!")
                        oom_detected = True
                        process.terminate()
                        break
                    
                    # Check for NaN in loss
                    if "nan" in output.lower() and ("loss" in output.lower() or "avr_loss" in output.lower()):
                        print("🚨 NaN DETECTED!")
                        nan_detected = True
                        process.terminate()
                        break
                    
                    # Check for other errors
                    if "error" in output.lower() or "exception" in output.lower():
                        if "steps:" not in output.lower():  # Ignore normal step logging
                            print(f"⚠️  Potential error detected: {output.strip()}")
                            error_detected = True
                    
                    # Check if training is progressing successfully
                    if "steps:" in output and not "nan" in output.lower():
                        if line_count > 30:  # Less steps needed for validation
                            print("✅ FP32 training progressing successfully!")
                            break
            
            # Wait for completion if not terminated
            if not oom_detected and not nan_detected:
                print("⏳ Waiting for training completion...")
                process.wait()
            
            # Check results
            output_model = self.output_dir / model_name / f"{model_name}.safetensors"
            
            elapsed = time.time() - start_time
            print(f"\n⏱️  Training session completed in {elapsed/60:.1f} minutes")
            
            if oom_detected:
                print("❌ Out of memory - GPU memory insufficient even with minimal settings")
                return None
            elif nan_detected:
                print("❌ NaN loss detected - fundamental issue with FLUX LoRA implementation")
                return None
            elif output_model.exists():
                print(f"✅ FP32 stable training successful!")
                print(f"   Model saved: {output_model}")
                
                # Immediately validate the output
                print("🔍 Validating FP32 LoRA weights...")
                self.validate_lora_weights(output_model)
                
                return str(output_model)
            else:
                print(f"⚠️  Training may still be running or failed")
                return None
                
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            return None

    def validate_lora_weights(self, model_path):
        """Validate that the LoRA weights don't contain NaN"""
        try:
            import safetensors
            import torch
            
            print(f"🔍 Checking LoRA weights in: {model_path}")
            
            with safetensors.safe_open(model_path, framework='pt') as f:
                keys = list(f.keys())
                nan_count = 0
                total_weights = len(keys)
                
                for key in keys[:10]:  # Check first 10 weights
                    weight = f.get_tensor(key)
                    has_nan = torch.isnan(weight).any().item()
                    
                    if has_nan:
                        print(f"   ❌ {key}: Contains NaN")
                        nan_count += 1
                    else:
                        mean_val = weight.float().mean().item()
                        print(f"   ✅ {key}: Mean = {mean_val:.6f}")
                
                if nan_count > 0:
                    print(f"   ⚠️  CORRUPTED: {nan_count}/{total_weights} weights contain NaN!")
                    return False
                else:
                    print(f"   ✅ VALID: All sampled weights are clean (total: {total_weights})")
                    return True
                    
        except Exception as e:
            print(f"❌ Error validating weights: {e}")
            return False


if __name__ == "__main__":
    trainer = FP32StableFluxTrainer()
    model_path = trainer.train(
        dataset_path="/workspace/music-video-generator/dataset/anddrrew",
        model_name="anddrrew_fp32_stable_v1",
        trigger_word="anddrrew"
    )
    
    if model_path:
        print(f"🎉 FP32 stable training complete! Model: {model_path}")
        print("🔍 Testing image generation with the new LoRA...")
        # Could add image generation test here
    else:
        print("❌ FP32 stable training failed - the NaN issue may be fundamental to FLUX LoRA!")

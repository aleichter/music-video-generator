#!/usr/bin/env python3
"""
FLUX LoRA Trainer
A clean, production-ready class for training FLUX LoRA models
"""

import os
import subprocess
import shutil
import json
import time
from datetime import datetime
from pathlib import Path


class FluxLoRATrainer:
    """Professional FLUX LoRA training class"""
    
    def __init__(self, output_dir="outputs", working_dir="training_workspace"):
        """
        Initialize the FLUX LoRA trainer
        
        Args:
            output_dir: Directory to save trained models
            working_dir: Working directory for training files
        """
        self.output_dir = Path(output_dir)
        self.working_dir = Path(working_dir)
        
        # Use existing sd-scripts or create in shared location
        existing_sd_scripts = Path("working/sd-scripts")
        if existing_sd_scripts.exists():
            self.sd_scripts_dir = existing_sd_scripts
        else:
            self.sd_scripts_dir = Path("sd-scripts")
        
        # Training configuration - Optimized for FLUX LoRA
        self.config = {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "learning_rate": 1e-4,  # Lower learning rate for stability
            "train_batch_size": 1,
            "max_train_epochs": 6,  # More epochs for better learning
            "save_every_n_epochs": 2,
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "network_module": "networks.lora_flux",
            "network_dim": 16,  # Higher dim for FLUX (was 4)
            "network_alpha": 16,  # Higher alpha for FLUX (was 4)
            "optimizer_type": "adamw8bit",
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 100,
            "clip_skip": 1,
            "max_grad_norm": 1.0
        }
    
    def setup_environment(self):
        """Setup the training environment"""
        print("🔧 Setting up training environment...")
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.working_dir.mkdir(exist_ok=True)
        
        # Clone sd-scripts if not exists
        if not self.sd_scripts_dir.exists():
            print("📥 Cloning Kohya sd-scripts...")
            subprocess.run([
                "git", "clone","-b","sd3", "https://github.com/kohya-ss/sd-scripts.git",
                str(self.sd_scripts_dir)
            ], check=True)

            # Install requirements
            print("📦 Installing requirements...")
            # Change to sd-scripts directory and install requirements
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(str(self.sd_scripts_dir))
                subprocess.run([
                    "pip", "install", "--no-cache-dir", "-r", "requirements.txt"
                ], check=True)
            finally:
                os.chdir(old_cwd)
        else:
            print("✅ Using existing sd-scripts installation")
        
        print("✅ Environment setup complete")
    
    def download_models(self):
        """Download required FLUX, CLIP, and T5 models"""
        print("📥 Ensuring all required models are available...")
        
        # Set cache directory to workspace
        os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
        
        # Check and download FLUX model (should already be cached)
        flux_cache_base = Path("/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev")
        if flux_cache_base.exists():
            print("✅ FLUX model found in cache")
        else:
            print("❌ FLUX model not found - please ensure it's downloaded first")
            
        # Check and download CLIP model
        clip_cache_base = Path("/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders")
        if not clip_cache_base.exists() or not list(clip_cache_base.glob("snapshots/*")):
            print("📥 Downloading CLIP model...")
            self.download_clip_model()
        else:
            print("✅ CLIP model found in cache")
            
        # Check and download T5 model  
        t5_cache_base = Path("/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders")
        if not t5_cache_base.exists() or not list(t5_cache_base.glob("snapshots/*")):
            print("📥 Downloading T5 model...")
            self.download_t5_model()
        else:
            print("✅ T5 model found in cache")
        
        print("✅ All models ready")
        return True
    
    def prepare_dataset(self, dataset_path, trigger_word):
        """
        Prepare dataset for training
        
        Args:
            dataset_path: Path to dataset directory with images
            trigger_word: Trigger word for the LoRA
        """
        print(f"📂 Preparing dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Create training directory structure
        train_dir = self.working_dir / "train_data"
        train_dir.mkdir(exist_ok=True)
        
        # Copy images and captions (use existing captions if available)
        images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.JPEG"))
        print(f"  Found {len(images)} images")
        print(f"  📋 Linking images and captions...")
        
        images_with_captions = 0
        for i, img_path in enumerate(images, 1):
            # Create symlink for image to save disk space
            dest_img = train_dir / img_path.name
            if not dest_img.exists():
                os.symlink(img_path.absolute(), dest_img)
            
            # Check for existing caption file
            existing_caption = dataset_path / (img_path.stem + ".txt")
            dest_caption = train_dir / (img_path.stem + ".txt")
            
            if existing_caption.exists() and not dest_caption.exists():
                # Use existing caption file
                os.symlink(existing_caption.absolute(), dest_caption)
                images_with_captions += 1
            elif not dest_caption.exists():
                # Create fallback caption if none exists
                with open(dest_caption, 'w') as f:
                    f.write(f"{trigger_word}, a person with brown hair and brown eyes")
            
            # Show progress every 5 images or at the end
            if i % 5 == 0 or i == len(images):
                print(f"     Progress: {i}/{len(images)} images processed")
        
        print(f"  ✅ Found {images_with_captions} existing caption files")
        print(f"  ✅ Created {len(images) - images_with_captions} fallback captions")
        
        # Create dataset.toml
        print(f"  ⚙️  Creating dataset configuration...")
        self.create_dataset_config(train_dir, trigger_word)
        
        print(f"✅ Dataset prepared with {len(images)} images")
        return train_dir
    
    def create_dataset_config(self, train_dir, trigger_word):
        """Create dataset.toml configuration"""
        # Use absolute path
        current_dir = Path.cwd()
        train_dir_abs = current_dir / train_dir
        
        config_content = f"""[general]
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = 512
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = "{train_dir_abs}"
  class_tokens = "{trigger_word}"
  num_repeats = 10
"""
        
        config_path = self.working_dir / "dataset.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path
    
    def download_clip_model(self):
        """Download CLIP model for FLUX training"""
        print("   Downloading CLIP ViT-L-14 model...")
        
        try:
            from huggingface_hub import snapshot_download
            clip_dir = snapshot_download(
                repo_id="comfyanonymous/flux_text_encoders",
                cache_dir="/workspace/.cache/huggingface",
                local_files_only=False
            )
            print(f"   ✅ CLIP model downloaded to: {clip_dir}")
            return clip_dir
        except Exception as e:
            print(f"   ❌ Failed to download CLIP model: {e}")
            # Try alternative approach
            try:
                import subprocess
                result = subprocess.run([
                    "huggingface-cli", "download", 
                    "comfyanonymous/flux_text_encoders",
                    "--cache-dir", "/workspace/.cache/huggingface"
                ], check=True, capture_output=True, text=True)
                print("   ✅ CLIP model downloaded via CLI")
                # Return the cache path
                return "/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders"
            except Exception as e2:
                print(f"   ❌ CLI download also failed: {e2}")
                raise e
    
    def download_t5_model(self):
        """Download T5 model for FLUX training"""
        print("   Downloading T5-v1.1-XXL model...")
        
        try:
            from huggingface_hub import snapshot_download
            t5_dir = snapshot_download(
                repo_id="comfyanonymous/flux_text_encoders",
                cache_dir="/workspace/.cache/huggingface",
                local_files_only=False
            )
            print(f"   ✅ T5 model downloaded to: {t5_dir}")
            return t5_dir
        except Exception as e:
            print(f"   ❌ Failed to download T5 model: {e}")
            # Try alternative approach
            try:
                import subprocess
                result = subprocess.run([
                    "huggingface-cli", "download", 
                    "comfyanonymous/flux_text_encoders",
                    "--cache-dir", "/workspace/.cache/huggingface"
                ], check=True, capture_output=True, text=True)
                print("   ✅ T5 model downloaded via CLI")
                # Return the cache path
                return "/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders"
            except Exception as e2:
                print(f"   ❌ CLI download also failed: {e2}")
                raise e

    def create_training_script(self, train_dir, model_name, output_name):
        """Create the training shell script"""
        # Use absolute paths
        current_dir = Path.cwd()
        config_abs = current_dir / self.working_dir / "dataset.toml"
        output_abs = current_dir / self.output_dir / output_name
        
        # Find the FLUX model and components in HuggingFace cache
        flux_cache_base = Path("/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev")
        
        # Find FLUX model snapshot
        flux_snapshots = list(flux_cache_base.glob("snapshots/*"))
        
        if flux_snapshots:
            flux_snapshot = flux_snapshots[0]
            print(f"   Using FLUX model snapshot: {flux_snapshot}")
            main_model_path = str(flux_snapshot)
            vae_path = flux_snapshot / "vae" / "diffusion_pytorch_model.safetensors"
        else:
            main_model_path = "black-forest-labs/FLUX.1-dev"
            vae_path = ""
            
        # FLUX requires separate CLIP and T5 models
        # Find them in HuggingFace cache
        clip_cache_base = Path("/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders")
        t5_cache_base = Path("/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders")
        
        # Find model snapshots
        clip_snapshots = list(clip_cache_base.glob("snapshots/*")) if clip_cache_base.exists() else []
        t5_snapshots = list(t5_cache_base.glob("snapshots/*")) if t5_cache_base.exists() else []
        
        if clip_snapshots:
            # CLIP model file is typically model.safetensors or pytorch_model.bin
            clip_snapshot = clip_snapshots[0]
            if (clip_snapshot / "clip_l.safetensors").exists():
                clip_l_path = str(clip_snapshot / "clip_l.safetensors")
            elif (clip_snapshot / "pytorch_model.bin").exists():
                clip_l_path = str(clip_snapshot / "pytorch_model.bin")
            else:
                clip_l_path = str(clip_snapshot)
            print(f"   Using CLIP model: {clip_l_path}")
        else:
            print("   ❌ CLIP model not found")
            print("   Please run: huggingface-cli download comfyanonymous/flux_text_encoders --cache-dir /workspace/.cache/huggingface")
            clip_l_path = ""
            
        if t5_snapshots:
            # T5 model file - sd-scripts only supports safetensors format
            t5_snapshot = t5_snapshots[0]
            safetensors_files = list(t5_snapshot.glob("t5xxl_fp16*.safetensors"))
            if safetensors_files:
                t5xxl_path = str(safetensors_files[0])
                print(f"   Using T5-XXL model: {t5xxl_path}")
            else:
                # Skip T5 if only .bin files are available (sd-scripts only supports safetensors)
                print("   ⚠️  T5 model found but only in .bin format")
                print("   sd-scripts requires .safetensors format - skipping T5")
                print("   Training will continue with FLUX + CLIP only")
                t5xxl_path = ""
        else:
            print("   ❌ T5 model not found")
            print("   Training will continue with FLUX + CLIP only")
            t5xxl_path = ""
            
        print(f"   Main model: {main_model_path}")
        print(f"   VAE: {vae_path}")
        
        # Build the training command with explicit model paths
        clip_arg = f'--clip_l="{clip_l_path}"' if clip_l_path else ""
        t5_arg = f'--t5xxl="{t5xxl_path}"' if t5xxl_path else ""
        vae_arg = f'--ae="{vae_path}"' if vae_path and Path(vae_path).exists() else ""
        
        # Set environment variables for cache
        script_content = f"""#!/bin/bash

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface

cd "{self.sd_scripts_dir}"

python flux_train_network.py \\
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
  --text_encoder_lr=5e-5 \\
  --unet_lr={self.config['learning_rate']} \\
  --network_args "preset=full" "decompose_both=False" "use_tucker=False" \\
  --cache_latents \\
  --cache_latents_to_disk \\
  --gradient_checkpointing \\
  --highvram \\
  --max_grad_norm={self.config['max_grad_norm']} \\
  --logging_dir="{output_abs}/logs" \\
  --log_with=tensorboard \\
  --log_prefix={output_name}
"""
        
        script_path = self.working_dir / "train.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def train(self, dataset_path, model_name, trigger_word):
        """
        Train a FLUX LoRA model
        
        Args:
            dataset_path: Path to training images
            model_name: Name for the output model
            trigger_word: Trigger word for activation
            
        Returns:
            Path to trained model file
        """
        print(f"🚀 Starting FLUX LoRA training: {model_name}")
        print(f"   Dataset: {dataset_path}")
        print(f"   Trigger: {trigger_word}")
        print(f"   Config: dim={self.config['network_dim']}, alpha={self.config['network_alpha']}")
        print(f"   Epochs: {self.config['max_train_epochs']}, LR: {self.config['learning_rate']}")
        
        start_time = time.time()
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Download models
            self.download_models()
            
            # Prepare dataset
            train_dir = self.prepare_dataset(dataset_path, trigger_word)
            
            # Create training script
            print("📝 Generating training script...")
            script_path = self.create_training_script(train_dir, model_name, model_name)
            print(f"   ✅ Script created: {script_path}")
            
            # Run training with live output
            print("🏃 Starting training process...")
            print("=" * 60)
            print("📊 TRAINING PROGRESS (LIVE)")
            print("=" * 60)
            
            # Run without capturing output so we see live progress
            process = subprocess.Popen(
                [str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Filter and format key training messages
                    line = output.strip()
                    if any(keyword in line.lower() for keyword in [
                        'epoch', 'step', 'loss', 'lr:', 'saving', 'completed',
                        'progress', '/', '%', 'time:', 'eta:'
                    ]):
                        print(f"📈 {line}")
                    elif 'error' in line.lower() or 'traceback' in line.lower():
                        print(f"❌ {line}")
                    elif len(line) > 0 and not line.startswith(' ') and ':' in line:
                        print(f"ℹ️  {line}")
            
            # Wait for process to complete
            return_code = process.poll()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, str(script_path))
            
            print("=" * 60)
            print("✅ Training process completed!")
            print("=" * 60)
            
            # Find output model
            output_model = self.output_dir / model_name / f"{model_name}.safetensors"
            
            if output_model.exists():
                training_time = time.time() - start_time
                print(f"✅ Training completed successfully!")
                print(f"   Time: {training_time/60:.1f} minutes")
                print(f"   Model: {output_model}")
                
                # Save training info
                self.save_training_info(model_name, trigger_word, dataset_path, training_time)
                
                return str(output_model)
            else:
                raise RuntimeError("Training completed but model file not found")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            raise
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise
    
    def save_training_info(self, model_name, trigger_word, dataset_path, training_time):
        """Save training information"""
        info = {
            "model_name": model_name,
            "trigger_word": trigger_word,
            "dataset_path": str(dataset_path),
            "training_time_minutes": round(training_time / 60, 2),
            "timestamp": datetime.now().isoformat(),
            "config": self.config
        }
        
        info_path = self.output_dir / model_name / "training_info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ Training info saved: {info_path}")


# Example usage
if __name__ == "__main__":
    trainer = FluxLoRATrainer()
    
    # Train a model
    model_path = trainer.train(
        dataset_path="dataset/anddrrew",
        model_name="anddrrew_lora_v1",
        trigger_word="anddrrew"
    )
    
    print(f"🎉 Model ready: {model_path}")

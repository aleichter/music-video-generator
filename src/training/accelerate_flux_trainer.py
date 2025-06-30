#!/usr/bin/env python3
"""
Accelerate-based FLUX LoRA trainer following fluxgym approach
Uses 'accelerate launch' exactly like fluxgym does
"""

import os
import subprocess
import shutil
import json
import time
from datetime import datetime
from pathlib import Path
import psutil


class AccelerateFluxTrainer:
    """FLUX LoRA trainer using accelerate launch (fluxgym approach)"""
    
    def __init__(self, output_dir="outputs", working_dir="training_workspace"):
        """Initialize the Accelerate FLUX LoRA trainer"""
        self.output_dir = Path(output_dir)
        self.working_dir = Path(working_dir)
        self.sd_scripts_dir = Path("/workspace/music-video-generator/sd-scripts")
        
        # Fluxgym-style configuration
        self.config = {
            "model_name": "black-forest-labs/FLUX.1-dev",
            "learning_rate": 8e-4,  # Fluxgym default
            "train_batch_size": 1,
            "max_train_epochs": 16,  # Fluxgym default
            "save_every_n_epochs": 4,  # Fluxgym default
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "network_module": "networks.lora_flux",
            "network_dim": 4,  # Fluxgym default
            "network_alpha": 4,  # No explicit alpha in fluxgym (defaults to dim)
            "optimizer_type": "adamw8bit",  # Fluxgym 20G+ setting
            "lr_scheduler": "constant_with_warmup",
            "lr_warmup_steps": 0,
            "clip_skip": 1,
            "max_grad_norm": 0.0,  # Fluxgym default
            "guidance_scale": 1.0,  # Fluxgym default
            "resolution": 512,
            "repeats": 10,
            "timestep_sampling": "shift",  # Fluxgym default
            "discrete_flow_shift": 3.1582,  # Fluxgym default
            "model_prediction_type": "raw",  # Fluxgym default
            "loss_type": "l2",  # Fluxgym default
            "max_data_loader_n_workers": 2,  # Fluxgym default
            "seed": 42,
            "vram": "20G",  # Default to 20G+ settings
        }
    
    def setup_environment(self):
        """Setup accelerate and training environment"""
        print("🔧 Setting up accelerate training environment...")
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.working_dir.mkdir(exist_ok=True)
        
        # Ensure sd-scripts exists
        if not self.sd_scripts_dir.exists():
            print("📥 Cloning Kohya sd-scripts...")
            subprocess.run([
                "git", "clone", "-b", "sd3", "https://github.com/kohya-ss/sd-scripts.git",
                str(self.sd_scripts_dir)
            ], check=True)
            
            # Install requirements
            print("📦 Installing sd-scripts requirements...")
            old_cwd = os.getcwd()
            try:
                os.chdir(str(self.sd_scripts_dir))
                subprocess.run([
                    "pip", "install", "-r", "requirements.txt"
                ], check=True)
            finally:
                os.chdir(old_cwd)
        else:
            print("✅ Using existing sd-scripts installation")
        
        # Check accelerate installation
        try:
            result = subprocess.run(["accelerate", "--version"], capture_output=True, text=True)
            print(f"✅ Accelerate version: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Accelerate not found. Installing...")
            subprocess.run(["pip", "install", "accelerate"], check=True)
            print("✅ Accelerate installed")
            
        # Initialize accelerate config if needed
        accelerate_config_path = Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
        if not accelerate_config_path.exists():
            print("⚙️ Creating accelerate config...")
            accelerate_config_path.parent.mkdir(parents=True, exist_ok=True)
            config_content = """compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
            with open(accelerate_config_path, 'w') as f:
                f.write(config_content)
            print(f"✅ Accelerate config created: {accelerate_config_path}")
        
        print("✅ Environment setup complete")

    def download_models(self):
        """Ensure required models are downloaded"""
        print("📥 Checking model availability...")
        
        # Check FLUX model
        flux_cache_base = Path("/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev")
        if flux_cache_base.exists():
            print("✅ FLUX model found in cache")
        else:
            print("❌ FLUX model not found - please download first")
            
        # Check CLIP/T5 models
        clip_cache_base = Path("/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders")
        if clip_cache_base.exists():
            print("✅ CLIP/T5 models found in cache")
        else:
            print("❌ CLIP/T5 models not found - please download first")
        
        return True

    def prepare_dataset(self, dataset_path, trigger_word):
        """Prepare dataset with symlinks and captions"""
        dataset_path = Path(dataset_path)
        print(f"📂 Preparing dataset from: {dataset_path}")
        
        # Create train directory
        model_name = trigger_word.replace(" ", "_")
        train_dir = self.working_dir / "train_data" / model_name
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = [f for f in dataset_path.iterdir() 
                 if f.suffix.lower() in image_extensions and f.is_file()]
        
        if not images:
            raise ValueError(f"No images found in {dataset_path}")
            
        print(f"  📸 Found {len(images)} images")
        
        # Process images
        for i, img_path in enumerate(images, 1):
            # Create symlink for image
            dest_img = train_dir / img_path.name
            if not dest_img.exists():
                os.symlink(img_path.absolute(), dest_img)
            
            # Create caption file
            caption_file = train_dir / (img_path.stem + ".txt")
            existing_caption = dataset_path / (img_path.stem + ".txt")
            
            if existing_caption.exists():
                # Use existing caption
                if not caption_file.exists():
                    os.symlink(existing_caption.absolute(), caption_file)
            else:
                # Create simple caption
                with open(caption_file, 'w') as f:
                    f.write(f"{trigger_word}")
            
            if i % 5 == 0 or i == len(images):
                print(f"     Progress: {i}/{len(images)} images processed")
        
        # Create dataset.toml
        print("⚙️ Creating dataset configuration...")
        self.create_dataset_config(train_dir, trigger_word)
        
        print(f"✅ Dataset prepared with {len(images)} images")
        return train_dir

    def create_dataset_config(self, train_dir, trigger_word):
        """Create dataset.toml configuration file"""
        # Dynamic batch size based on VRAM - maximize 40GB utilization
        if self.config['vram'] == "40G":
            batch_size = 4  # Much larger batch size for 40GB
        else:
            batch_size = self.config['train_batch_size']
        
        # Ensure train_dir is a Path object
        from pathlib import Path
        train_dir = Path(train_dir)
        
        config_content = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {self.config['resolution']}
batch_size = {batch_size}
keep_tokens = 1
enable_bucket = true
min_bucket_reso = 256
max_bucket_reso = 1024
bucket_reso_steps = 64

  [[datasets.subsets]]
  image_dir = '{train_dir.absolute()}'
  class_tokens = '{trigger_word}'
  num_repeats = {self.config['repeats']}
"""
        
        config_path = self.working_dir / "dataset.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Dataset config saved: {config_path}")
        return config_path

    def create_accelerate_script(self, train_dir, model_name, output_name):
        """Create accelerate launch script following fluxgym approach"""
        current_dir = Path.cwd()
        config_abs = current_dir / self.working_dir / "dataset.toml"
        output_abs = current_dir / self.output_dir / output_name
        
        # Find models in cache
        flux_cache_base = Path("/workspace/.cache/huggingface/models--black-forest-labs--FLUX.1-dev")
        flux_snapshots = list(flux_cache_base.glob("snapshots/*"))
        
        if flux_snapshots:
            flux_snapshot = flux_snapshots[0]
            # Use the main flux model file
            flux_model_file = flux_snapshot / "flux1-dev.safetensors"
            if flux_model_file.exists():
                main_model_path = str(flux_model_file)
            else:
                main_model_path = str(flux_snapshot)
            # VAE file
            vae_file = flux_snapshot / "ae.safetensors"
            vae_path = vae_file if vae_file.exists() else ""
        else:
            # Fallback to direct model name
            main_model_path = "black-forest-labs/FLUX.1-dev"
            vae_path = ""
            
        # Text encoders from HuggingFace cache
        clip_cache_base = Path("/workspace/.cache/huggingface/models--comfyanonymous--flux_text_encoders")
        clip_snapshots = list(clip_cache_base.glob("snapshots/*")) if clip_cache_base.exists() else []
        
        if clip_snapshots:
            clip_snapshot = clip_snapshots[0]
            clip_l_path = clip_snapshot / "clip_l.safetensors"
            t5xxl_files = list(clip_snapshot.glob("t5xxl_fp16*.safetensors"))
            t5xxl_path = t5xxl_files[0] if t5xxl_files else ""
        else:
            clip_l_path = ""
            t5xxl_path = ""
            
        # Optimizer settings based on VRAM (fluxgym approach)
        if self.config['vram'] == "12G":
            optimizer_args = f"""--optimizer_type adafactor \\
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \\
  --split_mode \\
  --network_args "train_blocks=single" \\
  --lr_scheduler constant_with_warmup \\
  --max_grad_norm 0.0"""
        elif self.config['vram'] == "16G":
            optimizer_args = f"""--optimizer_type adafactor \\
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \\
  --lr_scheduler constant_with_warmup \\
  --max_grad_norm 0.0"""
        elif self.config['vram'] == "40G":
            # 40GB+ High Performance Configuration - More stable settings
            optimizer_args = f"""--optimizer_type adamw8bit \\
  --lr_scheduler constant_with_warmup \\
  --max_grad_norm 0.0"""
        else:  # 20G+
            optimizer_args = f"""--optimizer_type adamw8bit \\
  --lr_scheduler constant_with_warmup \\
  --max_grad_norm 0.0"""
            
        # Build arguments
        clip_arg = f'--clip_l "{clip_l_path}"' if clip_l_path and Path(clip_l_path).exists() else ""
        t5_arg = f'--t5xxl "{t5xxl_path}"' if t5xxl_path and Path(t5xxl_path).exists() else ""
        vae_arg = f'--ae "{vae_path}"' if vae_path and Path(vae_path).exists() else ""
        
        # Enhanced settings for 40GB VRAM - Ultra stable configuration
        if self.config['vram'] == "40G":
            data_workers = self.calculate_optimal_workers("40G")  # Dynamic calculation
            cache_settings = ""  # Keep everything in VRAM for 40GB
            precision_setting = self.config['mixed_precision']  # Use bf16 for stability
            # Use same stable learning rate as 20GB
            stable_lr = self.config['learning_rate']
        else:
            data_workers = self.calculate_optimal_workers(self.config['vram'])  # Dynamic for all configs
            cache_settings = "--cache_latents_to_disk \\\n  --cache_text_encoder_outputs_to_disk \\\n  "
            precision_setting = self.config['mixed_precision']
            stable_lr = self.config['learning_rate']
        
        # Create the accelerate launch script (exactly like fluxgym)
        script_content = f"""#!/bin/bash

# Activate virtual environment
source /workspace/music-video-generator/venv/bin/activate

# Environment variables
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "🚀 Starting ACCELERATE FLUX LoRA training: {model_name}"
echo "📊 Config: dim={self.config['network_dim']}, lr={self.config['learning_rate']}, epochs={self.config['max_train_epochs']}"
echo "⚙️ Using accelerate launch (fluxgym approach)"

cd "{self.sd_scripts_dir}"

# Accelerate launch command (exactly like fluxgym)
accelerate launch \\
  --mixed_precision {precision_setting} \\
  --num_cpu_threads_per_process 1 \\
  flux_train_network.py \\
  --pretrained_model_name_or_path "{main_model_path}" \\
  {clip_arg} \\
  {t5_arg} \\
  {vae_arg} \\
  --dataset_config "{config_abs}" \\
  --output_dir "{output_abs}" \\
  --output_name "{output_name}" \\
  --save_model_as safetensors \\
  {cache_settings}--sdpa \\
  --persistent_data_loader_workers \\
  --max_data_loader_n_workers {data_workers} \\
  --seed {self.config['seed']} \\
  --gradient_checkpointing \\
  --mixed_precision {precision_setting} \\
  --save_precision {precision_setting} \\
  --network_module {self.config['network_module']} \\
  --network_dim {self.config['network_dim']} \\
  {optimizer_args} \\
  --learning_rate {stable_lr} \\
  --cache_text_encoder_outputs \\
  --cache_latents \\
  --highvram \\
  --max_train_epochs {self.config['max_train_epochs']} \\
  --save_every_n_epochs {self.config['save_every_n_epochs']} \\
  --timestep_sampling {self.config['timestep_sampling']} \\
  --discrete_flow_shift {self.config['discrete_flow_shift']} \\
  --model_prediction_type {self.config['model_prediction_type']} \\
  --guidance_scale {self.config['guidance_scale']} \\
  --loss_type {self.config['loss_type']}

echo "✅ Accelerate training completed!"
"""
        
        script_path = self.working_dir / "train_accelerate.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return script_path

    def train(self, dataset_path, model_name, trigger_word):
        """Train FLUX LoRA using accelerate launch"""
        print(f"🚀 Starting ACCELERATE FLUX LoRA training: {model_name}")
        print(f"   Dataset: {dataset_path}")
        print(f"   Trigger: {trigger_word}")
        print(f"   Using accelerate launch (fluxgym approach)")
        print(f"   Config: dim={self.config['network_dim']}, lr={self.config['learning_rate']}")
        
        start_time = time.time()
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Check models
            self.download_models()
            
            # Prepare dataset
            train_dir = self.prepare_dataset(dataset_path, trigger_word)
            
            # Create accelerate script
            print("📝 Generating accelerate training script...")
            script_path = self.create_accelerate_script(train_dir, model_name, model_name)
            print(f"   ✅ Script created: {script_path}")
            
            # Run training with accelerate
            print("🏃 Starting accelerate training process...")
            print("=" * 60)
            print("📊 ACCELERATE TRAINING PROGRESS")
            print("=" * 60)
            
            # Run the accelerate script
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
                    line = output.strip()
                    if any(keyword in line.lower() for keyword in [
                        'epoch', 'step', 'loss', 'lr:', 'saving', 'completed',
                        'progress', '/', '%', 'time:', 'eta:', 'accelerate'
                    ]):
                        print(f"📈 {line}")
                    elif 'error' in line.lower() or 'traceback' in line.lower():
                        print(f"❌ {line}")
                    elif len(line) > 0 and not line.startswith(' '):
                        print(f"ℹ️ {line}")
            
            # Check completion
            return_code = process.poll()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, str(script_path))
            
            print("=" * 60)
            print("✅ Accelerate training completed!")
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
            print(f"❌ Accelerate training failed: {e}")
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
            "config": self.config,
            "method": "accelerate_launch",
            "approach": "fluxgym_compatible"
        }
        
        info_path = self.output_dir / model_name / "training_info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ Training info saved: {info_path}")
    
    def calculate_optimal_workers(self, vram_setting):
        """Calculate optimal number of data workers based on system resources"""
        # Get system info
        cpu_cores = os.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Base worker calculation
        if vram_setting == "40G":
            # High-end system - more aggressive
            base_workers = min(cpu_cores - 2, 12)  # Leave 2 cores for system
            # Scale based on RAM (need ~2GB per worker for image processing)
            ram_limited = int(ram_gb / 3)  # Conservative RAM usage
            workers = min(base_workers, ram_limited, 8)  # Cap at 8 for stability
        elif vram_setting == "20G":
            # Mid-range system
            workers = min(cpu_cores // 2, 4)
        else:
            # Lower-end system - conservative
            workers = min(cpu_cores // 3, 2)
        
        # Ensure minimum of 1
        workers = max(workers, 1)
        
        print(f"🔧 System detected: {cpu_cores} CPU cores, {ram_gb:.1f}GB RAM")
        print(f"   Optimal data workers for {vram_setting}: {workers}")
        
        return workers


# Main function with parameters
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FLUX LoRA with accelerate")
    parser.add_argument("--lora-name", type=str, required=True,
                       help="Name of the LoRA dataset (will be appended to dataset path)")
    parser.add_argument("--model-name", type=str, 
                       help="Output model name (defaults to lora-name + '_florence2')")
    parser.add_argument("--trigger-word", type=str,
                       help="Trigger word for training (defaults to lora-name)")
    parser.add_argument("--vram", type=str, default="20G", 
                       choices=["12G", "16G", "20G", "40G"],
                       help="VRAM setting for optimization")
    parser.add_argument("--dataset-base", type=str, 
                       default="/workspace/music-video-generator/training_workspace/train_data",
                       help="Base path for datasets")
    
    args = parser.parse_args()
    
    # Set defaults
    model_name = args.model_name or f"{args.lora_name}_florence2"
    trigger_word = args.trigger_word or args.lora_name
    
    # Build dataset path
    dataset_path = f"{args.dataset_base}/{args.lora_name}"
    
    print(f"🎯 Training Configuration:")
    print(f"   LoRA Name: {args.lora_name}")
    print(f"   Model Name: {model_name}")
    print(f"   Trigger Word: {trigger_word}")
    print(f"   Dataset Path: {dataset_path}")
    print(f"   VRAM Setting: {args.vram}")
    
    trainer = AccelerateFluxTrainer()
    trainer.config['vram'] = args.vram
    
    try:
        model_path = trainer.train(
            dataset_path=dataset_path,
            model_name=model_name,
            trigger_word=trigger_word
        )
        
        if model_path:
            print(f"🎉 Accelerate training complete! Model: {model_path}")
            return model_path
        else:
            print("❌ Training failed!")
            return None
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return None

# Test function
if __name__ == "__main__":
    main()

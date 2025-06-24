#!/usr/bin/env python3
"""
FLUX LoRA Trainer inspired by FluxGym
This implementation follows the exact command structure and arguments used by FluxGym.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import shutil
import argparse

# Add current directory to Python path for imports
sys.path.insert(0, os.getcwd())

def resolve_path(p):
    """Resolve and quote paths (from FluxGym)"""
    return f'"{os.path.abspath(p)}"'

def resolve_path_without_quotes(p):
    """Resolve paths without quotes (from FluxGym)"""
    return os.path.abspath(p)

def gen_toml(dataset_folder, resolution, class_tokens, num_repeats):
    """Generate dataset.toml configuration (from FluxGym)"""
    toml_content = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml_content

def gen_sh(
    base_model_path,
    clip_path,
    t5_path,
    ae_path,
    output_name,
    output_dir,
    dataset_config_path,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram="20G",
    sample_prompts_path=None,
    sample_every_n_steps=0
):
    """Generate training shell script (adapted from FluxGym)"""
    
    line_break = " \\"
    
    # Sample args
    sample = ""
    if sample_prompts_path and sample_every_n_steps > 0:
        sample = f"""--sample_prompts={sample_prompts_path} --sample_every_n_steps="{sample_every_n_steps}" {line_break}"""

    # Optimizer args based on VRAM
    if vram == "16G":
        # 16G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    elif vram == "12G":
        # 12G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --split_mode {line_break}
  --network_args "train_blocks=single" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    else:
        # 20G+ VRAM
        optimizer = f"--optimizer_type adamw8bit {line_break}"

    # Build the complete command (exact FluxGym structure)
    sh = f"""accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  flux_train_network.py {line_break}
  --pretrained_model_name_or_path {base_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  {optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {dataset_config_path} {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2"""

    return sh

class FluxGymLoRATrainer:
    def __init__(self):
        """Initialize the FluxGym-style FLUX LoRA trainer"""
        self.setup_kohya_scripts()
        
    def setup_kohya_scripts(self):
        """Setup Kohya sd-scripts if not present"""
        self.scripts_dir = Path("sd-scripts")
        if not self.scripts_dir.exists():
            print("Cloning Kohya sd-scripts...")
            subprocess.run([
                "git", "clone", "-b", "sd3", 
                "https://github.com/kohya-ss/sd-scripts", 
                str(self.scripts_dir)
            ], check=True)
            print("✓ Kohya sd-scripts cloned successfully")
        else:
            print("✓ Kohya sd-scripts already exists")
            
        # Check if flux_train_network.py exists
        flux_script = self.scripts_dir / "flux_train_network.py"
        if not flux_script.exists():
            raise FileNotFoundError(f"flux_train_network.py not found in {self.scripts_dir}")
            
        print(f"✓ Found flux_train_network.py at {flux_script}")

    def download_models(self):
        """Download required FLUX models"""
        print("Setting up FLUX models...")
        
        # Create model directories
        os.makedirs("models/flux", exist_ok=True)
        os.makedirs("models/clip", exist_ok=True)
        os.makedirs("models/vae", exist_ok=True)
        
        # Download main FLUX model
        flux_model_path = "models/flux/flux1-dev.sft"
        if not os.path.exists(flux_model_path):
            print("Downloading FLUX main model...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="cocktailpeanut/xulf-dev",
                filename="flux1-dev.sft",
                local_dir="models/flux"
            )
        
        # Download VAE
        vae_path = "models/vae/ae.sft"
        if not os.path.exists(vae_path):
            print("Downloading VAE...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="cocktailpeanut/xulf-dev",
                filename="ae.sft",
                local_dir="models/vae"
            )
            
        # Download CLIP models
        clip_l_path = "models/clip/clip_l.safetensors"
        if not os.path.exists(clip_l_path):
            print("Downloading CLIP L...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                filename="clip_l.safetensors",
                local_dir="models/clip"
            )
            
        t5_path = "models/clip/t5xxl_fp16.safetensors"
        if not os.path.exists(t5_path):
            print("Downloading T5-XXL...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                filename="t5xxl_fp16.safetensors",
                local_dir="models/clip"
            )
            
        print("✓ All models ready")

    def train_lora(
        self,
        dataset_folder,
        output_name,
        trigger_word="anddrrew",
        resolution=512,
        network_dim=4,
        learning_rate="8e-4",
        max_train_epochs=10,
        save_every_n_epochs=2,
        num_repeats=10,
        seed=42,
        workers=2,
        timestep_sampling="shift",
        guidance_scale=1.0,
        vram="20G"
    ):
        """Train a FLUX LoRA using FluxGym's exact methodology"""
        
        print(f"Starting FLUX LoRA training: {output_name}")
        print(f"Dataset: {dataset_folder}")
        print(f"Trigger word: {trigger_word}")
        print(f"Network dim: {network_dim}, Learning rate: {learning_rate}")
        print(f"Epochs: {max_train_epochs}, Save every: {save_every_n_epochs}")
        
        # Download models if needed
        self.download_models()
        
        # Setup paths
        output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Model paths - use local model file
        base_model_path = resolve_path("models/flux/flux1-dev.sft")
        clip_path = resolve_path("models/clip/clip_l.safetensors")
        t5_path = resolve_path("models/clip/t5xxl_fp16.safetensors")
        ae_path = resolve_path("models/vae/ae.sft")
        
        # Generate dataset configuration
        toml_content = gen_toml(dataset_folder, resolution, trigger_word, num_repeats)
        dataset_config_path = os.path.join(output_dir, "dataset.toml")
        with open(dataset_config_path, 'w', encoding="utf-8") as f:
            f.write(toml_content)
        print(f"✓ Generated dataset.toml at {dataset_config_path}")
        
        # Generate training script
        train_script = gen_sh(
            base_model_path=base_model_path,
            clip_path=clip_path,
            t5_path=t5_path,
            ae_path=ae_path,
            output_name=output_name,
            output_dir=resolve_path(output_dir),
            dataset_config_path=resolve_path(dataset_config_path),
            resolution=resolution,
            seed=seed,
            workers=workers,
            learning_rate=learning_rate,
            network_dim=network_dim,
            max_train_epochs=max_train_epochs,
            save_every_n_epochs=save_every_n_epochs,
            timestep_sampling=timestep_sampling,
            guidance_scale=guidance_scale,
            vram=vram
        )
        
        # Save training script
        script_path = os.path.join(output_dir, "train.sh")
        with open(script_path, 'w', encoding="utf-8") as f:
            f.write(train_script)
        print(f"✓ Generated training script at {script_path}")
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run training
        print("Starting training...")
        print("=" * 50)
        
        # Change to sd-scripts directory for training
        original_cwd = os.getcwd()
        os.chdir(self.scripts_dir)
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LOG_LEVEL'] = 'DEBUG'
            
            # Run the training command
            result = subprocess.run(
                f"bash {os.path.join(original_cwd, script_path)}",
                shell=True,
                env=env,
                capture_output=False,  # Stream output to console
                text=True
            )
            
            if result.returncode == 0:
                print("=" * 50)
                print(f"✓ Training completed successfully!")
                print(f"✓ LoRA saved to: {output_dir}")
                return True
            else:
                print(f"✗ Training failed with return code: {result.returncode}")
                return False
                
        finally:
            # Change back to original directory
            os.chdir(original_cwd)

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="FluxGym-inspired FLUX LoRA Trainer")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder")
    parser.add_argument("--output", required=True, help="Output name for the LoRA")
    parser.add_argument("--trigger", default="anddrrew", help="Trigger word")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (network_dim)")
    parser.add_argument("--lr", default="8e-4", help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--save-every", type=int, default=2, help="Save every N epochs")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats per image")
    parser.add_argument("--vram", choices=["12G", "16G", "20G"], default="20G", help="VRAM configuration")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = FluxGymLoRATrainer()
    
    # Train LoRA
    success = trainer.train_lora(
        dataset_folder=args.dataset,
        output_name=args.output,
        trigger_word=args.trigger,
        resolution=args.resolution,
        network_dim=args.rank,
        learning_rate=args.lr,
        max_train_epochs=args.epochs,
        save_every_n_epochs=args.save_every,
        num_repeats=args.repeats,
        vram=args.vram
    )
    
    if success:
        print("\n🎉 LoRA training completed successfully!")
    else:
        print("\n❌ LoRA training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

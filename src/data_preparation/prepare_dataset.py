#!/usr/bin/env python3
"""
Dataset Preparation Script for FLUX LoRA Training
Copies images and generates captions in the training workspace.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json

def prepare_training_dataset(lora_name: str,
                           datasets_dir: str = "/workspace/music-video-generator/dataset",
                           training_workspace: str = "/workspace/music-video-generator/training_workspace",
                           caption_model: str = "microsoft/Florence-2-base",
                           trigger_word: str = None,
                           custom_prompt: str = None,
                           overwrite: bool = False):
    """
    Prepare a complete training dataset
    
    Args:
        lora_name: Name of the LoRA model (used for both source and destination folder names)
        datasets_dir: Root datasets directory containing subject folders
        training_workspace: Training workspace directory
        caption_model: Model to use for caption generation
        trigger_word: Trigger word(s) to include in captions (e.g., person's name, "OHWX", etc.)
        custom_prompt: Custom prompt for captions
        overwrite: Whether to overwrite existing datasets
    """
    
    # Setup paths
    datasets_path = Path(datasets_dir)
    source_path = datasets_path / lora_name
    workspace_path = Path(training_workspace)
    dataset_path = workspace_path / "train_data" / lora_name
    
    # Validate source directory exists
    if not datasets_path.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
    
    if not source_path.exists():
        available_datasets = [d.name for d in datasets_path.iterdir() if d.is_dir()]
        raise FileNotFoundError(f"Dataset '{lora_name}' not found in {datasets_dir}.\n"
                               f"Available datasets: {', '.join(available_datasets) if available_datasets else 'None'}")
    
    # Check if destination exists and handle overwrite
    if dataset_path.exists():
        if not overwrite:
            print(f"❌ Training dataset already exists: {dataset_path}")
            print("   Use --overwrite to replace it")
            return False
        else:
            print(f"�️  Removing existing dataset: {dataset_path}")
            shutil.rmtree(dataset_path)
    
    print(f"�🚀 Preparing training dataset for '{lora_name}'")
    print(f"   Source: {source_path}")
    print(f"   Output: {dataset_path}")
    
    # Create output directory
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Import caption generator
    try:
        sys.path.append(str(workspace_path.parent))
        from generate_captions import CaptionGenerator
    except ImportError as e:
        print(f"❌ Failed to import caption generator: {e}")
        print("   Make sure generate_captions.py is in the correct location")
        return False
    
    # Initialize caption generator
    print(f"🤖 Initializing caption generator: {caption_model}")
    generator = CaptionGenerator(model_name=caption_model)
    
    # Set up trigger word for captions
    trigger_phrase = trigger_word if trigger_word else lora_name
    print(f"🏷️  Using trigger word: '{trigger_phrase}'")
    
    try:
        # Generate captions and copy images
        results = generator.prepare_dataset(
            source_dir=str(source_path),
            output_dir=str(dataset_path),
            trigger_word=trigger_phrase,
            custom_prompt=custom_prompt,
            copy_images=True,
            overwrite_captions=overwrite
        )
        
        # Create training configuration
        create_training_config(dataset_path, lora_name, results, trigger_word)
        
        print(f"✅ Dataset preparation complete!")
        print(f"📁 Dataset location: {dataset_path}")
        print(f"📊 Images processed: {len(results['processed'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        return False
    
    finally:
        generator.cleanup()

def create_training_config(dataset_path: Path, lora_name: str, results: dict, trigger_word: str = None):
    """Create a training configuration file for the dataset"""
    
    trigger_phrase = trigger_word if trigger_word else lora_name
    
    config = {
        "dataset_info": {
            "lora_name": lora_name,
            "trigger_word": trigger_phrase,
            "created": datetime.now().isoformat(),
            "image_count": len(results["processed"]),
            "source_model": results.get("model_used", "unknown"),
            "custom_prompt": results.get("custom_prompt")
        },
        "recommended_training": {
            "steps": min(max(len(results["processed"]) * 100, 1000), 4000),
            "learning_rate": 1e-4,
            "batch_size": 1,
            "save_every": 500,
            "sample_prompts": [
                f"{trigger_phrase}",
                f"{trigger_phrase}, portrait",
                f"{trigger_phrase}, professional photo",
                f"a photo of {trigger_phrase}",
                f"{trigger_phrase}, headshot",
                f"close-up of {trigger_phrase}",
                f"{trigger_phrase} smiling",
                f"{trigger_phrase} looking at camera"
            ]
        },
        "accelerate_command": f"accelerate launch accelerate_flux_trainer.py --dataset_path {dataset_path} --output_name {lora_name}_lora --batch_size 1"
    }
    
    # Save config
    config_path = dataset_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Training config saved: {config_path}")
    
    # Create a quick start script
    script_content = f"""#!/bin/bash
# Quick start training script for {lora_name}
cd "$(dirname "$0")/../.."

echo "🚀 Starting FLUX LoRA training for {lora_name}"
echo "📁 Dataset: {dataset_path}"
echo "🏷️  Trigger word: {trigger_phrase}"

# Activate environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Run training
accelerate launch accelerate_flux_trainer.py \\
    --dataset_path "{dataset_path}" \\
    --output_name "{lora_name}_lora" \\
    --batch_size 1 \\
    --learning_rate 1e-4 \\
    --max_train_steps {config['recommended_training']['steps']} \\
    --save_every {config['recommended_training']['save_every']} \\
    --sample_prompts "{trigger_phrase}" "a photo of {trigger_phrase}" "{trigger_phrase}, portrait" "{trigger_phrase} smiling"

echo "✅ Training complete!"
echo "📁 Check outputs/ directory for your trained LoRA"
"""
    
    script_path = dataset_path / "start_training.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"🎯 Quick start script: {script_path}")

def list_available_datasets(datasets_dir: str = "/workspace/music-video-generator/dataset"):
    """List all available datasets in the datasets directory"""
    datasets_path = Path(datasets_dir)
    
    if not datasets_path.exists():
        print(f"📁 No datasets found - datasets directory doesn't exist: {datasets_dir}")
        return
    
    datasets = [d for d in datasets_path.iterdir() if d.is_dir()]
    
    if not datasets:
        print(f"📁 No datasets found in {datasets_dir}")
        return
    
    print(f"📋 Available datasets in {datasets_path}:")
    
    for dataset in datasets:
        image_count = len([f for f in dataset.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}])
        caption_count = len([f for f in dataset.iterdir() if f.suffix.lower() == '.txt'])
        
        print(f"\n  📂 {dataset.name}")
        print(f"     Images: {image_count}")
        print(f"     Captions: {caption_count}")
        
        # Check if training dataset exists
        training_path = Path("/workspace/music-video-generator/training_workspace/train_data") / dataset.name
        if training_path.exists():
            training_images = len([f for f in training_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}])
            training_captions = len([f for f in training_path.iterdir() if f.suffix.lower() == '.txt'])
            print(f"     Training dataset: ✅ ({training_images} images, {training_captions} captions)")
        else:
            print(f"     Training dataset: ❌ (not prepared)")

def list_training_datasets(training_workspace: str = "/workspace/music-video-generator/training_workspace"):
    """List all prepared training datasets"""
    workspace_path = Path(training_workspace)
    train_data_path = workspace_path / "train_data"
    
    if not train_data_path.exists():
        print("📁 No training datasets found - train_data directory doesn't exist")
        return
    
    datasets = [d for d in train_data_path.iterdir() if d.is_dir()]
    
    if not datasets:
        print("📁 No training datasets found in train_data directory")
        return
    
    print(f"📋 Prepared training datasets in {train_data_path}:")
    
    for dataset in datasets:
        config_path = dataset / "training_config.json"
        image_count = len(list(dataset.glob("*.jpg")) + list(dataset.glob("*.png")) + list(dataset.glob("*.jpeg")))
        caption_count = len(list(dataset.glob("*.txt")))
        
        print(f"\n  📂 {dataset.name}")
        print(f"     Images: {image_count}")
        print(f"     Captions: {caption_count}")
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                created = config.get("dataset_info", {}).get("created", "unknown")
                print(f"     Created: {created}")
                print(f"     Recommended steps: {config.get('recommended_training', {}).get('steps', 'unknown')}")
            except:
                print(f"     Config: error reading")
        else:
            print(f"     Config: not found")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for FLUX LoRA training")
    
    # Main operation
    parser.add_argument("--lora-name", type=str,
                       help="LoRA model name (used for source and destination folder names)")
    parser.add_argument("--datasets-dir", type=str, 
                       default="/workspace/music-video-generator/dataset",
                       help="Root datasets directory containing subject folders")
    
    # Options
    parser.add_argument("--workspace", type=str, 
                       default="/workspace/music-video-generator/training_workspace",
                       help="Training workspace directory")
    parser.add_argument("--model", type=str, 
                       default="microsoft/Florence-2-base",
                       help="Caption generation model (Florence-2-base, Florence-2-large, blip2-opt-2.7b, InternVL2-8B)")
    parser.add_argument("--trigger-word", type=str,
                       help="Trigger word(s) to include in captions (defaults to lora-name if not specified)")
    parser.add_argument("--prompt", type=str,
                       help="Custom prompt for caption generation")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing dataset")
    
    # Utility operations
    parser.add_argument("--list", action="store_true",
                       help="List available datasets in datasets directory")
    parser.add_argument("--list-training", action="store_true",
                       help="List prepared training datasets")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets(args.datasets_dir)
        return
    
    if args.list_training:
        list_training_datasets(args.workspace)
        return
    
    if not args.lora_name:
        parser.error("Must provide --lora-name (or use --list/--list-training)")
    
    # Prepare dataset
    success = prepare_training_dataset(
        lora_name=args.lora_name,
        datasets_dir=args.datasets_dir,
        training_workspace=args.workspace,
        caption_model=args.model,
        trigger_word=args.trigger_word,
        custom_prompt=args.prompt,
        overwrite=args.overwrite
    )
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"1. Review the generated captions in {args.workspace}/train_data/{args.lora_name}/")
        print(f"2. Run the quick start script: {args.workspace}/train_data/{args.lora_name}/start_training.sh")
        print(f"3. Or use accelerate_flux_trainer.py with custom settings")
    else:
        print(f"\n❌ Dataset preparation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

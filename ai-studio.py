#!/usr/bin/env python3
"""
AI Studio CLI (ai-studio)
Unified command-line interface for the FLUX LoRA training pipeline
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_script(script_path, args):
    """Run a script with the given arguments"""
    cmd = [sys.executable, script_path] + args
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return e.returncode
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="AI Studio CLI - Unified interface for FLUX LoRA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  setup          Download and setup required models
  prepare        Prepare dataset for training
  caption        Generate captions for images
  analyze        Analyze dataset statistics
  cleanup        Clean up dataset files
  train          Train FLUX LoRA model
  generate       Generate images using trained LoRA models
  list-models    List available trained LoRA models

Examples:
  ai-studio setup                                    # Download models
  ai-studio prepare dataset/my_photos                # Prepare dataset
  ai-studio caption dataset/my_photos --model florence2  # Generate captions
  ai-studio train dataset/my_photos --trigger "myname"   # Train LoRA
  ai-studio generate "portrait of myname" --model myname_lora  # Generate image
  ai-studio list-models                              # Show available models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Download and setup required models')
    setup_parser.add_argument('--model', choices=['florence2', 'blip2', 'internvl2', 'all'], 
                             default='florence2', help='Model to download (default: florence2)')
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset for training')
    prepare_parser.add_argument('dataset_path', help='Path to dataset directory')
    prepare_parser.add_argument('--min-size', type=int, default=512, help='Minimum image size')
    prepare_parser.add_argument('--max-size', type=int, default=2048, help='Maximum image size')
    prepare_parser.add_argument('--quality', type=int, default=95, help='JPEG quality (1-100)')
    
    # Caption command
    caption_parser = subparsers.add_parser('caption', help='Generate captions for images')
    caption_parser.add_argument('dataset_path', help='Path to dataset directory')
    caption_parser.add_argument('--model', choices=['florence2', 'blip2', 'internvl2'], 
                               default='florence2', help='Caption model to use')
    caption_parser.add_argument('--trigger', help='Trigger word to add to captions')
    caption_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing captions')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset statistics')
    analyze_parser.add_argument('dataset_path', help='Path to dataset directory')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up dataset files')
    cleanup_parser.add_argument('dataset_path', help='Path to dataset directory')
    cleanup_parser.add_argument('--remove-cache', action='store_true', help='Remove cache files')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be removed')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train FLUX LoRA model')
    train_parser.add_argument('dataset_path', help='Path to dataset directory')
    train_parser.add_argument('--trigger', required=True, help='Trigger word for LoRA training')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    train_parser.add_argument('--rank', type=int, default=16, help='LoRA rank')
    train_parser.add_argument('--output-name', help='Output model name (default: auto-generated)')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate images using trained LoRA models')
    generate_parser.add_argument('prompt', nargs='?', help='Text prompt for image generation')
    generate_parser.add_argument('--model', help='LoRA model name/path to use')
    generate_parser.add_argument('--epoch', type=int, help='Specific epoch to use')
    generate_parser.add_argument('--guidance-scale', '--scale', type=float, default=3.5, help='Guidance scale (default: 3.5)')
    generate_parser.add_argument('--steps', type=int, default=20, help='Number of inference steps')
    generate_parser.add_argument('--seed', type=int, help='Random seed (random if not specified)')
    generate_parser.add_argument('--width', type=int, default=1024, help='Image width')
    generate_parser.add_argument('--height', type=int, default=1024, help='Image height')
    generate_parser.add_argument('--num-images', '--count', type=int, default=1, help='Number of images to generate')
    generate_parser.add_argument('--prompts-file', '--prompt-file', dest='prompts_file', help='File containing prompts (one per line)')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available trained LoRA models')
    list_parser.add_argument('--detail', action='store_true', help='Show detailed model information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Map commands to script paths
    script_dir = Path(__file__).parent / "src"
    
    if args.command == 'setup':
        script_path = script_dir / "setup" / "setup_models.py"
        script_args = []
        if args.model != 'all':
            script_args.extend(['--model', args.model])
        return run_script(script_path, script_args)
    
    elif args.command == 'prepare':
        script_path = script_dir / "data_preparation" / "prepare_dataset.py"
        script_args = [args.dataset_path]
        script_args.extend(['--min-size', str(args.min_size)])
        script_args.extend(['--max-size', str(args.max_size)])
        script_args.extend(['--quality', str(args.quality)])
        return run_script(script_path, script_args)
    
    elif args.command == 'caption':
        script_path = script_dir / "data_preparation" / "generate_captions.py"
        script_args = [args.dataset_path]
        script_args.extend(['--model', args.model])
        if args.trigger:
            script_args.extend(['--trigger', args.trigger])
        if args.overwrite:
            script_args.append('--overwrite')
        return run_script(script_path, script_args)
    
    elif args.command == 'analyze':
        script_path = script_dir / "data_preparation" / "analyze_dataset.py"
        script_args = [args.dataset_path]
        return run_script(script_path, script_args)
    
    elif args.command == 'cleanup':
        script_path = script_dir / "data_preparation" / "cleanup_dataset.py"
        script_args = [args.dataset_path]
        if args.remove_cache:
            script_args.append('--remove-cache')
        if args.dry_run:
            script_args.append('--dry-run')
        return run_script(script_path, script_args)
    
    elif args.command == 'train':
        script_path = script_dir / "training" / "accelerate_flux_trainer.py"
        script_args = [args.dataset_path]
        script_args.extend(['--trigger_word', args.trigger])
        script_args.extend(['--max_train_epochs', str(args.epochs)])
        script_args.extend(['--learning_rate', str(args.learning_rate)])
        script_args.extend(['--train_batch_size', str(args.batch_size)])
        script_args.extend(['--network_dim', str(args.rank)])
        if args.output_name:
            script_args.extend(['--output_name', args.output_name])
        return run_script(script_path, script_args)
    
    elif args.command == 'generate':
        script_path = script_dir / "generation" / "generate_flux_images.py"
        script_args = []
        
        # Validate that either prompt or prompts-file is provided
        if not args.prompt and not args.prompts_file:
            print("Error: Either provide a prompt or use --prompt-file/--prompts-file")
            return 1
        
        if args.prompts_file:
            script_args.extend(['--prompt-file', args.prompts_file])
        elif args.prompt:
            script_args.extend(['--prompt', args.prompt])
        
        if args.model:
            script_args.extend(['--model', args.model])
        if args.epoch is not None:
            script_args.extend(['--epoch', str(args.epoch)])
        script_args.extend(['--guidance-scale', str(args.guidance_scale)])
        script_args.extend(['--steps', str(args.steps)])
        if args.seed is not None:
            script_args.extend(['--seed', str(args.seed)])
        script_args.extend(['--width', str(args.width)])
        script_args.extend(['--height', str(args.height)])
        script_args.extend(['--num-images', str(args.num_images)])
        return run_script(script_path, script_args)
    
    elif args.command == 'list-models':
        script_path = script_dir / "generation" / "generate_flux_images.py"
        script_args = ['--list-models']
        if args.detail:
            script_args.append('--detail')
        return run_script(script_path, script_args)
    
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

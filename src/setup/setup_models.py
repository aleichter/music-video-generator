#!/usr/bin/env python3
"""
Model Setup Script for FLUX LoRA Training Pipeline
Downloads and caches all required models for production use.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List
import torch

def check_system_requirements():
    """Check system requirements and available resources"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  No CUDA GPUs detected - CPU mode only")
    
    # Check disk space
    cache_path = Path("/workspace/.cache")
    if cache_path.exists():
        # Rough estimate: we need ~50GB for all models
        free_space = os.statvfs(str(cache_path)).f_bavail * os.statvfs(str(cache_path)).f_frsize / 1024**3
        print(f"✅ Available disk space: {free_space:.1f}GB")
        if free_space < 60:
            print("⚠️  Warning: You may need more disk space (recommended: 60GB+)")
    else:
        print("ℹ️  Cache directory will be created")

def setup_huggingface_environment(token: Optional[str] = None, cache_dir: str = "/workspace/.cache/huggingface"):
    """Setup HuggingFace environment and authentication"""
    print("🔧 Setting up HuggingFace environment...")
    
    # Set cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(cache_path)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_path)
    
    print(f"✅ Cache directory: {cache_path}")
    
    # Handle authentication
    if token:
        print("🔐 Setting up HuggingFace authentication...")
        try:
            from huggingface_hub import login
            login(token=token, write_permission=False)
            print("✅ Authentication successful")
        except Exception as e:
            print(f"⚠️  Authentication failed: {e}")
            print("   Continuing without authentication (may limit model access)")
    else:
        print("ℹ️  No token provided - using public models only")
    
    # Create models directory structure
    models_dir = cache_path / "hub"
    models_dir.mkdir(exist_ok=True)
    
    return cache_path

def download_model_with_progress(model_name: str, model_type: str = "diffusers", revision: str = "main"):
    """Download a model with progress tracking"""
    print(f"📥 Downloading {model_type}: {model_name}")
    
    try:
        if model_type == "diffusers":
            from diffusers import FluxPipeline, AutoencoderKL
            from transformers import CLIPTextModel, T5EncoderModel
            
            if "FLUX" in model_name:
                # Download FLUX pipeline components
                pipeline = FluxPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    revision=revision,
                    cache_dir=os.environ['HF_HOME']
                )
                print(f"✅ {model_name} downloaded successfully")
                return True
            else:
                # Other diffusers models
                from diffusers import DiffusionPipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    revision=revision,
                    cache_dir=os.environ['HF_HOME']
                )
                print(f"✅ {model_name} downloaded successfully")
                return True
                
        elif model_type == "transformers":
            # Handle different types of transformers models
            if "blip2" in model_name.lower():
                # BLIP2 models
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                
                processor = Blip2Processor.from_pretrained(
                    model_name,
                    cache_dir=os.environ['HF_HOME']
                )
                
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                    cache_dir=os.environ['HF_HOME']
                )
                
                print(f"✅ {model_name} downloaded successfully")
                return True
            else:
                # Generic transformers models
                from transformers import AutoModel, AutoProcessor, AutoTokenizer
                
                # Download model and processor/tokenizer
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=os.environ['HF_HOME']
                )
                
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        cache_dir=os.environ['HF_HOME']
                    )
                except:
                    # Fallback to tokenizer if processor not available
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        cache_dir=os.environ['HF_HOME']
                    )
                
                print(f"✅ {model_name} downloaded successfully")
                return True
            
        elif model_type == "internvl":
            # Special handling for InternVL models
            from transformers import AutoModel, AutoTokenizer
            
            # InternVL models use AutoModel and AutoTokenizer
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=os.environ['HF_HOME']
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=os.environ['HF_HOME']
            )
            
            print(f"✅ InternVL {model_name} downloaded successfully")
            return True
            
        elif model_type == "florence2":
            # Special handling for Florence2
            from transformers import AutoProcessor
            
            # Try the specific Florence2 class first
            try:
                from transformers import Florence2ForConditionalGeneration
                model = Florence2ForConditionalGeneration.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    cache_dir=os.environ['HF_HOME']
                )
            except ImportError:
                # Fallback to AutoModel with trust_remote_code
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    cache_dir=os.environ['HF_HOME']
                )
            
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=os.environ['HF_HOME']
            )
            
            print(f"✅ Florence2 {model_name} downloaded successfully")
            return True
            
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def get_model_list():
    """Get the list of models to download"""
    models = [
        # Core FLUX models
        {
            "name": "black-forest-labs/FLUX.1-dev",
            "type": "diffusers",
            "description": "FLUX.1 Development model (main training target)",
            "size_gb": 23.8,
            "required": True
        },
        {
            "name": "black-forest-labs/FLUX.1-schnell", 
            "type": "diffusers",
            "description": "FLUX.1 Schnell (fast inference)",
            "size_gb": 23.8,
            "required": False
        },
        
        # Caption generation models
        {
            "name": "Salesforce/blip2-opt-2.7b",
            "type": "transformers",
            "description": "BLIP2 - Reliable caption generation model",
            "size_gb": 5.4,
            "required": True
        },
        {
            "name": "OpenGVLab/InternVL2-8B",
            "type": "internvl",
            "description": "InternVL2 8B - Advanced multimodal model for detailed captioning",
            "size_gb": 16.1,
            "required": False
        },
        {
            "name": "microsoft/Florence-2-large",
            "type": "florence2", 
            "description": "Florence2 Large - Advanced vision-language model for captioning",
            "size_gb": 1.5,
            "required": False
        },
        {
            "name": "microsoft/Florence-2-base",
            "type": "florence2",
            "description": "Florence2 Base - Smaller, faster captioning model", 
            "size_gb": 0.8,
            "required": False
        },
        {
            "name": "llava-hf/llava-1.5-7b-hf",
            "type": "transformers", 
            "description": "LLaVA - Multi-modal conversation model",
            "size_gb": 13.5,
            "required": False
        }
    ]
    
    return models

def main():
    parser = argparse.ArgumentParser(description="Setup models for FLUX LoRA training pipeline")
    parser.add_argument("--token", type=str, help="HuggingFace access token")
    parser.add_argument("--cache-dir", type=str, default="/workspace/.cache/huggingface", 
                       help="Cache directory for models")
    parser.add_argument("--models", type=str, nargs="+", 
                       help="Specific models to download (default: required models only)")
    parser.add_argument("--all", action="store_true", 
                       help="Download all models including optional ones")
    parser.add_argument("--list", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check system requirements, don't download")
    
    args = parser.parse_args()
    
    # Check system requirements
    check_system_requirements()
    
    if args.check_only:
        print("✅ System check complete")
        return
    
    # Get model list
    models = get_model_list()
    
    if args.list:
        print("\n📋 Available models:")
        total_size = 0
        for model in models:
            status = "Required" if model["required"] else "Optional"
            print(f"  {model['name']}")
            print(f"    Type: {model['type']}")
            print(f"    Size: {model['size_gb']:.1f}GB")
            print(f"    Status: {status}")
            print(f"    Description: {model['description']}")
            print()
            if model["required"] or args.all:
                total_size += model["size_gb"]
        
        print(f"Total download size: {total_size:.1f}GB")
        return
    
    # Setup HuggingFace environment
    cache_path = setup_huggingface_environment(args.token, args.cache_dir)
    
    # Determine which models to download
    if args.models:
        # Download specific models
        models_to_download = [m for m in models if m["name"] in args.models]
        if not models_to_download:
            print("❌ No matching models found")
            return
    elif args.all:
        # Download all models
        models_to_download = models
    else:
        # Download required models only
        models_to_download = [m for m in models if m["required"]]
    
    # Calculate total size
    total_size = sum(m["size_gb"] for m in models_to_download)
    print(f"\n📊 Download plan:")
    print(f"  Models: {len(models_to_download)}")
    print(f"  Total size: {total_size:.1f}GB")
    print(f"  Cache location: {cache_path}")
    
    # Confirm download
    if not args.all and not args.models:
        response = input("\nProceed with download? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Download cancelled")
            return
    
    # Download models
    print("\n🚀 Starting model downloads...")
    successful = []
    failed = []
    
    for i, model in enumerate(models_to_download, 1):
        print(f"\n[{i}/{len(models_to_download)}] {model['name']}")
        
        if download_model_with_progress(model["name"], model["type"]):
            successful.append(model["name"])
        else:
            failed.append(model["name"])
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {len(successful)}")
    for model in successful:
        print(f"   {model}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for model in failed:
            print(f"   {model}")
    
    # Save model manifest
    manifest = {
        "timestamp": str(Path().resolve()),
        "cache_directory": str(cache_path),
        "successful_downloads": successful,
        "failed_downloads": failed,
        "models": {m["name"]: {
            "type": m["type"],
            "size_gb": m["size_gb"],
            "description": m["description"]
        } for m in models_to_download}
    }
    
    manifest_path = cache_path / "download_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📋 Download manifest saved: {manifest_path}")
    
    if successful:
        print(f"\n✅ Setup complete! {len(successful)} models ready for use.")
        print("\nNext steps:")
        print("1. Run: python generate_captions.py --help")
        print("2. Run: python generate_images.py --help") 
        print("3. Run: python accelerate_flux_trainer.py --help")
    else:
        print(f"\n❌ Setup failed - no models downloaded successfully")
        print("Check your internet connection and HuggingFace token")

if __name__ == "__main__":
    main()

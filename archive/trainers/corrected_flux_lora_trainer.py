#!/usr/bin/env python3

"""
Corrected FLUX LoRA Trainer - Using proper diffusion training protocol
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
import gc
from diffusers import FluxPipeline
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
import torchvision.transforms as transforms

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectedFluxDataset(Dataset):
    """Dataset that properly prepares data for FLUX diffusion training"""
    
    def __init__(self, dataset_path, resolution=1024):
        self.dataset_path = Path(dataset_path)
        self.resolution = resolution
        self.data = self.load_data()
        
        # Image preprocessing for FLUX
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
    def load_data(self):
        """Load image paths and captions"""
        data = []
        
        # Look for captions.txt file
        captions_file = self.dataset_path / "captions.txt"
        if not captions_file.exists():
            raise ValueError(f"No captions.txt found in {self.dataset_path}")
            
        # Parse captions file
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Expected format: filename.jpg: caption text here
                if ':' in line:
                    filename, caption = line.split(':', 1)
                    filename = filename.strip()
                    caption = caption.strip()
                    img_path = self.dataset_path / filename
                    
                    if img_path.exists():
                        data.append({
                            'image_path': str(img_path),
                            'caption': caption
                        })
                    else:
                        logger.warning(f"Image not found: {img_path}")
                else:
                    logger.warning(f"Invalid caption format: {line}")
        
        logger.info(f"Loaded {len(data)} image-caption pairs")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and preprocess image
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)
        
        return {
            'pixel_values': image,
            'caption': item['caption']
        }

class CorrectedFluxLoRATrainer:
    """Corrected FLUX LoRA trainer using proper diffusion training protocol"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("🚀 Initializing Corrected FLUX LoRA Trainer")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🔧 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"💾 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"🔋 GPU Memory: {gpu_memory:.1f} GB")
        
        self.pipe = None
        self.dataset = None
        self.noise_scheduler = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        
    def load_pipeline(self):
        """Load FLUX pipeline and extract components for training"""
        logger.info("📥 Loading FLUX pipeline...")
        
        try:
            # Load full pipeline first
            self.pipe = FluxPipeline.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            ).to(self.device)
            
            # Extract components for training
            self.vae = self.pipe.vae
            self.text_encoder = self.pipe.text_encoder
            self.tokenizer = self.pipe.tokenizer
            self.noise_scheduler = self.pipe.scheduler
            
            # Set components to eval mode (only transformer will be trained)
            self.vae.eval()
            self.text_encoder.eval()
            
            # Disable gradients for frozen components
            for param in self.vae.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
            logger.info("✅ Pipeline loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def load_dataset(self):
        """Load and prepare dataset"""
        logger.info("📂 Loading dataset...")
        
        self.dataset = CorrectedFluxDataset(
            self.args.dataset_path,
            resolution=self.args.resolution
        )
        
        logger.info(f"✅ Dataset loaded: {len(self.dataset)} samples")
    
    def setup_lora(self):
        """Setup LoRA using PEFT on the transformer"""
        logger.info("🎯 Setting up LoRA...")
        
        # Count target modules
        target_modules = []
        for name, module in self.pipe.transformer.named_modules():
            if isinstance(module, nn.Linear) and any(keyword in name for keyword in [
                'to_q', 'to_k', 'to_v', 'to_out'
            ]):
                target_modules.append(name)
        
        logger.info(f"📊 Targeting {len(target_modules)} modules")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=None,  # No specific task type for diffusion models
        )
        
        # Apply LoRA to transformer
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)
        
        # Convert to bfloat16
        logger.info("🔄 Converting LoRA parameters to bfloat16...")
        for name, param in self.pipe.transformer.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
        
        # Print trainable parameters
        self.pipe.transformer.print_trainable_parameters()
        
        logger.info("✅ LoRA setup complete!")
    
    def encode_text(self, captions):
        """Encode text captions using FLUX text encoder"""
        with torch.no_grad():
            text_inputs = self.tokenizer(
                captions,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
            
        return text_embeddings
    
    def encode_images(self, pixel_values):
        """Encode images using VAE"""
        with torch.no_grad():
            pixel_values = pixel_values.to(device=self.device, dtype=torch.bfloat16)
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    def train(self):
        """Main training loop using proper diffusion training protocol"""
        logger.info("🎓 Starting corrected FLUX LoRA training...")
        
        # Ensure pipeline and dataset are loaded
        if self.pipe is None:
            logger.info("Pipeline not loaded, loading now...")
            self.load_pipeline()
        
        if self.dataset is None:
            logger.info("Dataset not loaded, loading now...")
            self.load_dataset()
        
        # Setup LoRA
        self.setup_lora()
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.pipe.transformer.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01
        )
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            logger.info(f"\n📚 Epoch {epoch + 1}/{self.args.num_epochs}")
            
            epoch_loss = 0.0
            progress_bar = tqdm(
                dataloader, 
                desc=f"Epoch {epoch + 1}",
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    optimizer.zero_grad()
                    
                    # Get batch data
                    pixel_values = batch['pixel_values']
                    captions = batch['caption']
                    
                    # Encode text
                    text_embeddings = self.encode_text(captions)
                    
                    # Encode images to latents
                    latents = self.encode_images(pixel_values)
                    
                    # Sample noise and timesteps
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=self.device
                    ).long()
                    
                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get model prediction using proper FLUX format
                    model_pred = self.pipe.transformer(
                        hidden_states=noisy_latents,
                        encoder_hidden_states=text_embeddings,
                        timestep=timesteps,
                        return_dict=False
                    )[0]
                    
                    # Compute diffusion loss
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = latents
                    
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.pipe.transformer.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"📊 Epoch {epoch + 1} complete - Average loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:  # Save every epoch for now
                self.save_checkpoint(epoch + 1)
        
        logger.info("🎉 Corrected FLUX LoRA training complete!")
    
    def save_checkpoint(self, epoch):
        """Save LoRA checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / f"corrected_flux_lora_epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving corrected LoRA checkpoint for epoch {epoch}...")
        
        # Save LoRA weights
        self.pipe.transformer.save_pretrained(checkpoint_dir)
        
        logger.info(f"✅ Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Corrected FLUX LoRA Training")
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dataset_path", default="./dataset/anddrrew")
    parser.add_argument("--output_dir", default="./models/anddrrew_corrected_flux_lora")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    
    trainer = CorrectedFluxLoRATrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

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

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFluxDataset(Dataset):
    """Simple dataset for FLUX LoRA training"""
    
    def __init__(self, dataset_path, resolution=512):
        self.dataset_path = Path(dataset_path)
        self.resolution = resolution
        self.data = self.load_data()
        
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
                if ':' in line:
                    filename, caption = line.split(':', 1)
                    filename = filename.strip()
                    caption = caption.strip()
                    
                    image_path = self.dataset_path / filename
                    if image_path.exists():
                        data.append({
                            'image_path': str(image_path),
                            'caption': caption
                        })
        
        if not data:
            raise ValueError(f"No valid training data found in {self.dataset_path}")
            
        logger.info(f"Loaded {len(data)} training samples")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            
            # Simple center crop and resize
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))
            image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            
            # Convert to tensor [-1, 1]
            image_array = np.array(image) / 127.5 - 1.0
            image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1)
            
            return {
                'image': image_tensor,
                'caption': item['caption'],
            }
            
        except Exception as e:
            logger.warning(f"Error loading image {item['image_path']}: {e}")
            return {
                'image': torch.zeros(3, self.resolution, self.resolution),
                'caption': item['caption'],
            }

class FixedFluxLoRATrainer:
    """Fixed FLUX LoRA trainer with proper module targeting"""
    
    def __init__(self, args):
        self.args = args
        self.pipe = None
        self.model = None
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🚀 Initializing Fixed FLUX LoRA Trainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Args: {vars(args)}")
        
    def load_pipeline(self):
        """Load FLUX pipeline with memory optimization"""
        logger.info("📥 Loading FLUX pipeline...")
        
        try:
            self.pipe = FluxPipeline.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            ).to(self.device)
            
            # Enable memory efficient attention
            self.pipe.enable_model_cpu_offload()
            
            logger.info("✅ Pipeline loaded successfully!")
            logger.info(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        except Exception as e:
            logger.error(f"❌ Failed to load pipeline: {e}")
            raise
    
    def setup_lora(self):
        """Setup LoRA with proper module targeting across all transformer blocks"""
        logger.info("🎯 Setting up LoRA with improved targeting...")
        
        # Find attention modules across ALL transformer blocks
        target_modules = []
        
        # Get modules from transformer_blocks (double blocks)
        for i in range(19):  # FLUX has 19 transformer blocks
            for suffix in ['to_q', 'to_k', 'to_v', 'to_out.0']:
                module_name = f"transformer_blocks.{i}.attn.{suffix}"
                target_modules.append(module_name)
                
        # Get modules from single_transformer_blocks
        for i in range(38):  # FLUX has 38 single transformer blocks
            for suffix in ['to_q', 'to_k', 'to_v']:
                module_name = f"single_transformer_blocks.{i}.attn.{suffix}"
                target_modules.append(module_name)
        
        logger.info(f"🎯 Targeting {len(target_modules)} modules across all transformer blocks")
        logger.info("📋 Target modules preview (first 10):")
        for i, target in enumerate(target_modules[:10]):
            logger.info(f"  {i+1}. {target}")
        logger.info(f"  ... and {len(target_modules) - 10} more")
        
        # Verify modules exist
        existing_modules = []
        for name, module in self.pipe.transformer.named_modules():
            if name in target_modules:
                existing_modules.append(name)
        
        logger.info(f"✅ Found {len(existing_modules)} existing target modules out of {len(target_modules)}")
        
        if len(existing_modules) == 0:
            logger.error("❌ No target modules found! This will not work.")
            raise ValueError("No valid target modules found")
        
        # Create LoRA config with more conservative settings
        lora_config = LoraConfig(
            r=8,  # Smaller rank for stability
            lora_alpha=16,  # Reduced alpha
            target_modules=existing_modules,  # Use only existing modules
            lora_dropout=0.05,  # Reduced dropout
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        logger.info(f"📊 LoRA Config:")
        logger.info(f"  Rank: {lora_config.r}")
        logger.info(f"  Alpha: {lora_config.lora_alpha}")
        logger.info(f"  Dropout: {lora_config.lora_dropout}")
        logger.info(f"  Target modules: {len(existing_modules)}")
        
        # Apply LoRA
        try:
            self.model = get_peft_model(self.pipe.transformer, lora_config)
            
            # Count LoRA parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"📊 Model Statistics:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
            
            # Replace transformer in pipeline
            self.pipe.transformer = self.model
            
            logger.info("✅ LoRA setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup LoRA: {e}")
            raise
    
    def load_dataset(self):
        """Load training dataset"""
        logger.info("📂 Loading dataset...")
        
        try:
            self.dataset = SimpleFluxDataset(
                self.args.dataset_path,
                resolution=self.args.resolution
            )
            
            logger.info(f"✅ Dataset loaded: {len(self.dataset)} samples")
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def train(self):
        """Main training loop"""
        logger.info("🎓 Starting training...")
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0
            
            logger.info(f"\n📚 Epoch {epoch + 1}/{self.args.num_epochs}")
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move batch to device
                    images = batch['image'].to(self.device, dtype=torch.bfloat16)
                    captions = batch['caption']
                    
                    # Encode images and text
                    with torch.no_grad():
                        # Encode images to latents
                        latents = self.pipe.vae.encode(images).latent_dist.sample()
                        latents = latents * self.pipe.vae.config.scaling_factor
                        
                        # Encode text - returns (prompt_embeds, pooled_prompt_embeds, text_ids)
                        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
                            captions,
                            captions,  # prompt_2
                            device=self.device,
                            num_images_per_prompt=1,
                        )
                    
                    # Sample noise and timesteps for FLUX flow matching
                    noise = torch.randn_like(latents)
                    timesteps = torch.rand((latents.shape[0],), device=self.device)
                    
                    # For FLUX flow matching: interpolate between noise and data
                    noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise
                    
                    # Convert timesteps to the format expected by FLUX
                    timesteps = timesteps * 1000.0  # Scale to [0, 1000]
                    
                    # Prepare additional inputs for FLUX
                    guidance = torch.full((latents.shape[0],), 3.5, device=self.device, dtype=torch.bfloat16)
                    
                    # Create img_ids for FLUX (required for positional encoding)
                    batch_size, channels, height, width = latents.shape
                    img_ids = torch.zeros((batch_size, height * width, 3), device=self.device, dtype=torch.bfloat16)
                    
                    # Forward pass through transformer
                    model_pred = self.model(
                        hidden_states=noisy_latents,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timesteps.long(),  # Convert to long
                        img_ids=img_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        return_dict=False,
                    )[0]
                    
                    # Compute loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Update progress
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                        'gpu_mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB'
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_idx}: {e}")
                    continue
                
                # Cleanup
                if batch_idx % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"📊 Epoch {epoch + 1} complete - Average loss: {avg_loss:.4f}")
            
            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(epoch + 1)
        
        logger.info("🎉 Training complete!")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / f"fixed_flux_lora_epoch_{epoch}_peft"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving checkpoint for epoch {epoch}...")
        
        try:
            self.model.save_pretrained(checkpoint_dir)
            logger.info(f"✅ Checkpoint saved to {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fixed FLUX LoRA Training")
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dataset_path", default="dataset/anddrrew")
    parser.add_argument("--output_dir", default="models/anddrrew_fixed_flux_lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)  # Reduced learning rate
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        trainer = FixedFluxLoRATrainer(args)
        trainer.load_pipeline()
        trainer.setup_lora()
        trainer.load_dataset()
        trainer.train()
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

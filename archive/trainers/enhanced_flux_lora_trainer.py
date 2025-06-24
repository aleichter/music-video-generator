#!/usr/bin/env python3

"""
Enhanced FLUX LoRA Trainer - Fixed B weight updates with explicit gradient monitoring
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

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFluxDataset(Dataset):
    """Simple dataset for FLUX LoRA training"""
    
    def __init__(self, dataset_path, resolution=1024):
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
                    
                    # Find the actual image file
                    image_path = self.dataset_path / filename
                    if image_path.exists():
                        data.append({
                            'image_path': str(image_path),
                            'caption': caption
                        })
                    else:
                        logger.warning(f"Image not found: {image_path}")
        
        if not data:
            raise ValueError(f"No valid image-caption pairs found in {self.dataset_path}")
            
        logger.info(f"Loaded {len(data)} image-caption pairs")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image = Image.open(item['image_path']).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - 0.5) / 0.5
        
        return {
            'image': image_tensor,
            'caption': item['caption']
        }

class EnhancedFluxLoRATrainer:
    """Enhanced FLUX LoRA trainer with B weight debugging"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.dataset = None
        self.initial_b_weights = []
        
        logger.info(f"🚀 Initializing Enhanced FLUX LoRA Trainer")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🔧 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"💾 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"🔋 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_pipeline(self):
        """Load FLUX pipeline"""
        logger.info("📥 Loading FLUX pipeline...")
        
        self.pipe = FluxPipeline.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(self.device)
        
        logger.info("✅ Pipeline loaded successfully!")
    
    def setup_lora(self):
        """Setup LoRA configuration and apply to transformer"""
        logger.info("🎯 Setting up LoRA...")
        
        # Create target modules list for all attention layers
        target_modules = []
        
        # Add transformer_blocks (double blocks) - 19 blocks
        for i in range(19):
            target_modules.extend([
                f"transformer_blocks.{i}.attn.to_q",
                f"transformer_blocks.{i}.attn.to_k", 
                f"transformer_blocks.{i}.attn.to_v",
                f"transformer_blocks.{i}.attn.to_out.0"
            ])
        
        # Add single_transformer_blocks - 38 blocks
        for i in range(38):
            target_modules.extend([
                f"single_transformer_blocks.{i}.attn.to_q",
                f"single_transformer_blocks.{i}.attn.to_k",
                f"single_transformer_blocks.{i}.attn.to_v"
            ])
        
        logger.info(f"📊 Targeting {len(target_modules)} modules")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,  # Start with no dropout
            bias="none",
            task_type=None,  # Custom task
            init_lora_weights=True,  # Ensure proper initialization
        )
        
        # Apply LoRA to transformer
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)
        
        # Convert LoRA parameters to bfloat16 to match base model
        logger.info("🔄 Converting LoRA parameters to bfloat16...")
        for name, param in self.pipe.transformer.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.data = param.data.to(torch.bfloat16)
        
        # Enable training mode
        self.pipe.transformer.train()
        
        # Print model info
        self.pipe.transformer.print_trainable_parameters()
        
        logger.info("✅ LoRA setup complete!")
    
    def initialize_b_weights_with_small_values(self):
        """Initialize B weights with small non-zero values instead of zeros"""
        logger.info("🎲 Initializing LoRA B weights with small random values...")
        
        b_weight_count = 0
        for name, param in self.pipe.transformer.named_parameters():
            if 'lora_B' in name and param.requires_grad:
                # Initialize with small random values instead of zeros
                with torch.no_grad():
                    param.data = torch.randn_like(param.data) * 0.001  # Small random values
                b_weight_count += 1
                logger.debug(f"Initialized {name} with small random values")
        
        logger.info(f"✅ Initialized {b_weight_count} LoRA B parameters with small random values")
    
    def capture_initial_b_weights(self):
        """Capture initial B weights for monitoring"""
        logger.info("📸 Capturing initial LoRA B weights...")
        
        self.initial_b_weights = []
        b_weight_names = []
        
        for name, param in self.pipe.transformer.named_parameters():
            if 'lora_B' in name and param.requires_grad:
                self.initial_b_weights.append(param.data.clone())
                b_weight_names.append(name)
        
        logger.info(f"✅ Captured {len(self.initial_b_weights)} LoRA B weights for monitoring")
        
        # Log first few B weight statistics
        for i, (name, initial_weight) in enumerate(zip(b_weight_names[:3], self.initial_b_weights[:3])):
            logger.info(f"B weight {i}: {name}")
            logger.info(f"  Shape: {initial_weight.shape}")
            logger.info(f"  Mean: {initial_weight.mean():.6f}, Std: {initial_weight.std():.6f}")
            logger.info(f"  Min: {initial_weight.min():.6f}, Max: {initial_weight.max():.6f}")
    
    def check_gradient_flow(self):
        """Check if gradients are flowing to LoRA parameters"""
        lora_a_grads = []
        lora_b_grads = []
        
        for name, param in self.pipe.transformer.named_parameters():
            if param.grad is not None:
                if 'lora_A' in name:
                    grad_norm = param.grad.norm().item()
                    lora_a_grads.append(grad_norm)
                elif 'lora_B' in name:
                    grad_norm = param.grad.norm().item()
                    lora_b_grads.append(grad_norm)
        
        a_grad_sum = sum(lora_a_grads)
        b_grad_sum = sum(lora_b_grads)
        
        logger.info(f"🔍 Gradient flow check:")
        logger.info(f"  LoRA A gradients: {len(lora_a_grads)} layers, total norm: {a_grad_sum:.6f}")
        logger.info(f"  LoRA B gradients: {len(lora_b_grads)} layers, total norm: {b_grad_sum:.6f}")
        
        return a_grad_sum > 0, b_grad_sum > 0
    
    def check_b_weight_updates(self):
        """Check if LoRA B weights have been updated"""
        current_b_weights = []
        
        for name, param in self.pipe.transformer.named_parameters():
            if 'lora_B' in name and param.requires_grad:
                current_b_weights.append(param.data)
        
        changes_detected = False
        total_change = 0.0
        
        for i, (initial, current) in enumerate(zip(self.initial_b_weights[:3], current_b_weights[:3])):
            change = (current - initial).abs().sum().item()
            total_change += change
            logger.info(f"B weight {i}: change = {change:.8f}")
            if change > 1e-6:
                changes_detected = True
        
        logger.info(f"Total B weight change: {total_change:.8f}")
        return changes_detected, total_change
    
    def load_dataset(self):
        """Load training dataset"""
        logger.info("📂 Loading dataset...")
        
        self.dataset = SimpleFluxDataset(
            self.args.dataset_path,
            resolution=self.args.resolution
        )
        
        logger.info(f"✅ Dataset loaded: {len(self.dataset)} samples")
    
    def train(self):
        """Main training loop with enhanced B weight monitoring"""
        logger.info("🎓 Starting enhanced LoRA training...")
        
        # Ensure pipeline is loaded
        if self.pipe is None:
            logger.info("Pipeline not loaded, loading now...")
            self.load_pipeline()
        
        # Ensure dataset is loaded
        if self.dataset is None:
            logger.info("Dataset not loaded, loading now...")
            self.load_dataset()
        
        # Setup LoRA
        self.setup_lora()
        
        # Initialize B weights with small values instead of zeros
        self.initialize_b_weights_with_small_values()
        
        # Capture initial B weights
        self.capture_initial_b_weights()
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Setup optimizer with higher learning rate for B weights
        lora_a_params = []
        lora_b_params = []
        
        for name, param in self.pipe.transformer.named_parameters():
            if param.requires_grad:
                if 'lora_A' in name:
                    lora_a_params.append(param)
                elif 'lora_B' in name:
                    lora_b_params.append(param)
        
        # Create separate optimizers with different learning rates
        optimizer_a = torch.optim.AdamW(
            lora_a_params,
            lr=self.args.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        optimizer_b = torch.optim.AdamW(
            lora_b_params,
            lr=self.args.learning_rate * 2.0,  # Higher learning rate for B weights
            weight_decay=0.01,
            eps=1e-8
        )
        
        logger.info(f"📊 LoRA A parameters: {len(lora_a_params)}")
        logger.info(f"📊 LoRA B parameters: {len(lora_b_params)}")
        logger.info(f"🎯 LoRA A learning rate: {self.args.learning_rate}")
        logger.info(f"🎯 LoRA B learning rate: {self.args.learning_rate * 2.0}")
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0
            
            logger.info(f"\n📚 Epoch {epoch + 1}/{self.args.num_epochs}")
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move batch to device
                    images = batch['image'].to(self.device, dtype=torch.bfloat16)
                    captions = batch['caption']
                    
                    # Zero gradients
                    optimizer_a.zero_grad()
                    optimizer_b.zero_grad()
                    
                    # Encode inputs
                    with torch.no_grad():
                        latents = self.pipe.vae.encode(images).latent_dist.sample()
                        latents = latents * self.pipe.vae.config.scaling_factor
                        
                        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
                            captions, captions,
                            device=self.device,
                            num_images_per_prompt=1,
                        )
                    
                    # Sample noise and timesteps
                    noise = torch.randn_like(latents, dtype=torch.bfloat16)
                    timesteps = torch.rand((latents.shape[0],), device=self.device, dtype=torch.bfloat16)
                    
                    # FLUX flow matching
                    noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise
                    timesteps_scaled = (timesteps * 1000.0).to(torch.long)
                    
                    guidance = torch.full((latents.shape[0],), 3.5, device=self.device, dtype=torch.bfloat16)
                    
                    batch_size, channels, height, width = latents.shape
                    img_ids = torch.zeros((height * width, 3), device=self.device, dtype=torch.bfloat16)
                    
                    # Forward pass through transformer with LoRA
                    self.pipe.transformer.train()  # Ensure training mode
                    
                    model_pred = self.pipe.transformer(
                        hidden_states=noisy_latents,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timesteps_scaled,
                        img_ids=img_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        return_dict=False,
                    )[0]
                    
                    # Compute loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Check gradient flow on first batch
                    if batch_idx == 0:
                        has_a_grads, has_b_grads = self.check_gradient_flow()
                        if not has_b_grads:
                            logger.warning("⚠️  No gradients detected for LoRA B weights!")
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(lora_a_params + lora_b_params, 1.0)
                    
                    # Update parameters
                    optimizer_a.step()
                    optimizer_b.step()
                    
                    epoch_loss += loss.item()
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                    })
                    
                    # Check B weight updates every 5 batches
                    if batch_idx % 5 == 0 and batch_idx > 0:
                        changes_detected, total_change = self.check_b_weight_updates()
                        if changes_detected:
                            logger.info(f"✅ LoRA B weights updating! Total change: {total_change:.6f}")
                        
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Check parameter changes after epoch
            logger.info(f"\n🔍 Epoch {epoch + 1} B weight analysis:")
            changes_detected, total_change = self.check_b_weight_updates()
            
            if changes_detected:
                logger.info(f"✅ LoRA B weights are updating! Total change: {total_change:.6f}")
            else:
                logger.warning(f"⚠️  LoRA B weights may not be updating (change: {total_change:.8f})")
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"📊 Epoch {epoch + 1} complete - Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:  # Save every epoch
                self.save_checkpoint(epoch + 1)
        
        logger.info("🎉 Enhanced LoRA training complete!")
    
    def save_checkpoint(self, epoch):
        """Save LoRA checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / f"enhanced_flux_lora_epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving enhanced LoRA checkpoint for epoch {epoch}...")
        
        # Save PEFT checkpoint
        self.pipe.transformer.save_pretrained(checkpoint_dir)
        
        logger.info(f"✅ Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced FLUX LoRA Training")
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dataset_path", default="./dataset/anddrrew")
    parser.add_argument("--output_dir", default="./models/anddrrew_enhanced_flux_lora")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    
    trainer = EnhancedFluxLoRATrainer(args)
    trainer.load_pipeline()
    trainer.load_dataset()
    trainer.train()

if __name__ == "__main__":
    main()

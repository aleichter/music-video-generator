#!/usr/bin/env python3

"""
DEBUGGED FLUX LoRA Trainer - Addresses zero LoRA B weights issue
Key fixes:
1. Explicit parameter verification and gradient checking
2. Proper LoRA initialization validation
3. Enhanced loss computation and backpropagation
4. Detailed debugging output for parameter updates
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
                    else:
                        logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"Loaded {len(data)} samples from dataset")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image = Image.open(item['image_path']).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return {
            'image': image_tensor,
            'caption': item['caption']
        }

class DebuggedFluxLoRATrainer:
    """Debugged FLUX LoRA trainer that ensures LoRA B weights are updated"""
    
    def __init__(self, args):
        self.args = args
        self.pipe = None
        self.model = None
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("🚀 Initializing DEBUGGED FLUX LoRA Trainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Args: {vars(args)}")
        
    def load_pipeline(self):
        """Load FLUX pipeline"""
        logger.info("📥 Loading FLUX pipeline...")
        
        try:
            self.pipe = FluxPipeline.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            ).to(self.device)
            
            logger.info("✅ Pipeline loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load pipeline: {e}")
            raise
    
    def verify_lora_parameters(self):
        """Verify LoRA parameters are properly initialized and trainable"""
        logger.info("🔍 Verifying LoRA parameters...")
        
        lora_A_params = {}
        lora_B_params = {}
        
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                lora_A_params[name] = param
            elif 'lora_B' in name:
                lora_B_params[name] = param
        
        logger.info(f"Found {len(lora_A_params)} LoRA A parameters")
        logger.info(f"Found {len(lora_B_params)} LoRA B parameters")
        
        # Check initialization values
        for name, param in list(lora_A_params.items())[:3]:  # Check first 3
            logger.info(f"LoRA A {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, requires_grad={param.requires_grad}")
            
        for name, param in list(lora_B_params.items())[:3]:  # Check first 3
            logger.info(f"LoRA B {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, requires_grad={param.requires_grad}")
            
        # Verify B parameters are not all zeros initially
        all_zero_B = True
        for name, param in lora_B_params.items():
            if param.data.abs().sum() > 1e-8:
                all_zero_B = False
                break
        
        if all_zero_B:
            logger.info("✅ LoRA B parameters initialized to zero - this is correct for LoRA")
        else:
            logger.warning("⚠️  LoRA B parameters have non-zero initialization")
            
        return lora_A_params, lora_B_params
    
    def setup_lora(self):
        """Setup LoRA with verified initialization"""
        logger.info("🎯 Setting up LoRA...")
        
        # Target only key attention modules for stability
        target_modules = []
        
        # Focus on transformer blocks
        for i in range(19):  
            for suffix in ['to_q', 'to_k', 'to_v', 'to_out.0']:
                target_modules.append(f"transformer_blocks.{i}.attn.{suffix}")
                
        for i in range(38):  
            for suffix in ['to_q', 'to_k', 'to_v']:
                target_modules.append(f"single_transformer_blocks.{i}.attn.{suffix}")
        
        logger.info(f"🎯 Targeting {len(target_modules)} modules")
        
        # Verify modules exist
        existing_modules = []
        for name, module in self.pipe.transformer.named_modules():
            if name in target_modules:
                existing_modules.append(name)
        
        logger.info(f"✅ Found {len(existing_modules)} existing target modules")
        
        # Create LoRA config with explicit initialization
        lora_config = LoraConfig(
            r=16,  # Increased rank for stronger effect
            lora_alpha=32,  # Higher alpha for stronger effect
            target_modules=existing_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            init_lora_weights=True,  # Ensure proper initialization
        )
        
        logger.info(f"📊 LoRA Config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        # Apply LoRA
        self.model = get_peft_model(self.pipe.transformer, lora_config)
        
        # Verify parameters after LoRA application
        self.lora_A_params, self.lora_B_params = self.verify_lora_parameters()
        
        # Replace transformer in pipeline
        self.pipe.transformer = self.model
        
        logger.info("✅ LoRA setup complete!")
    
    def load_dataset(self):
        """Load training dataset"""
        logger.info("📂 Loading dataset...")
        
        self.dataset = SimpleFluxDataset(
            self.args.dataset_path,
            resolution=self.args.resolution
        )
        
        logger.info(f"✅ Dataset loaded: {len(self.dataset)} samples")
    
    def check_gradients(self, step):
        """Check if gradients are flowing to LoRA parameters"""
        if step % 5 == 0:  # Check every 5 steps
            a_has_grad = 0
            b_has_grad = 0
            a_grad_sum = 0.0
            b_grad_sum = 0.0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if 'lora_A' in name:
                        a_has_grad += 1
                        a_grad_sum += param.grad.abs().sum().item()
                    elif 'lora_B' in name:
                        b_has_grad += 1
                        b_grad_sum += param.grad.abs().sum().item()
            
            logger.info(f"Step {step}: Gradients - A: {a_has_grad} modules ({a_grad_sum:.6f}), B: {b_has_grad} modules ({b_grad_sum:.6f})")
    
    def train(self):
        """Main training loop with enhanced debugging"""
        logger.info("🎓 Starting training...")
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Setup optimizer with higher learning rate
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Store initial parameter values for comparison
        initial_B_values = {}
        for name, param in self.lora_B_params.items():
            initial_B_values[name] = param.data.clone()
        
        self.model.train()
        step = 0
        
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0
            
            logger.info(f"\n📚 Epoch {epoch + 1}/{self.args.num_epochs}")
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                step += 1
                
                try:
                    # Move batch to device
                    images = batch['image'].to(self.device, dtype=torch.bfloat16)
                    captions = batch['caption']
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
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
                    noise = torch.randn_like(latents)
                    timesteps = torch.rand((latents.shape[0],), device=self.device)
                    
                    # FLUX flow matching
                    noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise
                    timesteps_scaled = timesteps * 1000.0
                    
                    guidance = torch.full((latents.shape[0],), 3.5, device=self.device, dtype=torch.bfloat16)
                    
                    batch_size, channels, height, width = latents.shape
                    img_ids = torch.zeros((batch_size, height * width, 3), device=self.device, dtype=torch.bfloat16)
                    
                    # Forward pass
                    model_pred = self.model(
                        hidden_states=noisy_latents,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timesteps_scaled.long(),
                        img_ids=img_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        return_dict=False,
                    )[0]
                    
                    # Compute loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Check gradients
                    self.check_gradients(step)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Update parameters
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_idx}: {e}")
                    continue
            
            # Check parameter changes after epoch
            logger.info(f"\n🔍 Checking parameter changes after epoch {epoch + 1}:")
            changes_detected = False
            
            for name, param in list(self.lora_B_params.items())[:5]:  # Check first 5
                initial = initial_B_values[name]
                current = param.data
                change = (current - initial).abs().sum().item()
                
                logger.info(f"LoRA B {name}: change = {change:.8f}")
                if change > 1e-8:
                    changes_detected = True
            
            if changes_detected:
                logger.info("✅ LoRA B parameters are being updated!")
            else:
                logger.warning("❌ LoRA B parameters NOT updating - major issue!")
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"📊 Epoch {epoch + 1} complete - Average loss: {avg_loss:.4f}")
            
            # Save checkpoint every epoch for debugging
            self.save_checkpoint(epoch + 1)
        
        logger.info("🎉 Training complete!")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / f"debugged_flux_lora_epoch_{epoch}_peft"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving checkpoint for epoch {epoch}...")
        
        self.model.save_pretrained(checkpoint_dir)
        logger.info(f"✅ Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Debugged FLUX LoRA Training")
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dataset_path", default="./dataset/anddrrew")
    parser.add_argument("--output_dir", default="./models/anddrrew_debugged_flux_lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # Higher learning rate
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = DebuggedFluxLoRATrainer(args)
    trainer.load_pipeline()
    trainer.setup_lora()
    trainer.load_dataset()
    trainer.train()

if __name__ == "__main__":
    main()

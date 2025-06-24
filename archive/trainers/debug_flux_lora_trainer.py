#!/usr/bin/env python3

"""
Debug FLUX LoRA Trainer - Enhanced PEFT approach with detailed gradient monitoring
"""

import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import os

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
        captions_file = self.dataset_path / "captions.txt"
        
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
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = Image.open(item['image_path']).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        
        # Convert to tensor in [0, 1] range, then to [-1, 1] for VAE
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return {
            'image': image_tensor,
            'caption': item['caption']
        }

class DebugFluxLoRATrainer:
    """Debug FLUX LoRA trainer with detailed monitoring"""
    
    def __init__(self, args):
        self.args = args
        self.pipe = None
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("🚀 Initializing Debug FLUX LoRA Trainer")
        
    def load_pipeline(self):
        """Load FLUX pipeline"""
        logger.info("📥 Loading FLUX pipeline...")
        
        self.pipe = FluxPipeline.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(self.device)
        
        logger.info("✅ Pipeline loaded successfully!")
    
    def apply_lora(self):
        """Apply LoRA using PEFT"""
        logger.info("🎯 Applying LoRA to transformer...")
        
        # Define target modules - focusing on attention layers
        target_modules = []
        
        # Add attention modules from double transformer blocks
        for i in range(19):
            target_modules.extend([
                f"transformer_blocks.{i}.attn.to_q",
                f"transformer_blocks.{i}.attn.to_k", 
                f"transformer_blocks.{i}.attn.to_v",
                f"transformer_blocks.{i}.attn.to_out.0"
            ])
        
        # Add attention modules from single transformer blocks  
        for i in range(38):
            target_modules.extend([
                f"single_transformer_blocks.{i}.attn.to_q",
                f"single_transformer_blocks.{i}.attn.to_k",
                f"single_transformer_blocks.{i}.attn.to_v"
            ])
        
        logger.info(f"📊 Targeting {len(target_modules)} modules")
        
        # Create LoRA config with more aggressive settings
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,  # No dropout initially
            bias="none",
            task_type=None,  # Custom task
            init_lora_weights=True,  # Ensure proper initialization
        )
        
        # Apply LoRA
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)
        
        # Convert LoRA parameters to bfloat16 to match base model
        for name, param in self.pipe.transformer.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.data = param.data.to(torch.bfloat16)
        
        # Force enable training mode
        self.pipe.transformer.train()
        
        # Print model info
        self.pipe.transformer.print_trainable_parameters()
        
        logger.info("✅ LoRA applied successfully!")
    
    def verify_lora_parameters(self):
        """Verify LoRA parameters are set up correctly"""
        logger.info("🔍 Verifying LoRA parameters...")
        
        lora_a_params = []
        lora_b_params = []
        
        for name, param in self.pipe.transformer.named_parameters():
            if 'lora_A' in name and param.requires_grad:
                lora_a_params.append((name, param))
            elif 'lora_B' in name and param.requires_grad:
                lora_b_params.append((name, param))
        
        logger.info(f"📊 Found {len(lora_a_params)} LoRA A parameters")
        logger.info(f"📊 Found {len(lora_b_params)} LoRA B parameters")
        
        # Check first few B parameters
        for i, (name, param) in enumerate(lora_b_params[:3]):
            logger.info(f"Layer {i}: B {name}")
            logger.info(f"  Shape: {param.shape}, Dtype: {param.dtype}")
            logger.info(f"  Mean: {param.mean():.6f}, Std: {param.std():.6f}")
            logger.info(f"  Requires grad: {param.requires_grad}")
            logger.info(f"  Device: {param.device}")
            
        # Store initial B weights for monitoring
        self.initial_B_weights = []
        for _, param in lora_b_params:
            self.initial_B_weights.append(param.data.clone())
            
        return lora_b_params
    
    def load_dataset(self):
        """Load training dataset"""
        logger.info("📂 Loading dataset...")
        
        self.dataset = SimpleFluxDataset(
            self.args.dataset_path,
            resolution=self.args.resolution
        )
        
        logger.info(f"✅ Dataset loaded: {len(self.dataset)} samples")
    
    def train(self):
        """Main training loop with explicit gradient monitoring"""
        logger.info("🎓 Starting debug LoRA training...")
        
        # Get LoRA B parameters for monitoring
        lora_b_params = self.verify_lora_parameters()
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Setup optimizer - only trainable parameters
        trainable_params = [p for p in self.pipe.transformer.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        logger.info(f"📊 Optimizing {len(trainable_params)} parameters")
        
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
                    noise = torch.randn_like(latents, dtype=torch.bfloat16)
                    timesteps = torch.rand((latents.shape[0],), device=self.device, dtype=torch.bfloat16)
                    
                    # FLUX flow matching
                    noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise
                    timesteps_scaled = (timesteps * 1000.0).to(torch.long)
                    
                    guidance = torch.full((latents.shape[0],), 3.5, device=self.device, dtype=torch.bfloat16)
                    
                    batch_size, channels, height, width = latents.shape
                    img_ids = torch.zeros((height * width, 3), device=self.device, dtype=torch.bfloat16)
                    
                    # Forward pass through transformer with LoRA
                    # Ensure transformer is in training mode
                    self.pipe.transformer.train()
                    
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
                    
                    # Ensure loss requires gradients
                    if not loss.requires_grad:
                        logger.error(f"❌ Loss does not require gradients!")
                        continue
                        
                    # Backward pass
                    loss.backward()
                    
                    # Check gradients in detail
                    if batch_idx == 0:  # Detailed check on first batch
                        grad_info = self.check_gradients_detailed()
                        if grad_info['has_lora_b_grad']:
                            logger.info(f"✅ LoRA B gradients detected: {grad_info['lora_b_grad_sum']:.6f}")
                        else:
                            logger.warning(f"⚠️  No LoRA B gradients detected!")
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    
                    # Update parameters
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'grad_norm': f'{grad_norm:.4f}',
                        'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                    })
                    
                    # Check parameter updates every few batches
                    if batch_idx % 3 == 0 and batch_idx > 0:
                        changes_detected = self.check_parameter_updates(lora_b_params)
                        if changes_detected:
                            logger.info(f"✅ LoRA B weights updating at batch {batch_idx}")
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Check parameter changes after epoch
            logger.info(f"\n🔍 Checking LoRA B weight changes after epoch {epoch + 1}:")
            changes_detected = self.check_parameter_updates(lora_b_params)
            
            if changes_detected:
                logger.info("✅ LoRA B weights are updating!")
            else:
                logger.warning("⚠️  LoRA B weights may not be updating")
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"📊 Epoch {epoch + 1} complete - Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self.save_checkpoint(epoch + 1)
        
        logger.info("🎉 Debug LoRA training complete!")
    
    def check_gradients_detailed(self):
        """Check gradients in detail"""
        has_grad = False
        has_lora_b_grad = False
        lora_b_grad_sum = 0.0
        
        for name, param in self.pipe.transformer.named_parameters():
            if param.grad is not None:
                grad_sum = param.grad.abs().sum().item()
                if grad_sum > 0:
                    has_grad = True
                    if 'lora_B' in name:
                        has_lora_b_grad = True
                        lora_b_grad_sum += grad_sum
                        logger.debug(f"  B gradient {name}: {grad_sum:.6f}")
        
        return {
            'has_grad': has_grad,
            'has_lora_b_grad': has_lora_b_grad, 
            'lora_b_grad_sum': lora_b_grad_sum
        }
    
    def check_parameter_updates(self, lora_b_params):
        """Check if LoRA B parameters have been updated"""
        changes_detected = False
        
        for i, ((name, param), initial) in enumerate(zip(lora_b_params[:3], self.initial_B_weights[:3])):
            change = (param.data - initial).abs().sum().item()
            logger.info(f"Layer {i}: B weight change = {change:.8f}")
            if change > 1e-6:
                changes_detected = True
                
        return changes_detected
    
    def save_checkpoint(self, epoch):
        """Save LoRA checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / f"debug_flux_lora_epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving debug LoRA checkpoint for epoch {epoch}...")
        
        # Save PEFT checkpoint
        self.pipe.transformer.save_pretrained(checkpoint_dir)
        
        logger.info(f"✅ Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Debug FLUX LoRA Training")
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dataset_path", default="./dataset/anddrrew")
    parser.add_argument("--output_dir", default="./models/anddrrew_debug_flux_lora")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    
    trainer = DebugFluxLoRATrainer(args)
    trainer.load_pipeline()
    trainer.apply_lora()
    trainer.load_dataset()
    trainer.train()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Native FLUX LoRA Implementation - bypasses PEFT compatibility issues
This manually implements LoRA layers compatible with FLUX transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
import gc
from diffusers import FluxPipeline
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoRALinear(nn.Module):
    """Native LoRA implementation for Linear layers"""
    
    def __init__(self, original_layer, rank=16, alpha=32, dropout=0.1, device=None, dtype=None):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Use same device and dtype as original layer
        if device is None:
            device = next(original_layer.parameters()).device
        if dtype is None:
            dtype = next(original_layer.parameters()).dtype
        
        # LoRA matrices with correct dtype
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)  # B starts at zero (standard)
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Ensure input has the correct dtype
        x = x.to(dtype=self.lora_A.weight.dtype)
        
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA output - ensure all operations use consistent dtype
        try:
            lora_output = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
            return original_output + lora_output
        except RuntimeError as e:
            # If there's a dimension mismatch, fall back to original layer only
            print(f"LoRA dimension mismatch in layer: {e}")
            print(f"Input shape: {x.shape}, Original output: {original_output.shape}")
            print(f"LoRA A: {self.lora_A.weight.shape}, LoRA B: {self.lora_B.weight.shape}")
            return original_output

class SimpleFluxDataset(Dataset):
    """Simple dataset for FLUX LoRA training"""
    
    def __init__(self, dataset_path, resolution=512):
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
        
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return {
            'image': image_tensor,
            'caption': item['caption']
        }

class NativeFluxLoRATrainer:
    """Native FLUX LoRA trainer that manually replaces attention layers"""
    
    def __init__(self, args):
        self.args = args
        self.pipe = None
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_layers = []
        
        logger.info("🚀 Initializing Native FLUX LoRA Trainer")
        
    def load_pipeline(self):
        """Load FLUX pipeline"""
        logger.info("📥 Loading FLUX pipeline...")
        
        self.pipe = FluxPipeline.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(self.device)
        
        logger.info("✅ Pipeline loaded successfully!")
    
    def apply_lora_to_attention(self):
        """Manually replace attention layers with LoRA versions"""
        logger.info("🎯 Applying LoRA to attention layers...")
        
        replaced_count = 0
        
        # Replace in transformer_blocks (double blocks)
        for i in range(19):
            block = self.pipe.transformer.transformer_blocks[i]
            attn = block.attn
            
            # Replace to_q, to_k, to_v, to_out[0]
            for name in ['to_q', 'to_k', 'to_v']:
                if hasattr(attn, name):
                    original_layer = getattr(attn, name)
                    lora_layer = LoRALinear(
                        original_layer, 
                        rank=self.args.lora_rank, 
                        alpha=self.args.lora_alpha,
                        device=self.device,
                        dtype=torch.bfloat16
                    )
                    setattr(attn, name, lora_layer)
                    self.lora_layers.append(lora_layer)
                    replaced_count += 1
            
            # Handle to_out (it's a Sequential with Linear at index 0)
            if hasattr(attn, 'to_out') and len(attn.to_out) > 0:
                original_layer = attn.to_out[0]
                lora_layer = LoRALinear(
                    original_layer, 
                    rank=self.args.lora_rank, 
                    alpha=self.args.lora_alpha,
                    device=self.device,
                    dtype=torch.bfloat16
                )
                attn.to_out[0] = lora_layer
                self.lora_layers.append(lora_layer)
                replaced_count += 1
        
        # Replace in single_transformer_blocks
        for i in range(38):
            block = self.pipe.transformer.single_transformer_blocks[i]
            attn = block.attn
            
            for name in ['to_q', 'to_k', 'to_v']:
                if hasattr(attn, name):
                    original_layer = getattr(attn, name)
                    lora_layer = LoRALinear(
                        original_layer, 
                        rank=self.args.lora_rank, 
                        alpha=self.args.lora_alpha,
                        device=self.device,
                        dtype=torch.bfloat16
                    )
                    setattr(attn, name, lora_layer)
                    self.lora_layers.append(lora_layer)
                    replaced_count += 1
        
        logger.info(f"✅ Replaced {replaced_count} attention layers with LoRA")
        logger.info(f"📊 Total LoRA layers: {len(self.lora_layers)}")
        
        # Count trainable parameters
        total_params = sum(p.numel() for layer in self.lora_layers for p in layer.parameters())
        logger.info(f"📊 Trainable LoRA parameters: {total_params:,}")
    
    def verify_lora_parameters(self):
        """Verify LoRA parameters are set up correctly"""
        logger.info("🔍 Verifying LoRA parameters...")
        
        a_params = 0
        b_params = 0
        
        for i, layer in enumerate(self.lora_layers[:3]):  # Check first 3
            a_weight = layer.lora_A.weight
            b_weight = layer.lora_B.weight
            
            logger.info(f"Layer {i}: A mean={a_weight.mean():.6f}, std={a_weight.std():.6f}")
            logger.info(f"Layer {i}: B mean={b_weight.mean():.6f}, std={b_weight.std():.6f}")
            
            a_params += a_weight.numel()
            b_params += b_weight.numel()
        
        logger.info(f"A parameters: {a_params}, B parameters: {b_params}")
    
    def load_dataset(self):
        """Load training dataset"""
        logger.info("📂 Loading dataset...")
        
        self.dataset = SimpleFluxDataset(
            self.args.dataset_path,
            resolution=self.args.resolution
        )
        
        logger.info(f"✅ Dataset loaded: {len(self.dataset)} samples")
    
    def train(self):
        """Main training loop"""
        logger.info("🎓 Starting native LoRA training...")
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Setup optimizer - only LoRA parameters
        lora_params = []
        for layer in self.lora_layers:
            lora_params.extend(layer.parameters())
        
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.args.learning_rate,
            weight_decay=0.01
        )
        
        logger.info(f"📊 Optimizing {len(lora_params)} LoRA parameters")
        
        # Store initial B weights for monitoring
        initial_B_weights = []
        for layer in self.lora_layers:
            initial_B_weights.append(layer.lora_B.weight.data.clone())
        
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
                    # Fix img_ids to be 2D as expected
                    img_ids = torch.zeros((height * width, 3), device=self.device, dtype=torch.bfloat16)
                    
                    # Forward pass through transformer with LoRA
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
                    
                    # Compute loss (convert to float32 for numerical stability)
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    
                    # Update
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_idx}: {e}")
                    continue
            
            # Check parameter changes
            logger.info(f"\n🔍 Checking LoRA B weight changes after epoch {epoch + 1}:")
            changes_detected = False
            
            for i, layer in enumerate(self.lora_layers[:3]):  # Check first 3
                current_B = layer.lora_B.weight.data
                initial_B = initial_B_weights[i]
                change = (current_B - initial_B).abs().sum().item()
                
                logger.info(f"Layer {i}: B weight change = {change:.8f}")
                if change > 1e-6:
                    changes_detected = True
            
            if changes_detected:
                logger.info("✅ LoRA B weights are updating!")
            else:
                logger.warning("⚠️  LoRA B weights may not be updating")
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"📊 Epoch {epoch + 1} complete - Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(epoch + 1)
        
        logger.info("🎉 Native LoRA training complete!")
    
    def save_checkpoint(self, epoch):
        """Save LoRA weights"""
        checkpoint_dir = Path(self.args.output_dir) / f"native_flux_lora_epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving native LoRA checkpoint for epoch {epoch}...")
        
        # Save LoRA weights
        lora_state_dict = {}
        for i, layer in enumerate(self.lora_layers):
            lora_state_dict[f'lora_layer_{i}_A'] = layer.lora_A.weight
            lora_state_dict[f'lora_layer_{i}_B'] = layer.lora_B.weight
        
        torch.save(lora_state_dict, checkpoint_dir / "lora_weights.pt")
        
        # Save metadata
        metadata = {
            'rank': self.args.lora_rank,
            'alpha': self.args.lora_alpha,
            'num_layers': len(self.lora_layers),
            'epoch': epoch
        }
        torch.save(metadata, checkpoint_dir / "metadata.pt")
        
        logger.info(f"✅ Checkpoint saved to {checkpoint_dir}")

def main():
    parser = argparse.ArgumentParser(description="Native FLUX LoRA Training")
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dataset_path", default="./dataset/anddrrew")
    parser.add_argument("--output_dir", default="./models/anddrrew_native_flux_lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    
    trainer = NativeFluxLoRATrainer(args)
    trainer.load_pipeline()
    trainer.apply_lora_to_attention()
    trainer.verify_lora_parameters()
    trainer.load_dataset()
    trainer.train()

if __name__ == "__main__":
    main()

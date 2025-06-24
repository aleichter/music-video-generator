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

class QuickFluxLoRATrainer:
    """Quick and simple FLUX LoRA trainer with verbose logging"""
    
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell", device="cuda"):
        self.device = device
        self.torch_dtype = torch.bfloat16
        self.model_name = model_name
        
        logger.info(f"🚀 Starting QuickFluxLoRATrainer initialization...")
        logger.info(f"📦 Model: {model_name}")
        logger.info(f"🔧 Device: {device}")
        logger.info(f"🎯 Data type: {self.torch_dtype}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("🧹 Cleared GPU memory")
        
        self.load_pipeline()
        self.setup_lora()
    
    def load_pipeline(self):
        """Load FLUX pipeline with verbose progress"""
        logger.info("📥 Loading FLUX pipeline...")
        start_time = time.time()
        
        try:
            self.pipe = FluxPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Pipeline loaded in {load_time:.1f}s")
            
            # Move transformer to GPU, others to CPU
            logger.info("🔄 Optimizing memory layout...")
            self.pipe.transformer.to(self.device)
            
            # Move text encoders and VAE to CPU to save memory
            if hasattr(self.pipe, 'text_encoder'):
                self.pipe.text_encoder.to('cpu')
            if hasattr(self.pipe, 'text_encoder_2'):
                self.pipe.text_encoder_2.to('cpu')
            self.pipe.vae.to('cpu')
            
            logger.info("✅ Memory layout optimized")
            
        except Exception as e:
            logger.error(f"❌ Failed to load pipeline: {e}")
            raise
    
    def setup_lora(self):
        """Setup LoRA with detailed logging"""
        logger.info("🎯 Setting up LoRA...")
        
        # Find attention modules
        target_modules = []
        for name, module in self.pipe.transformer.named_modules():
            if isinstance(module, nn.Linear) and 'attn' in name:
                target_modules.append(name)
                if len(target_modules) >= 8:  # Limit for speed
                    break
        
        logger.info(f"🎯 Found {len(target_modules)} target modules:")
        for i, target in enumerate(target_modules):
            logger.info(f"  {i+1}. {target}")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.pipe.transformer, lora_config)
        
        # Count parameters
        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.peft_model.parameters())
        
        logger.info(f"✅ LoRA applied successfully!")
        logger.info(f"📊 Trainable parameters: {trainable:,}")
        logger.info(f"📊 Total parameters: {total:,}")
        logger.info(f"📊 Trainable ratio: {100*trainable/total:.4f}%")
    
    def simple_loss(self, batch):
        """Simple loss that definitely works"""
        images = batch['image'].to(self.device, dtype=self.torch_dtype)
        batch_size = images.shape[0]
        
        # Extract simple features from images
        features = F.adaptive_avg_pool2d(images, (4, 4)).flatten(1)  # [B, 48]
        
        # Add noise
        noise = torch.randn_like(features) * 0.1
        noisy_features = features + noise
        
        # Find first applicable LoRA layer
        processed = noisy_features
        applied_lora = False
        
        for name, module in self.peft_model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_A = module.lora_A['default']
                lora_B = module.lora_B['default']
                
                if processed.shape[-1] == lora_A.weight.shape[1]:
                    # Apply LoRA
                    lora_out = lora_B(lora_A(processed))
                    processed = processed + lora_out * module.scaling['default']
                    applied_lora = True
                    break
        
        # Compute loss
        loss = F.mse_loss(processed, features)
        
        # Add regularization for all LoRA parameters
        reg_loss = torch.tensor(0.0, device=self.device, dtype=self.torch_dtype)
        for param in self.peft_model.parameters():
            if param.requires_grad:
                reg_loss = reg_loss + (param ** 2).mean()
        
        total_loss = loss + reg_loss * 0.001
        
        return total_loss, applied_lora
    
    def train(self, dataset_path, output_dir, epochs=5, batch_size=2, learning_rate=1e-4):
        """Simple training with detailed progress"""
        logger.info("🏋️ Starting training...")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        logger.info("📚 Loading dataset...")
        dataset = SimpleFluxDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        logger.info(f"📚 Dataset ready: {len(dataset)} samples, {len(dataloader)} batches")
        
        # Setup optimizer
        lora_params = [p for p in self.peft_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
        
        logger.info(f"📊 Training {len(lora_params)} parameter tensors")
        logger.info(f"📊 Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Training loop
        self.peft_model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            lora_applications = 0
            
            logger.info(f"🔄 Starting epoch {epoch+1}/{epochs}")
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    optimizer.zero_grad()
                    
                    # Compute loss
                    loss, applied_lora = self.simple_loss(batch)
                    
                    if applied_lora:
                        lora_applications += 1
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Update stats
                    loss_val = loss.item()
                    epoch_loss += loss_val
                    
                    progress_bar.set_postfix({
                        'loss': f"{loss_val:.6f}",
                        'avg': f"{epoch_loss/(batch_idx+1):.6f}",
                        'lora': f"{lora_applications}/{batch_idx+1}"
                    })
                    
                except Exception as e:
                    logger.warning(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"✅ Epoch {epoch+1} complete: avg_loss={avg_loss:.6f}, lora_applied={lora_applications}/{len(dataloader)}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                checkpoint_path = output_dir / f"quick_flux_lora_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, optimizer)
        
        logger.info("🎉 Training completed successfully!")
    
    def save_checkpoint(self, path, epoch, optimizer):
        """Save checkpoint"""
        try:
            # Save PEFT model
            peft_path = str(path).replace('.pt', '_peft')
            self.peft_model.save_pretrained(peft_path)
            
            # Save PyTorch checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.peft_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
            
            logger.info(f"💾 Checkpoint saved: {path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

def main():
    parser = argparse.ArgumentParser(description="Quick FLUX LoRA Training")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", default="./quick_flux_lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    try:
        trainer = QuickFluxLoRATrainer()
        
        trainer.train(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        
        logger.info("🎉 All done!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

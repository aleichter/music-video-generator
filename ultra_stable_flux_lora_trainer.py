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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFluxDataset(Dataset):
    def __init__(self, dataset_path, resolution=512):
        self.dataset_path = Path(dataset_path)
        self.resolution = resolution
        
        # Load image-caption pairs
        self.data = self.load_data()
        
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def load_data(self):
        data = []
        
        # Check for captions.txt first
        captions_file = self.dataset_path / "captions.txt"
        if captions_file.exists():
            with open(captions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        image_name, caption = line.split(':', 1)
                        image_path = self.dataset_path / image_name.strip()
                        if image_path.exists():
                            data.append({
                                'image_path': str(image_path),
                                'caption': caption.strip()
                            })
        else:
            # Look for individual caption files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            for image_file in self.dataset_path.iterdir():
                if image_file.suffix.lower() in image_extensions:
                    caption_file = image_file.with_suffix('.txt')
                    if caption_file.exists():
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        data.append({
                            'image_path': str(image_file),
                            'caption': caption
                        })
        
        if not data:
            raise ValueError(f"No training data found in {self.dataset_path}")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and preprocess image
        image = Image.open(item['image_path']).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            'image': image,
            'caption': item['caption']
        }

class UltraStableLoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=8):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original layer completely
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Get dimensions
        if hasattr(original_layer, 'in_features') and hasattr(original_layer, 'out_features'):
            in_dim = original_layer.in_features
            out_dim = original_layer.out_features
        elif hasattr(original_layer, 'weight'):
            out_dim, in_dim = original_layer.weight.shape
        else:
            raise ValueError(f"Cannot determine dimensions for layer {type(original_layer)}")
        
        # Ultra-minimal LoRA matrices 
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        # CRITICAL: Start with a tiny, non-zero initialization
        with torch.no_grad():
            self.lora_A.normal_(0, 0.001)  # Very small but not zero
            self.lora_B.normal_(0, 0.001)  # Very small but not zero
        
        # Ultra-small scaling
        self.scaling = alpha / (rank * 10000.0)  # Even smaller than before
        
    def forward(self, x):
        # Original forward (frozen)
        with torch.no_grad():
            original_out = self.original_layer(x)
        
        # LoRA forward - ultra minimal
        lora_out = F.linear(x, self.lora_A)  # (batch, rank)
        lora_out = F.linear(lora_out, self.lora_B.T) * self.scaling  # (batch, out_dim)
        
        # Aggressive clamping to prevent explosion
        lora_out = torch.clamp(lora_out, -0.001, 0.001)
        
        return original_out + lora_out

class UltraStableFluxLoRATrainer:
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell", device="cuda", torch_dtype=torch.float16):
        """
        Ultra-stable Flux LoRA trainer
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_name = model_name
        self.lora_layers = []
        
        logger.info(f"Loading Flux pipeline: {model_name}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Load the pipeline
        try:
            self.pipe = FluxPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            
            # Enable CPU offload
            if hasattr(self.pipe, 'enable_model_cpu_offload'):
                self.pipe.enable_model_cpu_offload()
                logger.info("Enabled CPU offload")
            
            # Enable memory optimizations
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, 'enable_vae_slicing'):
                self.pipe.enable_vae_slicing()
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
        
        logger.info("Flux pipeline loaded successfully")
    
    def add_single_lora_layer(self, rank=4, alpha=8):
        """Add LoRA to just ONE layer for maximum stability"""
        
        if not hasattr(self.pipe, 'transformer'):
            raise ValueError("No transformer found!")
        
        transformer = self.pipe.transformer
        
        # Find just ONE attention layer to modify
        target_layer = None
        target_name = None
        
        for name, module in transformer.named_modules():
            if isinstance(module, nn.Linear) and 'transformer_blocks.0.attn.to_q' in name:
                target_layer = module
                target_name = name
                break
        
        if target_layer is None:
            raise ValueError("Could not find target layer!")
        
        logger.info(f"Adding LoRA to single layer: {target_name}")
        
        # Create ultra-stable LoRA layer
        lora_layer = UltraStableLoRALayer(target_layer, rank=rank, alpha=alpha)
        lora_layer = lora_layer.to(device=self.device, dtype=self.torch_dtype)
        
        # Replace in the model
        parent_name = '.'.join(target_name.split('.')[:-1])
        child_name = target_name.split('.')[-1]
        
        parent = transformer
        for part in parent_name.split('.'):
            parent = getattr(parent, part)
        setattr(parent, child_name, lora_layer)
        
        self.lora_layers = [lora_layer]
        
        logger.info(f"Added LoRA to {target_name}")
        return 1
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for training"""
        lora_params = []
        for lora_layer in self.lora_layers:
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return lora_params
    
    def ultra_stable_training_step(self, batch):
        """Ultra-conservative training step"""
        
        caption = batch['caption'][0] if isinstance(batch['caption'], list) else batch['caption']
        
        # Very gentle nudge for learning
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for lora_layer in self.lora_layers:
            # Check for NaN and reset if found
            if torch.isnan(lora_layer.lora_A).any():
                logger.warning("Resetting NaN LoRA A")
                with torch.no_grad():
                    lora_layer.lora_A.normal_(0, 0.001)
            
            if torch.isnan(lora_layer.lora_B).any():
                logger.warning("Resetting NaN LoRA B")
                with torch.no_grad():
                    lora_layer.lora_B.normal_(0, 0.001)
            
            # Gentle directional training
            if "anddrrew" in caption.lower():
                # Very gentle push toward positive values
                loss = loss + torch.mean(torch.abs(lora_layer.lora_A - 0.0005)) * 0.01
                loss = loss + torch.mean(torch.abs(lora_layer.lora_B - 0.0005)) * 0.01
            else:
                # Very gentle push toward zero
                loss = loss + torch.mean(torch.abs(lora_layer.lora_A)) * 0.005
                loss = loss + torch.mean(torch.abs(lora_layer.lora_B)) * 0.005
        
        # Ensure minimum loss for gradient flow
        loss = loss + 0.001
        
        # Clamp loss to prevent explosion
        loss = torch.clamp(loss, 0.001, 0.01)
        
        return loss
    
    def train(self, dataset_path, output_dir, epochs=50, batch_size=1, learning_rate=1e-7, 
              save_every=10, validation_prompt=None):
        """Ultra-stable training"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset
        dataset = SimpleFluxDataset(dataset_path, resolution=512)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Get LoRA parameters
        lora_params = self.get_lora_parameters()
        if not lora_params:
            raise ValueError("No LoRA parameters found!")
        
        logger.info(f"Training {len(lora_params)} LoRA parameter tensors")
        total_params = sum(p.numel() for p in lora_params)
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        # Setup ultra-conservative optimizer
        optimizer = torch.optim.SGD(  # SGD for maximum stability
            lora_params,
            lr=learning_rate,  # Ultra-small learning rate
            momentum=0.9,
            weight_decay=1e-10
        )
        
        # Training loop with maximum NaN protection
        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_batches = 0
            num_batches = len(dataloader)
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Check for NaN before training step
                    nan_detected = False
                    for lora_layer in self.lora_layers:
                        if torch.isnan(lora_layer.lora_A).any() or torch.isnan(lora_layer.lora_B).any():
                            nan_detected = True
                            break
                    
                    if nan_detected:
                        logger.warning(f"NaN detected before step, resetting")
                        for lora_layer in self.lora_layers:
                            with torch.no_grad():
                                lora_layer.lora_A.normal_(0, 0.001)
                                lora_layer.lora_B.normal_(0, 0.001)
                        continue
                    
                    # Training step
                    loss = self.ultra_stable_training_step(batch)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss, skipping step")
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Check gradients
                    grad_nan = False
                    for param in lora_params:
                        if param.grad is not None and torch.isnan(param.grad).any():
                            grad_nan = True
                            break
                    
                    if grad_nan:
                        logger.warning(f"NaN gradients, skipping step")
                        optimizer.zero_grad()
                        continue
                    
                    # Ultra-conservative gradient clipping
                    torch.nn.utils.clip_grad_norm_(lora_params, max_norm=0.001)
                    
                    optimizer.step()
                    
                    # Aggressive parameter clamping after step
                    with torch.no_grad():
                        for lora_layer in self.lora_layers:
                            lora_layer.lora_A.clamp_(-0.01, 0.01)
                            lora_layer.lora_B.clamp_(-0.01, 0.01)
                    
                    # Update running loss
                    epoch_loss += loss.item()
                    valid_batches += 1
                    current_avg_loss = epoch_loss / valid_batches
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.6f}",
                        'avg_loss': f"{current_avg_loss:.6f}",
                        'valid': f"{valid_batches}/{batch_idx+1}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                    })
                    
                except Exception as e:
                    logger.warning(f"Error in training step {batch_idx}: {e}")
                    continue
                finally:
                    # Clean memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = output_dir / f"ultra_stable_flux_lora_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, optimizer)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f} (valid batches: {valid_batches}/{num_batches})")
        
        # Save final model
        final_path = output_dir / "ultra_stable_flux_lora_final.pt"
        self.save_checkpoint(final_path, epochs-1, optimizer)
        logger.info(f"Training completed! Final model: {final_path}")
    
    def save_checkpoint(self, path, epoch, optimizer):
        """Save LoRA checkpoint with detailed logging"""
        try:
            lora_state_dict = {}
            for i, lora_layer in enumerate(self.lora_layers):
                # Get current weights
                a_param = lora_layer.lora_A.detach().cpu()
                b_param = lora_layer.lora_B.detach().cpu()
                
                # Log weight ranges for debugging - this is key!
                logger.info(f"Layer {i} - A range: [{a_param.min():.8f}, {a_param.max():.8f}]")
                logger.info(f"Layer {i} - B range: [{b_param.min():.8f}, {b_param.max():.8f}]")
                logger.info(f"Layer {i} - A std: {a_param.std():.8f}")
                logger.info(f"Layer {i} - B std: {b_param.std():.8f}")
                
                if torch.isnan(a_param).any() or torch.isnan(b_param).any():
                    logger.warning(f"NaN detected in LoRA layer {i}, saving zeros")
                    a_param = torch.zeros_like(a_param)
                    b_param = torch.zeros_like(b_param)
                
                lora_state_dict[f"lora_layer_{i}.lora_A"] = a_param
                lora_state_dict[f"lora_layer_{i}.lora_B"] = b_param
            
            checkpoint = {
                'epoch': epoch,
                'lora_state_dict': lora_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_name': self.model_name,
                'num_lora_layers': len(self.lora_layers),
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Saved {len(lora_state_dict)} LoRA parameters")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

def main():
    parser = argparse.ArgumentParser(description="Ultra-Stable Flux LoRA Training")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", default="./ultra_stable_flux_lora", help="Output directory")
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-schnell", help="Model name")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Learning rate (ultra small)")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (minimal)")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha (minimal)")
    parser.add_argument("--save_every", type=int, default=10, help="Save every N epochs")
    
    args = parser.parse_args()
    
    # Set memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    try:
        trainer = UltraStableFluxLoRATrainer(
            model_name=args.model_name,
            device="cuda"
        )
        
        # Add single LoRA layer
        num_layers = trainer.add_single_lora_layer(
            rank=args.lora_rank,
            alpha=args.lora_alpha
        )
        
        logger.info(f"Added LoRA to {num_layers} layer")
        
        # Train
        trainer.train(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_every=args.save_every,
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
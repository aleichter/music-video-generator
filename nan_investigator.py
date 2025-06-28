#!/usr/bin/env python3
"""
Investigate NaN sources in FLUX LoRA training
"""

import torch
import os
import sys
from pathlib import Path

# Add sd-scripts to path
sys.path.insert(0, '/workspace/music-video-generator/sd-scripts')

def test_flux_lora_network():
    """Test if the LoRA network itself produces NaNs"""
    print("🔬 Testing FLUX LoRA Network Initialization...")
    
    try:
        from networks.lora_flux import LoRANetwork
        
        # Create minimal network
        network = LoRANetwork(
            text_encoder=None,
            unet=None,
            multiplier=1.0,
            lora_dim=8,
            alpha=4,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            use_cp=False,
            conv_lora_dim=None,
            conv_alpha=None,
            block_dims=None,
            block_alphas=None,
            train_norm=False,
            varbose=False
        )
        
        print("✅ LoRA network created successfully")
        
        # Check if any weights are NaN immediately after initialization
        nan_found = False
        for name, param in network.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ NaN found in {name} after initialization!")
                nan_found = True
            else:
                print(f"✅ {name}: Mean = {param.mean().item():.6f}")
        
        if not nan_found:
            print("✅ All network weights initialized correctly")
        
        return network
        
    except Exception as e:
        print(f"❌ Error creating LoRA network: {e}")
        return None

def test_optimizer_compatibility():
    """Test if AdamW optimizer causes issues with bf16"""
    print("\n🔬 Testing Optimizer with bf16...")
    
    # Create simple tensor
    x = torch.randn(10, 10, dtype=torch.bfloat16, requires_grad=True)
    print(f"✅ Test tensor: Mean = {x.float().mean().item():.6f}")
    
    # Test AdamW
    optimizer = torch.optim.AdamW([x], lr=2e-5)
    print("✅ AdamW optimizer created")
    
    # Simulate training step
    loss = (x ** 2).sum()
    print(f"✅ Loss: {loss.item():.6f}")
    
    loss.backward()
    print(f"✅ Gradient: Mean = {x.grad.float().mean().item():.6f}")
    
    optimizer.step()
    print(f"✅ After step: Mean = {x.float().mean().item():.6f}")
    
    if torch.isnan(x).any():
        print("❌ NaN detected after optimizer step!")
        return False
    else:
        print("✅ Optimizer step successful")
        return True

def test_mixed_precision():
    """Test if mixed precision training causes NaNs"""
    print("\n🔬 Testing Mixed Precision...")
    
    # Test with GradScaler
    scaler = torch.cuda.amp.GradScaler()
    
    x = torch.randn(10, 10, dtype=torch.float32, requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW([x], lr=2e-5)
    
    print(f"✅ Initial: Mean = {x.mean().item():.6f}")
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = (x ** 2).sum()
        print(f"✅ Loss (autocast): {loss.item():.6f}")
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if torch.isnan(x).any():
        print("❌ NaN detected after mixed precision step!")
        return False
    else:
        print(f"✅ After mixed precision: Mean = {x.mean().item():.6f}")
        return True

def test_large_learning_rates():
    """Test if learning rate is too high"""
    print("\n🔬 Testing Learning Rate Sensitivity...")
    
    learning_rates = [1e-6, 1e-5, 2e-5, 1e-4, 1e-3]
    
    for lr in learning_rates:
        x = torch.randn(10, 10, dtype=torch.bfloat16, requires_grad=True)
        optimizer = torch.optim.AdamW([x], lr=lr)
        
        # Multiple steps to see if NaN develops
        for step in range(5):
            loss = (x ** 2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if torch.isnan(x).any():
                print(f"❌ NaN at LR {lr} after step {step+1}")
                break
        else:
            print(f"✅ LR {lr}: Stable after 5 steps, Mean = {x.float().mean().item():.6f}")

def main():
    print("🧪 FLUX LoRA NaN Investigation")
    print("=" * 50)
    
    # Test 1: Network initialization
    network = test_flux_lora_network()
    
    # Test 2: Optimizer compatibility
    test_optimizer_compatibility()
    
    # Test 3: Mixed precision
    if torch.cuda.is_available():
        test_mixed_precision()
    else:
        print("⚠️ CUDA not available, skipping mixed precision test")
    
    # Test 4: Learning rate sensitivity
    test_large_learning_rates()
    
    print("\n🎯 Investigation Complete!")
    print("Check the results above to identify potential NaN sources.")

if __name__ == "__main__":
    main()

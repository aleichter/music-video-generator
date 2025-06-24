#!/usr/bin/env python3

"""
Quick test to verify LoRA initialization and parameter updates
"""

import torch
from diffusers import FluxPipeline
from peft import LoraConfig, TaskType, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lora_initialization():
    """Test LoRA parameter initialization and update capability"""
    
    logger.info("🧪 Testing LoRA initialization...")
    
    # Load pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")
    
    # Setup LoRA on a few modules only for testing
    target_modules = [
        "single_transformer_blocks.0.attn.to_q",
        "single_transformer_blocks.0.attn.to_k", 
        "single_transformer_blocks.0.attn.to_v",
        "single_transformer_blocks.1.attn.to_q",
        "single_transformer_blocks.1.attn.to_k",
        "single_transformer_blocks.1.attn.to_v",
    ]
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        init_lora_weights=True,
    )
    
    # Apply LoRA
    model = get_peft_model(pipe.transformer, lora_config)
    model.train()
    
    # Check initial parameter values
    logger.info("📊 Initial LoRA parameter values:")
    lora_A_params = {}
    lora_B_params = {}
    
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            lora_A_params[name] = param
            logger.info(f"A {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, max_abs={param.data.abs().max():.6f}")
        elif 'lora_B' in name:
            lora_B_params[name] = param
            logger.info(f"B {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, max_abs={param.data.abs().max():.6f}")
    
    # Test gradient computation with dummy loss
    logger.info("\n🎯 Testing gradient computation...")
    
    # Create dummy input that matches FLUX transformer signature
    batch_size = 1
    seq_len = 64  # Small for testing
    hidden_dim = 3072
    
    # Dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(batch_size, 77, 4096, device="cuda", dtype=torch.bfloat16)
    pooled_projections = torch.randn(batch_size, 768, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([500], device="cuda", dtype=torch.long)
    img_ids = torch.zeros(batch_size, seq_len, 3, device="cuda", dtype=torch.bfloat16)
    txt_ids = torch.zeros(batch_size, 77, 3, device="cuda", dtype=torch.bfloat16)
    guidance = torch.tensor([3.5], device="cuda", dtype=torch.bfloat16)
    
    try:
        # Forward pass
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
        
        # Dummy loss
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        
        logger.info(f"Forward pass successful, loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        logger.info("\n📈 Gradient check:")
        a_with_grad = 0
        b_with_grad = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().sum().item()
                if 'lora_A' in name:
                    a_with_grad += 1
                    logger.info(f"A {name}: grad_norm={grad_norm:.8f}")
                elif 'lora_B' in name:
                    b_with_grad += 1
                    logger.info(f"B {name}: grad_norm={grad_norm:.8f}")
        
        logger.info(f"\nGradient summary: A params with grad: {a_with_grad}, B params with grad: {b_with_grad}")
        
        if b_with_grad == 0:
            logger.error("❌ NO GRADIENTS ON LoRA B PARAMETERS!")
            logger.error("This explains why they remain zero during training.")
        else:
            logger.info("✅ LoRA B parameters have gradients!")
        
        # Test parameter update
        logger.info("\n🔄 Testing parameter update...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Store initial values
        initial_B_values = {}
        for name, param in lora_B_params.items():
            initial_B_values[name] = param.data.clone()
        
        # Update
        optimizer.step()
        
        # Check changes
        changes_detected = False
        for name, param in lora_B_params.items():
            change = (param.data - initial_B_values[name]).abs().sum().item()
            logger.info(f"B {name}: parameter change = {change:.8f}")
            if change > 1e-8:
                changes_detected = True
        
        if changes_detected:
            logger.info("✅ LoRA B parameters updated successfully!")
        else:
            logger.error("❌ LoRA B parameters did NOT update!")
            
    except Exception as e:
        logger.error(f"❌ Error during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lora_initialization()

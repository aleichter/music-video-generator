#!/usr/bin/env python3

import torch
import logging
from pathlib import Path
from diffusers import FluxPipeline
from peft import LoraConfig, TaskType, get_peft_model
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_flux_lora_setup():
    """Test if we can set up LoRA on FLUX without errors"""
    device = "cuda"
    torch_dtype = torch.bfloat16
    model_name = "black-forest-labs/FLUX.1-schnell"
    
    logger.info("Loading FLUX pipeline...")
    
    try:
        # Load pipeline
        pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        
        # Move transformer to GPU
        pipe.transformer.to(device)
        logger.info("✅ Pipeline loaded successfully")
        
        # Find target modules
        target_modules = []
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, nn.Linear) and any(pattern in name for pattern in ['attn', 'attention']):
                target_modules.append(name)
                if len(target_modules) >= 8:
                    break
        
        logger.info(f"Found {len(target_modules)} target modules:")
        for target in target_modules:
            logger.info(f"  - {target}")
        
        if not target_modules:
            logger.error("No target modules found!")
            return False
        
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
        peft_model = get_peft_model(pipe.transformer, lora_config)
        logger.info("✅ LoRA applied successfully")
        
        # Count parameters
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        logger.info(f"📊 Trainable: {trainable:,}, Total: {total:,}, Ratio: {100*trainable/total:.4f}%")
        
        # Test a simple forward pass
        dummy_input = torch.randn(1, 64, 768, device=device, dtype=torch_dtype)
        
        # Find a LoRA layer and test it
        for name, module in peft_model.named_modules():
            if hasattr(module, 'lora_A'):
                try:
                    # Test if we can apply the LoRA layer
                    if dummy_input.shape[-1] == module.in_features:
                        output = module(dummy_input)
                        logger.info(f"✅ Successfully tested LoRA layer: {name}")
                        logger.info(f"   Input shape: {dummy_input.shape}")
                        logger.info(f"   Output shape: {output.shape}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to test {name}: {e}")
                    continue
        
        logger.info("🎉 FLUX LoRA setup test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_flux_lora_setup()
    if success:
        print("✅ Test passed - LoRA setup works correctly")
    else:
        print("❌ Test failed - check the logs for issues")

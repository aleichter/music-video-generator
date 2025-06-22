import torch
import argparse
from pathlib import Path
from diffusers import FluxPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_minimal_lora(pipe, lora_path, scaling=1.0):
    """Load minimal LoRA weights into Flux pipeline"""
    
    logger.info(f"Loading minimal LoRA from: {lora_path}")
    
    # Load checkpoint
    checkpoint = torch.load(lora_path, map_location='cpu', weights_only=False)
    lora_state_dict = checkpoint['lora_state_dict']
    
    logger.info(f"Found {len(lora_state_dict)} LoRA parameters")
    
    # Get the target layer (transformer_blocks.0.attn.to_q)
    target_layer = pipe.transformer.transformer_blocks[0].attn.to_q
    
    # Determine device and dtype from the target layer
    device = target_layer.weight.device
    dtype = target_layer.weight.dtype
    
    # Load LoRA weights
    lora_A = lora_state_dict['lora_layer_0.lora_A'].to(device=device, dtype=dtype)
    lora_B = lora_state_dict['lora_layer_0.lora_B'].to(device=device, dtype=dtype)
    
    logger.info(f"LoRA A shape: {lora_A.shape}")
    logger.info(f"LoRA B shape: {lora_B.shape}")
    logger.info(f"Target layer weight shape: {target_layer.weight.shape}")
    
    # Create LoRA weight delta
    lora_weight = (lora_B @ lora_A) * scaling
    
    logger.info(f"LoRA delta shape: {lora_weight.shape}")
    logger.info(f"LoRA delta range: [{lora_weight.min():.6f}, {lora_weight.max():.6f}]")
    
    # Apply LoRA to original weights
    with torch.no_grad():
        original_weight = target_layer.weight.data.clone()
        target_layer.weight.data = original_weight + lora_weight
    
    logger.info("✅ LoRA applied successfully!")
    
    return pipe

def main():
    parser = argparse.ArgumentParser(description="Test Minimal LoRA")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--prompt", default="anddrrew, portrait of a man", help="Generation prompt")
    parser.add_argument("--output", default="test_lora_output.png", help="Output image path")
    parser.add_argument("--scaling", type=float, default=1.0, help="LoRA scaling factor")
    parser.add_argument("--compare", action="store_true", help="Generate comparison (with/without LoRA)")
    
    args = parser.parse_args()
    
    # Load pipeline
    logger.info("Loading Flux pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    
    if args.compare:
        # Generate without LoRA first
        logger.info("Generating WITHOUT LoRA...")
        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                width=512,
                height=512,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=torch.Generator().manual_seed(42)
            )
            result.images[0].save("without_lora.png")
            logger.info("Saved: without_lora.png")
    
    # Load and apply LoRA
    pipe = load_minimal_lora(pipe, args.lora_path, scaling=args.scaling)
    
    # Generate with LoRA
    logger.info(f"Generating WITH LoRA (scaling={args.scaling})...")
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            width=512,
            height=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator().manual_seed(42)
        )
        result.images[0].save(args.output)
        logger.info(f"Saved: {args.output}")
    
    logger.info("✅ Test completed!")

if __name__ == "__main__":
    main()
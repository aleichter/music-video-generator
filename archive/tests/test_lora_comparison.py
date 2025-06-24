#!/usr/bin/env python3

import torch
import os
import numpy as np
from PIL import Image
from safetensors import safe_open
from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import gc

def load_flux_models():
    """Load all FLUX model components"""
    print("Loading FLUX models...")
    
    # Load transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Load text encoders
    text_encoder = CLIPTextModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder_2", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Load tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="tokenizer"
    )
    
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="tokenizer_2"
    )
    
    # Load VAE
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="vae",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="scheduler"
    )
    
    return transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, scheduler

def encode_prompt(prompt, tokenizer, text_encoder, tokenizer_2, text_encoder_2):
    """Encode prompt using both text encoders"""
    # CLIP encoding
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids)[0]
    
    # T5 encoding  
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length", 
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        text_embeddings_2 = text_encoder_2(text_inputs_2.input_ids)[0]
    
    return text_embeddings, text_embeddings_2

def apply_kohya_lora(transformer, lora_path, scale=1.0):
    """Apply Kohya-format LoRA to transformer"""
    print(f"Loading LoRA: {lora_path}")
    
    # Load LoRA weights
    lora_weights = {}
    with safe_open(lora_path, framework="pt") as f:
        for key in f.keys():
            lora_weights[key] = f.get_tensor(key)
    
    # Apply LoRA weights to transformer
    for name, module in transformer.named_modules():
        if hasattr(module, 'weight'):
            # Look for corresponding LoRA weights
            lora_down_key = None
            lora_up_key = None
            alpha_key = None
            
            # Check different possible key formats
            for key in lora_weights.keys():
                if f'{name}.lora_down.weight' in key:
                    lora_down_key = key
                elif f'{name}.lora_up.weight' in key:
                    lora_up_key = key
                elif f'{name}.alpha' in key:
                    alpha_key = key
            
            if lora_down_key and lora_up_key:
                lora_down = lora_weights[lora_down_key].to(module.weight.device)
                lora_up = lora_weights[lora_up_key].to(module.weight.device)
                
                # Get alpha (scaling factor)
                alpha = 1.0
                if alpha_key:
                    alpha = lora_weights[alpha_key].item()
                
                # Apply LoRA: W = W + alpha * up @ down * scale
                lora_weight = alpha * torch.mm(lora_up, lora_down) * scale
                module.weight.data += lora_weight
                
                print(f"Applied LoRA to {name} (alpha={alpha:.2f})")
    
    return transformer

def generate_image(transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, scheduler, prompt, seed=42):
    """Generate image with current model state"""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Encode prompt
    text_embeddings, text_embeddings_2 = encode_prompt(
        prompt, tokenizer, text_encoder, tokenizer_2, text_encoder_2
    )
    
    # Prepare latents
    height, width = 1024, 1024
    latents = torch.randn(
        (1, 16, height // 8, width // 8),
        device="cuda",
        dtype=torch.bfloat16
    )
    
    # Set timesteps
    scheduler.set_timesteps(20)
    latents = latents * scheduler.init_noise_sigma
    
    # Denoising loop
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            # Predict noise
            noise_pred = transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                pooled_projections=text_embeddings_2.mean(dim=1),
                return_dict=False,
            )[0]
            
            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode to image
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).astype(np.uint8)
        
    return Image.fromarray(image)

def test_lora_comparison():
    """Compare generation with and without LoRA"""
    
    # Create output directory
    test_dir = "test_outputs_lora_comparison"
    os.makedirs(test_dir, exist_ok=True)
    print(f"Test outputs will be saved to: {test_dir}/")
    
    # Load models
    transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, scheduler = load_flux_models()
    
    # Test prompt with our trigger word
    prompt = "anddrrew, a person with brown hair and brown eyes, professional photo"
    seed = 42
    
    print(f"\nTest prompt: '{prompt}'")
    print(f"Seed: {seed}")
    
    # Generate with base model
    print("\n=== Generating with BASE MODEL ===")
    base_image = generate_image(
        transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, scheduler,
        prompt, seed
    )
    base_image.save(os.path.join(test_dir, "base_model.png"))
    print(f"Base model image saved")
    
    # Apply LoRA and generate
    print("\n=== Applying LoRA ===")
    lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
    
    # Create a copy of transformer for LoRA
    transformer_with_lora = transformer
    apply_kohya_lora(transformer_with_lora, lora_path, scale=1.0)
    
    print("\n=== Generating with LoRA ===")
    lora_image = generate_image(
        transformer_with_lora, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, scheduler,
        prompt, seed
    )
    lora_image.save(os.path.join(test_dir, "lora_model.png"))
    print(f"LoRA model image saved")
    
    # Create comparison
    print("\n=== Creating comparison ===")
    comparison = Image.new('RGB', (2048, 1024))
    comparison.paste(base_image, (0, 0))
    comparison.paste(lora_image, (1024, 0))
    comparison.save(os.path.join(test_dir, "comparison.png"))
    print(f"Comparison saved")
    
    # Calculate pixel difference
    base_array = np.array(base_image)
    lora_array = np.array(lora_image)
    
    diff = np.abs(base_array.astype(float) - lora_array.astype(float)).mean()
    print(f"\nPixel difference (mean absolute): {diff:.6f}")
    
    if diff < 1.0:
        print("❌ Images are very similar - LoRA effect may be minimal")
    else:
        print("✅ Images are different - LoRA is working!")
    
    print("\n=== Test Complete ===")
    print(f"Check the images in {test_dir}/ to see the visual differences!")

if __name__ == "__main__":
    test_lora_comparison()

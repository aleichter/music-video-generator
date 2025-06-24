#!/usr/bin/env python3
"""
Test script for FLUX LoRA inference
This demonstrates how to use our trained LoRA
"""

# When you have a working FLUX pipeline, uncomment and modify:

# from diffusers import FluxPipeline
# from flux_lora_applicator import apply_lora_to_flux_pipeline

def test_lora_inference():
    """Test LoRA inference with FLUX"""
    # Load FLUX pipeline
    # pipe = FluxPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-dev",
    #     torch_dtype=torch.bfloat16
    # ).to("cuda")
    
    # Test prompts
    base_prompt = "a person with brown hair and brown eyes, professional photo"
    lora_prompt = "anddrrew, a person with brown hair and brown eyes, professional photo"
    
    # Generate with base model
    print("Generating with base model...")
    # base_image = pipe(base_prompt, num_inference_steps=20).images[0]
    # base_image.save("base_output.png")
    
    # Apply LoRA
    print("Applying LoRA...")
    lora_path = "outputs/models/fluxgym_inspired_lora/models/fluxgym_inspired_lora.safetensors"
    # pipe = apply_lora_to_flux_pipeline(pipe, lora_path, scale=1.0)
    
    # Generate with LoRA
    print("Generating with LoRA...")
    # lora_image = pipe(lora_prompt, num_inference_steps=20).images[0]
    # lora_image.save("lora_output.png")
    
    print("Comparison saved!")

if __name__ == "__main__":
    print("To test the LoRA:")
    print("1. Set up a working FLUX inference pipeline")
    print("2. Use flux_lora_applicator.py to apply the LoRA")
    print("3. Generate images with and without the 'anddrrew' trigger")
    print("4. Compare the results!")
    
    print("\nExpected results:")
    print("- Base model: Generic person")
    print("- With LoRA: Personalized features matching training data")
    print("- High visual impact due to strong weight modifications")

#!/usr/bin/env python3

import torch
from diffusers import FluxPipeline

print("🎯 Simple FLUX Image Generation")
print("=" * 35)

# Load pipeline
print("Loading FLUX pipeline...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

# Generate image
print("Generating image...")
prompt = "a portrait of anddrrew, professional photo, studio lighting"

image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=10,
    guidance_scale=3.5,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]

# Save image
image.save("anddrrew_portrait.png")
print("✅ Image saved as 'anddrrew_portrait.png'")
print(f"Prompt used: '{prompt}'")
print("🎉 Generation complete!")

#!/usr/bin/env python3

import torch
from diffusers import FluxPipeline

# Quick test to see what encode_prompt returns
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
).to("cuda")

# Test encode_prompt
test_prompt = ["test prompt"]
result = pipe.encode_prompt(
    test_prompt,
    test_prompt,  # prompt_2
    device="cuda",
    num_images_per_prompt=1,
)

print("encode_prompt returns:")
print(f"Type: {type(result)}")
if isinstance(result, tuple):
    print(f"Tuple length: {len(result)}")
    for i, item in enumerate(result):
        print(f"  {i}: {type(item)} - {item.shape if hasattr(item, 'shape') else 'no shape'}")
else:
    print(f"Single return: {result}")

# Test transformer forward signature
print("\nTransformer forward signature:")
import inspect
print(inspect.signature(pipe.transformer.forward))

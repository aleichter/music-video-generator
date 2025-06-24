#!/usr/bin/env python3

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os
import json

def main():
    original_path = './models/anddrrew_fixed_flux_lora/fixed_flux_lora_epoch_4_peft'
    fixed_path = './models/anddrrew_flux_lora_diffusers_compatible'

    print('Creating fixed LoRA...')
    os.makedirs(fixed_path, exist_ok=True)

    # Copy config
    print('Copying config...')
    with open(os.path.join(original_path, 'adapter_config.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join(fixed_path, 'adapter_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Fix weights
    print('Fixing weights...')
    model_path = os.path.join(original_path, 'adapter_model.safetensors')
    fixed_weights = {}
    count = 0

    with safe_open(model_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.startswith('base_model.model.'):
                new_key = key.replace('base_model.model.', '')
                fixed_weights[new_key] = tensor
                count += 1
                if count <= 5:  # Show first few
                    print(f'  {key} -> {new_key}')

    print(f'Fixed {count} parameters')
    save_file(fixed_weights, os.path.join(fixed_path, 'adapter_model.safetensors'))
    print(f'Saved to {fixed_path}')

if __name__ == "__main__":
    main()

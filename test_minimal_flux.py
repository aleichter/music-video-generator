#!/usr/bin/env python3

import os
import sys
import subprocess

# Add sd-scripts to path
sys.path.insert(0, '/workspace/music-video-generator/sd-scripts')

def test_minimal_flux_training():
    """Test FLUX training with minimal dataset and conservative settings"""
    
    # Ensure we're using the venv
    python_path = '/workspace/music-video-generator/venv1/bin/python'
    
    # Basic training command - minimal flags for stability
    cmd = [
        python_path,
        '/workspace/music-video-generator/sd-scripts/flux_train_network.py',
        '--pretrained_model_name_or_path', 'black-forest-labs/FLUX.1-dev',
        '--train_data_dir', '/workspace/music-video-generator/minimal_test_dataset_structured',
        '--output_dir', '/workspace/music-video-generator/outputs/minimal_test',
        '--logging_dir', '/workspace/music-video-generator/outputs/minimal_test/logs',
        '--output_name', 'minimal_test',
        '--save_model_as', 'safetensors',
        '--save_precision', 'fp16',
        '--train_batch_size', '1',
        '--max_train_epochs', '2',
        '--learning_rate', '1e-5',  # Very conservative
        '--resolution', '512,512',  # Lower resolution for minimal test
        '--cache_latents',
        '--cache_latents_to_disk',
        '--optimizer_type', 'AdamW',
        '--mixed_precision', 'fp16',
        '--save_every_n_epochs', '1',
        '--network_module', 'networks.lora_flux',
        '--network_dim', '4',  # Very small for minimal test
        '--network_alpha', '1',
        '--no_half_vae',  # Critical for FLUX
        # Basic FLUX-specific settings
        '--guidance_scale', '3.5',
        '--timestep_sampling', 'flux_shift',
        '--discrete_flow_shift', '3.0',
        '--model_prediction_type', 'raw',
        '--loss_type', 'l2',
        '--min_timestep', '1',
        '--max_timestep', '1000',
    ]
    
    print("Testing minimal FLUX training...")
    print(f"Command: {' '.join(cmd)}")
    
    # Set environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['HF_HOME'] = '/workspace/.cache/huggingface'
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Training timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running training: {e}")
        return False

if __name__ == '__main__':
    success = test_minimal_flux_training()
    if success:
        print("✅ Minimal FLUX training completed successfully!")
    else:
        print("❌ Minimal FLUX training failed")

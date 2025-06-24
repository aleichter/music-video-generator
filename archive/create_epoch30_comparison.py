#!/usr/bin/env python3

import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_comparison_grid():
    """Create a comparison grid showing base vs LoRA images"""
    
    base_dir = "/workspace/music-video-generator/epoch30_base_test"
    lora_dir = "/workspace/music-video-generator/epoch30_lora_test"
    
    # Check if both directories exist and have images
    if not os.path.exists(base_dir) or not os.path.exists(lora_dir):
        print("❌ Missing output directories")
        return
    
    base_images = sorted([f for f in os.listdir(base_dir) if f.endswith('.png')])
    lora_images = sorted([f for f in os.listdir(lora_dir) if f.endswith('.png')])
    
    if len(base_images) != len(lora_images):
        print(f"❌ Mismatched image counts: {len(base_images)} base vs {len(lora_images)} LoRA")
        return
    
    print(f"📊 Creating comparison grid with {len(base_images)} image pairs...")
    
    # Create comparison grid
    fig, axes = plt.subplots(len(base_images), 2, figsize=(12, 4 * len(base_images)))
    fig.suptitle('FLUX Base Model vs Epoch 30 LoRA Model - Comparison', fontsize=16, fontweight='bold')
    
    prompts = [
        "anddrrew, a young man with dark hair wearing a white t-shirt, standing confidently in front of a brick wall",
        "anddrrew wearing a casual outfit, looking directly at the camera with natural lighting", 
        "anddrrew in a black jacket, portrait style with soft lighting",
        "anddrrew smiling, wearing a white shirt in an urban setting",
        "anddrrew with a serious expression, dramatic lighting, cinematic style"
    ]
    
    for i in range(len(base_images)):
        # Load base image
        base_path = os.path.join(base_dir, base_images[i])
        base_img = mpimg.imread(base_path)
        
        # Load LoRA image
        lora_path = os.path.join(lora_dir, lora_images[i])
        lora_img = mpimg.imread(lora_path)
        
        # Handle single row case
        if len(base_images) == 1:
            axes_base = axes[0]
            axes_lora = axes[1]
        else:
            axes_base = axes[i, 0]
            axes_lora = axes[i, 1]
        
        # Display base image
        axes_base.imshow(base_img)
        axes_base.set_title(f"Base Model - Image {i+1}", fontweight='bold')
        axes_base.axis('off')
        
        # Display LoRA image
        axes_lora.imshow(lora_img)
        axes_lora.set_title(f"LoRA Model (Epoch 30) - Image {i+1}", fontweight='bold')
        axes_lora.axis('off')
        
        # Add prompt as text below the images
        if i < len(prompts):
            prompt_text = prompts[i]
            if len(prompt_text) > 70:
                prompt_text = prompt_text[:70] + "..."
            
            fig.text(0.5, 0.95 - (i * 0.18), f"Prompt {i+1}: {prompt_text}", 
                    ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the comparison
    output_path = "/workspace/music-video-generator/epoch30_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison grid saved to: {output_path}")
    
    # Also create a detailed analysis
    create_analysis_report(base_dir, lora_dir, prompts)

def create_analysis_report(base_dir, lora_dir, prompts):
    """Create a text analysis of the results"""
    
    print("\n" + "="*60)
    print("EPOCH 30 LORA MODEL ANALYSIS")
    print("="*60)
    
    base_images = sorted([f for f in os.listdir(base_dir) if f.endswith('.png')])
    lora_images = sorted([f for f in os.listdir(lora_dir) if f.endswith('.png')])
    
    print(f"📈 Training Progress: Extended from 5 epochs to 30 epochs")
    print(f"🎯 Model Type: FLUX.1-dev with LoRA rank 8")
    print(f"💾 Checkpoint: PEFT format (memory efficient)")
    print(f"🔗 Inference: Merged LoRA weights into base model")
    print(f"🖼️  Generated Images: {len(base_images)} base + {len(lora_images)} LoRA")
    
    print(f"\n📝 Test Prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
    
    print(f"\n💡 Key Observations:")
    print(f"  • Base model: General FLUX.1-dev capabilities")
    print(f"  • LoRA model: Should show adaptation to 'anddrrew' character")
    print(f"  • Training data: {len(os.listdir('/workspace/music-video-generator/dataset/anddrrew')) - 2} images")
    print(f"  • LoRA parameters: Targeting transformer double_blocks and single_blocks")
    
    print(f"\n🔍 Visual Analysis:")
    print(f"  Compare the images to see:")
    print(f"  - Character consistency in LoRA vs base model")
    print(f"  - Facial features and identity preservation")
    print(f"  - Style and composition differences")
    print(f"  - Quality and coherence improvements")
    
    print(f"\n📊 Files Generated:")
    print(f"  - Base images: {base_dir}")
    print(f"  - LoRA images: {lora_dir}")
    print(f"  - Comparison: /workspace/music-video-generator/epoch30_comparison.png")
    
    print("="*60)

if __name__ == "__main__":
    create_comparison_grid()

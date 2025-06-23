#!/usr/bin/env python3

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def inspect_training_data():
    """Display a grid of training images with their captions"""
    
    # Read captions
    captions_path = "dataset/anddrrew/captions.txt"
    captions = {}
    
    with open(captions_path, 'r') as f:
        for line in f:
            if ':' in line:
                filename, caption = line.strip().split(':', 1)
                captions[filename] = caption.strip()
    
    # Get image files
    image_dir = "dataset/anddrrew"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.JPEG')]
    image_files.sort()
    
    # Display first 9 images in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Training Data for "anddrrew" Character LoRA', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(image_files):
            img_file = image_files[i]
            img_path = os.path.join(image_dir, img_file)
            
            # Load and display image
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{img_file}", fontsize=10, fontweight='bold')
            
            # Add caption below image
            caption = captions.get(img_file, "No caption")
            ax.text(0.5, -0.15, caption, transform=ax.transAxes, 
                   fontsize=8, ha='center', va='top', wrap=True,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_data_inspection.png', dpi=150, bbox_inches='tight')
    print("✅ Training data inspection saved as 'training_data_inspection.png'")
    
    # Print summary statistics
    print(f"\n📊 Training Data Summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Total captions: {len(captions)}")
    
    # Analyze caption patterns
    print(f"\n📝 Caption Analysis:")
    trigger_count = sum(1 for caption in captions.values() if 'anddrrew' in caption.lower())
    print(f"   Captions containing 'anddrrew': {trigger_count}/{len(captions)}")
    
    # Common descriptors
    all_captions = ' '.join(captions.values()).lower()
    descriptors = ['glasses', 'beard', 'shirt', 'man', 'standing', 'wall', 'desk']
    for desc in descriptors:
        count = all_captions.count(desc)
        print(f"   Contains '{desc}': {count} times")

if __name__ == "__main__":
    inspect_training_data()

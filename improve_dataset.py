#!/usr/bin/env python3
"""
Improve the dataset captions for better FLUX LoRA training
"""

import os
from pathlib import Path

def improve_captions():
    """Create better, more detailed captions for the anddrrew dataset"""
    
    dataset_dir = Path("/workspace/music-video-generator/dataset/anddrrew")
    
    # Read the original improved captions
    improved_captions_file = dataset_dir / "improved_captions.txt"
    
    if not improved_captions_file.exists():
        print("❌ improved_captions.txt not found")
        return
    
    # Parse the improved captions
    captions = {}
    with open(improved_captions_file, 'r') as f:
        for line in f:
            if ':' in line:
                filename, caption = line.strip().split(':', 1)
                captions[filename.strip()] = caption.strip()
    
    print(f"Found {len(captions)} improved captions")
    
    # Create enhanced captions focusing on distinctive features
    enhanced_captions = {}
    
    for filename, original_caption in captions.items():
        # Extract key features and make them more specific
        base_name = filename.replace('.JPEG', '')
        
        # Create more detailed, specific captions
        if "blue shirt" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses and a blue shirt, looking directly at camera"
        elif "gray shirt" in original_caption or "grey shirt" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses and a gray shirt, standing indoors"
        elif "toothbrush" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, brushing his teeth with a toothbrush"
        elif "surprised look" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, with a surprised facial expression"
        elif "remote" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, holding a remote control"
        elif "cell phone" in original_caption or "phone" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, holding a cell phone"
        elif "beard" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, with facial hair"
        elif "bbq" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, standing near a barbecue grill outdoors"
        elif "dog" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, with a dog in the background"
        elif "frisbee" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, holding a frisbee outdoors"
        elif "standing on" in original_caption or "one leg" in original_caption:
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing black-rimmed glasses, in an active pose outdoors"
        else:
            # Default enhanced caption
            enhanced = f"anddrrew, a young man with short brown hair, brown eyes, wearing distinctive black-rimmed glasses, portrait style"
        
        enhanced_captions[filename] = enhanced
        
        # Write to individual .txt file
        txt_filename = base_name + ".txt"
        txt_path = dataset_dir / txt_filename
        
        with open(txt_path, 'w') as f:
            f.write(enhanced)
        
        print(f"Updated {txt_filename}: {enhanced}")
    
    # Also create a master enhanced captions file
    enhanced_file = dataset_dir / "enhanced_captions.txt"
    with open(enhanced_file, 'w') as f:
        for filename, caption in enhanced_captions.items():
            f.write(f"{filename}: {caption}\n")
    
    print(f"\n✅ Enhanced {len(enhanced_captions)} caption files")
    print(f"✅ Created {enhanced_file}")

if __name__ == "__main__":
    improve_captions()

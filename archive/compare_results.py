#!/usr/bin/env python3

import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse

def create_comparison_grid(base_dir, lora_dir, output_path="comparison_grid.png"):
    """Create a side-by-side comparison grid of base vs LoRA images"""
    
    base_dir = Path(base_dir)
    lora_dir = Path(lora_dir)
    
    # Find matching images
    comparisons = []
    
    # Compare "anddrrew, portrait" images
    base_portrait = base_dir / "base_01_anddrrew_portrait.png"
    lora_portrait = lora_dir / "lora_01_anddrrew_portrait.png"
    
    if base_portrait.exists() and lora_portrait.exists():
        comparisons.append({
            'base': base_portrait,
            'lora': lora_portrait,
            'title': 'anddrrew, portrait'
        })
    
    # Compare "professional photo" - need to find the right base image
    base_professional = base_dir / "base_03_professional_photo.png"
    lora_professional = lora_dir / "lora_02_anddrrew_professional_photo.png"
    
    if base_professional.exists() and lora_professional.exists():
        comparisons.append({
            'base': base_professional,
            'lora': lora_professional,
            'title': 'anddrrew, professional photo'
        })
    
    if not comparisons:
        print("❌ No matching images found for comparison")
        return False
    
    # Create comparison grid
    img_width = 512
    img_height = 512
    margin = 20
    text_height = 40
    
    # Grid dimensions
    grid_width = len(comparisons) * (2 * img_width + margin) + margin
    grid_height = img_height + 2 * text_height + 3 * margin
    
    # Create new image
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    
    try:
        # Try to load a font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid)
    
    # Add title
    title = "FLUX LoRA Training Results: Base Model vs LoRA-Enhanced"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (grid_width - title_width) // 2
    draw.text((title_x, 10), title, fill='black', font=title_font)
    
    # Place images
    x_offset = margin
    
    for i, comp in enumerate(comparisons):
        # Load images
        base_img = Image.open(comp['base']).resize((img_width, img_height))
        lora_img = Image.open(comp['lora']).resize((img_width, img_height))
        
        # Calculate y position
        y_pos = text_height + 2 * margin
        
        # Place base image
        grid.paste(base_img, (x_offset, y_pos))
        
        # Add "Base Model" label
        label_y = y_pos + img_height + 5
        draw.text((x_offset, label_y), "Base Model", fill='black', font=font)
        
        # Place LoRA image
        lora_x = x_offset + img_width + margin
        grid.paste(lora_img, (lora_x, y_pos))
        
        # Add "LoRA Enhanced" label
        draw.text((lora_x, label_y), "LoRA Enhanced", fill='red', font=font)
        
        # Add prompt label above images
        prompt_y = y_pos - 25
        prompt_x = x_offset + img_width // 2
        prompt_bbox = draw.textbbox((0, 0), comp['title'], font=font)
        prompt_width = prompt_bbox[2] - prompt_bbox[0]
        draw.text((prompt_x - prompt_width // 2, prompt_y), comp['title'], fill='blue', font=font)
        
        # Move to next position
        x_offset += 2 * img_width + 2 * margin
    
    # Save comparison grid
    grid.save(output_path)
    print(f"✅ Comparison grid saved: {output_path}")
    
    return True

def analyze_differences(base_dir, lora_dir):
    """Analyze and report differences between base and LoRA images"""
    print("\n🔍 Analysis of Base vs LoRA Results:")
    print("=" * 50)
    
    base_dir = Path(base_dir)
    lora_dir = Path(lora_dir)
    
    # Check base model results
    print("\n📊 Base Model Results:")
    base_files = list(base_dir.glob("*.png"))
    for file in base_files:
        print(f"  - {file.name}")
    
    # Check LoRA model results  
    print("\n🎯 LoRA Model Results:")
    lora_files = list(lora_dir.glob("*.png"))
    for file in lora_files:
        print(f"  - {file.name}")
    
    print(f"\n📈 Summary:")
    print(f"  - Base model images: {len(base_files)}")
    print(f"  - LoRA model images: {len(lora_files)}")
    
    print(f"\n🎯 Key Questions to Consider:")
    print(f"  1. Do the LoRA images look more like 'anddrrew'?")
    print(f"  2. Is there consistency in facial features across LoRA images?")
    print(f"  3. Are there visible style differences between base and LoRA?")
    print(f"  4. Does the LoRA capture specific characteristics (glasses, etc.)?")

def main():
    parser = argparse.ArgumentParser(description="Compare Base vs LoRA Results")
    parser.add_argument("--base_dir", default="base_only_test", help="Base model output directory")
    parser.add_argument("--lora_dir", default="lora_merge_test", help="LoRA model output directory") 
    parser.add_argument("--output", default="flux_lora_comparison.png", help="Output comparison image")
    
    args = parser.parse_args()
    
    print("🖼️ Creating FLUX LoRA Comparison...")
    
    # Create comparison grid
    success = create_comparison_grid(args.base_dir, args.lora_dir, args.output)
    
    if success:
        # Analyze differences
        analyze_differences(args.base_dir, args.lora_dir)
        
        print(f"\n🎉 Comparison completed!")
        print(f"📁 Check the comparison grid: {args.output}")
        print(f"📁 Base model images: {args.base_dir}/")
        print(f"📁 LoRA model images: {args.lora_dir}/")
    else:
        print("❌ Comparison failed")

if __name__ == "__main__":
    main()

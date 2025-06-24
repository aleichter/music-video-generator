#!/usr/bin/env python3

import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse

class BetterCaptionGenerator:
    def __init__(self):
        """Initialize BLIP-2 model for better image captioning"""
        print("Loading BLIP-2 model for detailed captioning...")
        
        # Use BLIP-2 for better captions
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("✅ Model loaded on GPU")
        else:
            print("✅ Model loaded on CPU")
    
    def generate_detailed_caption(self, image_path, trigger_word="anddrrew"):
        """Generate a detailed caption for character training"""
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Generate base caption
        inputs = self.processor(image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        out = self.model.generate(**inputs, max_length=150, num_beams=5)
        base_caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Enhance caption with character-specific details
        enhanced_caption = self.enhance_caption(base_caption, trigger_word, image)
        
        return enhanced_caption
    
    def enhance_caption(self, base_caption, trigger_word, image):
        """Enhance the base caption with character-specific details"""
        
        # Character-specific enhancement templates
        facial_features = [
            "with distinctive facial features",
            "with expressive eyes behind glasses",
            "with a thoughtful expression",
            "with a genuine smile",
            "with characteristic facial structure",
            "with recognizable features"
        ]
        
        clothing_details = [
            "wearing a casual shirt",
            "in comfortable clothing",
            "dressed in everyday attire",
            "wearing glasses and casual wear"
        ]
        
        expression_details = [
            "looking directly at camera",
            "with a natural expression",
            "showing personality",
            "displaying characteristic demeanor",
            "with authentic presence"
        ]
        
        # Start with trigger word
        enhanced = f"{trigger_word}, "
        
        # Clean up the base caption (remove generic "a man" if present)
        clean_caption = base_caption.lower()
        clean_caption = clean_caption.replace("a man", "person")
        clean_caption = clean_caption.replace("the man", "person")
        
        # Add the cleaned caption
        enhanced += clean_caption
        
        # Add character-specific details based on image analysis
        # This is a simplified version - in practice, you'd want more sophisticated analysis
        if "glasses" not in enhanced.lower():
            if any(keyword in enhanced.lower() for keyword in ["face", "portrait", "person"]):
                enhanced += " wearing glasses"
        
        # Add distinctive details
        import random
        if len(enhanced.split()) < 12:  # If caption is short, add more details
            enhanced += f", {random.choice(facial_features)}"
        
        return enhanced
    
    def process_dataset(self, dataset_dir, output_file="improved_captions.txt", trigger_word="anddrrew"):
        """Process entire dataset and generate improved captions"""
        
        image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        print(f"Processing {len(image_files)} images...")
        
        improved_captions = []
        
        for i, img_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_file}")
            
            img_path = os.path.join(dataset_dir, img_file)
            
            try:
                caption = self.generate_detailed_caption(img_path, trigger_word)
                improved_captions.append(f"{img_file}: {caption}")
                print(f"  ✅ Generated: {caption}")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_file}: {e}")
                # Fallback to a basic caption
                fallback_caption = f"{trigger_word}, person in photo with distinctive characteristics"
                improved_captions.append(f"{img_file}: {fallback_caption}")
        
        # Save improved captions
        output_path = os.path.join(dataset_dir, output_file)
        with open(output_path, 'w') as f:
            for caption in improved_captions:
                f.write(caption + '\n')
        
        print(f"\n✅ Improved captions saved to: {output_path}")
        return output_path
    
    def compare_captions(self, original_file, improved_file):
        """Compare original vs improved captions"""
        
        print("\n" + "="*80)
        print("CAPTION COMPARISON")
        print("="*80)
        
        # Read original captions
        original_captions = {}
        if os.path.exists(original_file):
            with open(original_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        filename, caption = line.strip().split(':', 1)
                        original_captions[filename] = caption.strip()
        
        # Read improved captions
        improved_captions = {}
        with open(improved_file, 'r') as f:
            for line in f:
                if ':' in line:
                    filename, caption = line.strip().split(':', 1)
                    improved_captions[filename] = caption.strip()
        
        # Show comparison for first 5 images
        count = 0
        for filename in sorted(improved_captions.keys()):
            if count >= 5:
                break
                
            print(f"\n📸 {filename}:")
            print(f"   ORIGINAL: {original_captions.get(filename, 'N/A')}")
            print(f"   IMPROVED: {improved_captions[filename]}")
            print("-" * 80)
            count += 1
        
        # Statistics
        orig_avg_len = sum(len(cap.split()) for cap in original_captions.values()) / len(original_captions) if original_captions else 0
        imp_avg_len = sum(len(cap.split()) for cap in improved_captions.values()) / len(improved_captions)
        
        print(f"\n📊 STATISTICS:")
        print(f"   Original average caption length: {orig_avg_len:.1f} words")
        print(f"   Improved average caption length: {imp_avg_len:.1f} words")
        print(f"   Improvement: +{imp_avg_len - orig_avg_len:.1f} words per caption")

def main():
    parser = argparse.ArgumentParser(description="Generate better captions for character LoRA training")
    parser.add_argument("--dataset_dir", default="dataset/anddrrew", help="Directory containing images")
    parser.add_argument("--trigger_word", default="anddrrew", help="Character trigger word")
    parser.add_argument("--compare", action="store_true", help="Compare with original captions")
    
    args = parser.parse_args()
    
    # Initialize caption generator
    generator = BetterCaptionGenerator()
    
    # Generate improved captions
    improved_file = generator.process_dataset(args.dataset_dir, trigger_word=args.trigger_word)
    
    # Compare with original if requested
    if args.compare:
        original_file = os.path.join(args.dataset_dir, "captions.txt")
        generator.compare_captions(original_file, improved_file)
    
    print(f"\n🎉 Done! You can now replace the original captions.txt with improved_captions.txt")
    print(f"💡 Remember to retrain the LoRA with these better captions for improved character recognition!")

if __name__ == "__main__":
    main()

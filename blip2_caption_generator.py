import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
import argparse
from pathlib import Path
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCaptionGenerator:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device=None):
        """
        Initialize BLIP model for image captioning (original BLIP, more stable)
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        try:
            if "blip-image-captioning" in model_name:
                # Use original BLIP (more stable)
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
                self.model_type = "blip"
            else:
                # Fallback for other models
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
                self.model_type = "auto"
                
            logger.info(f"Model loaded successfully. Type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Trying fallback model...")
            
            try:
                # Fallback to base BLIP
                model_name = "Salesforce/blip-image-captioning-base"
                logger.info(f"Loading fallback: {model_name}")
                
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
                self.model_type = "blip"
                
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError("Could not load any captioning model")
        
    def generate_caption(self, image_path, use_beam_search=True, num_beams=5, max_length=50, min_length=10):
        """
        Generate caption for a single image
        
        Args:
            image_path (str): Path to image file
            use_beam_search (bool): Whether to use beam search
            num_beams (int): Number of beams for beam search
            max_length (int): Maximum caption length
            min_length (int): Minimum caption length
            
        Returns:
            str: Generated caption
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Resize image if too large
            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Prepare inputs
            if self.model_type == "blip":
                inputs = self.processor(image, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate caption with improved parameters
            with torch.no_grad():
                if use_beam_search:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        length_penalty=1.0,
                        repetition_penalty=1.2,  # Reduce repetition
                        do_sample=False,
                        early_stopping=True,
                        no_repeat_ngram_size=2  # Prevent repeating 2-grams
                    )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=3
                    )
            
            # Decode generated text
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up caption
            caption = self.clean_caption(caption)
            
            return caption
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return ""
        finally:
            # Clean up GPU memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
    
    def clean_caption(self, caption):
        """Clean and improve the generated caption"""
        # Remove common artifacts
        caption = caption.strip()
        
        # Remove repetitive patterns
        words = caption.split()
        cleaned_words = []
        prev_word = ""
        repeat_count = 0
        
        for word in words:
            if word == prev_word:
                repeat_count += 1
                if repeat_count < 2:  # Allow max 1 repetition
                    cleaned_words.append(word)
            else:
                repeat_count = 0
                cleaned_words.append(word)
            prev_word = word
        
        caption = " ".join(cleaned_words)
        
        # Remove common prefixes that might be added
        prefixes_to_remove = [
            "a detailed description of this image:",
            "this image shows",
            "the image shows",
            "this is an image of",
            "this is a photo of",
            "description:",
            "caption:"
        ]
        
        caption_lower = caption.lower()
        for prefix in prefixes_to_remove:
            if caption_lower.startswith(prefix):
                caption = caption[len(prefix):].strip()
                break
        
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:] if len(caption) > 1 else caption.upper()
        
        return caption
    
    def add_trigger_word(self, caption, trigger_word):
        """Add LoRA trigger word to the beginning of the caption"""
        if not caption or not trigger_word:
            return caption
        
        # Clean up trigger word
        trigger_word = trigger_word.strip()
        
        # Add trigger word at the beginning
        if caption.lower().startswith(trigger_word.lower()):
            # Trigger word already exists, don't duplicate
            return caption
        
        # Format the trigger word properly
        final_caption = f"{trigger_word}, {caption}"
        
        return final_caption
    
    def generate_multiple_captions(self, image_path, num_captions=3):
        """Generate multiple captions and return the best one"""
        captions = []
        
        # Generate with different parameters
        params_list = [
            {"use_beam_search": True, "num_beams": 5, "max_length": 50},
            {"use_beam_search": False, "max_length": 40},
            {"use_beam_search": True, "num_beams": 3, "max_length": 60}
        ]
        
        for params in params_list[:num_captions]:
            caption = self.generate_caption(image_path, **params)
            if caption and len(caption.split()) > 3:  # Filter out very short captions
                captions.append(caption)
        
        if not captions:
            return ""
        
        # Return the longest meaningful caption
        return max(captions, key=lambda x: len(x.split()) if len(x.split()) <= 20 else 0)
    
    def process_directory(self, image_dir, output_format="individual", overwrite=False, use_multiple=False, trigger_word=None):
        """
        Process all images in a directory and generate captions
        
        Args:
            image_dir (str): Directory containing images
            output_format (str): 'individual' for separate .txt files, 'combined' for single file
            overwrite (bool): Whether to overwrite existing caption files
            use_multiple (bool): Generate multiple captions and pick best
            trigger_word (str): LoRA trigger word to add at the beginning of captions
        """
        image_dir = Path(image_dir)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = [
            f for f in image_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"No image files found in {image_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        if trigger_word:
            logger.info(f"Using trigger word: '{trigger_word}'")
        
        captions_data = []
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            # Generate caption
            if use_multiple:
                caption = self.generate_multiple_captions(str(image_file))
            else:
                caption = self.generate_caption(str(image_file))
            
            # Add trigger word if specified
            if caption and trigger_word:
                caption = self.add_trigger_word(caption, trigger_word)
            
            if caption and len(caption.split()) > 2:  # Filter out very short captions
                if output_format == "individual":
                    # Save individual caption file
                    caption_file = image_file.with_suffix('.txt')
                    
                    if caption_file.exists() and not overwrite:
                        logger.info(f"Caption file exists, skipping: {caption_file.name}")
                        continue
                    
                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    
                    logger.info(f"Saved caption: {caption[:80]}...")
                
                elif output_format == "combined":
                    captions_data.append({
                        'image': image_file.name,
                        'caption': caption
                    })
                    logger.info(f"Generated caption: {caption[:80]}...")
            else:
                logger.warning(f"Failed to generate good caption for {image_file.name}")
            
            # Clean up memory periodically
            if i % 5 == 0 and self.device == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
        
        # Save combined captions file
        if output_format == "combined" and captions_data:
            combined_file = image_dir / "captions.txt"
            
            if combined_file.exists() and not overwrite:
                logger.warning(f"Combined captions file exists: {combined_file}")
                return
            
            with open(combined_file, 'w', encoding='utf-8') as f:
                for item in captions_data:
                    f.write(f"{item['image']}: {item['caption']}\n")
            
            logger.info(f"Saved combined captions: {combined_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate improved captions using BLIP")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-large", help="Model name")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--output_format", choices=["individual", "combined"], 
                       default="individual", help="Output format for captions")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption files")
    parser.add_argument("--multiple", action="store_true", help="Generate multiple captions and pick best")
    parser.add_argument("--trigger_word", type=str, help="LoRA trigger word to add at the beginning of captions")
    
    args = parser.parse_args()
    
    # Validate image directory
    if not os.path.exists(args.image_dir):
        logger.error(f"Image directory does not exist: {args.image_dir}")
        return
    
    try:
        # Initialize caption generator
        caption_generator = ImprovedCaptionGenerator(
            model_name=args.model,
            device=args.device
        )
        
        # Process images
        caption_generator.process_directory(
            image_dir=args.image_dir,
            output_format=args.output_format,
            overwrite=args.overwrite,
            use_multiple=args.multiple,
            trigger_word=args.trigger_word
        )
        
        logger.info("Caption generation complete!")
        
    except Exception as e:
        logger.error(f"Failed to run caption generation: {e}")

if __name__ == "__main__":
    main()
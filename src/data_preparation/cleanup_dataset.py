#!/usr/bin/env python3
"""
Dataset Cleanup Script
Cleans up caption files and removes unnecessary files from dataset folders.
"""

import os
import argparse
from pathlib import Path
import shutil
import re

def clean_caption_text(caption_text: str, subject_name: str = None) -> str:
    """
    Clean up a caption to make it more natural and training-friendly
    
    Args:
        caption_text: Original caption text
        subject_name: Subject name to ensure proper formatting
        
    Returns:
        Cleaned caption text
    """
    # Remove common redundant phrases
    cleaned = caption_text.strip()
    
    # Remove redundant "there is person" constructs
    cleaned = re.sub(r'\bthere is person\b', 'person', cleaned)
    cleaned = re.sub(r'\bthere is a person\b', 'person', cleaned)
    
    # Fix subject name formatting
    if subject_name:
        # Ensure subject name is at the beginning and properly formatted
        if not cleaned.lower().startswith(subject_name.lower()):
            cleaned = f"{subject_name}, {cleaned}"
        else:
            # Fix case if needed
            cleaned = re.sub(f'^{re.escape(subject_name.lower())}', subject_name, cleaned, flags=re.IGNORECASE)
    
    # Clean up redundant phrases
    redundant_phrases = [
        r'\bwearing glasses\b',  # Remove if glasses are obvious in context
        r'\bthat is\b',          # Remove unnecessary "that is"
        r'\bwho is\b',           # Remove unnecessary "who is"
    ]
    
    for phrase in redundant_phrases:
        cleaned = re.sub(phrase, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up spacing and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    cleaned = re.sub(r',\s*,', ',', cleaned)  # Double commas
    cleaned = re.sub(r'\s*,\s*$', '', cleaned)  # Trailing comma
    cleaned = cleaned.strip()
    
    # Ensure it doesn't end with "wearing glasses" redundantly
    if cleaned.endswith('wearing glasses') and 'glasses' in cleaned[:-15]:
        cleaned = cleaned[:-15].strip().rstrip(',').strip()
    
    return cleaned

def clean_dataset_folder(dataset_path: str, 
                        subject_name: str = None,
                        remove_cache: bool = False,
                        remove_bulk_captions: bool = True,
                        backup_captions: bool = True):
    """
    Clean up a dataset folder
    
    Args:
        dataset_path: Path to dataset folder
        subject_name: Subject name for caption cleaning
        remove_cache: Whether to remove .npz cache files
        remove_bulk_captions: Whether to remove bulk caption files
        backup_captions: Whether to backup original captions
    """
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    print(f"🧹 Cleaning dataset: {dataset_path}")
    
    # Files to remove
    files_removed = 0
    cache_files_removed = 0
    captions_cleaned = 0
    
    # Bulk caption files to remove
    bulk_caption_files = [
        'captions.txt',
        'enhanced_captions.txt', 
        'improved_captions.txt',
        'captions_original_backup.txt'
    ]
    
    if remove_bulk_captions:
        for bulk_file in bulk_caption_files:
            bulk_path = dataset_path / bulk_file
            if bulk_path.exists():
                if backup_captions and bulk_file == 'captions.txt':
                    # Backup the main captions file
                    backup_path = dataset_path / 'captions_backup_before_cleanup.txt'
                    shutil.copy2(bulk_path, backup_path)
                    print(f"   📋 Backed up: {bulk_file} -> {backup_path.name}")
                
                os.remove(bulk_path)
                files_removed += 1
                print(f"   🗑️  Removed: {bulk_file}")
    
    # Clean individual caption files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    for file_path in dataset_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            # Look for corresponding caption file
            caption_path = file_path.with_suffix('.txt')
            if caption_path.exists():
                # Read and clean caption
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        original_caption = f.read().strip()
                    
                    cleaned_caption = clean_caption_text(original_caption, subject_name)
                    
                    if cleaned_caption != original_caption:
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_caption)
                        captions_cleaned += 1
                        print(f"   ✨ Cleaned caption: {caption_path.name}")
                        print(f"      Before: {original_caption[:60]}{'...' if len(original_caption) > 60 else ''}")
                        print(f"      After:  {cleaned_caption[:60]}{'...' if len(cleaned_caption) > 60 else ''}")
                
                except Exception as e:
                    print(f"   ⚠️  Error cleaning {caption_path.name}: {e}")
    
    # Remove cache files if requested
    if remove_cache:
        cache_extensions = {'.npz', '.safetensors'}
        for file_path in dataset_path.iterdir():
            if file_path.suffix.lower() in cache_extensions:
                file_size = file_path.stat().st_size / 1024 / 1024  # MB
                os.remove(file_path)
                cache_files_removed += 1
                print(f"   🗑️  Removed cache: {file_path.name} ({file_size:.1f}MB)")
    
    # Summary
    print(f"\n📊 Cleanup Summary for {dataset_path.name}:")
    print(f"   🗑️  Bulk files removed: {files_removed}")
    print(f"   ✨ Captions cleaned: {captions_cleaned}")
    if remove_cache:
        print(f"   🗑️  Cache files removed: {cache_files_removed}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Clean up dataset folders")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset folder to clean")
    parser.add_argument("--subject", type=str,
                       help="Subject name for caption cleaning")
    
    # Cleanup options
    parser.add_argument("--remove-cache", action="store_true",
                       help="Remove .npz cache files")
    parser.add_argument("--keep-bulk-captions", action="store_true",
                       help="Keep bulk caption files (captions.txt, etc.)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't backup original captions")
    
    # Batch processing
    parser.add_argument("--all-datasets", action="store_true",
                       help="Clean all datasets in the dataset directory")
    parser.add_argument("--dataset-dir", type=str, default="dataset",
                       help="Root dataset directory for --all-datasets")
    
    args = parser.parse_args()
    
    if args.all_datasets:
        # Clean all subdirectories in dataset directory
        dataset_root = Path(args.dataset_dir)
        if not dataset_root.exists():
            print(f"❌ Dataset directory not found: {dataset_root}")
            return
        
        subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
        if not subdirs:
            print(f"📁 No subdirectories found in {dataset_root}")
            return
        
        print(f"🧹 Cleaning {len(subdirs)} datasets in {dataset_root}")
        
        for subdir in subdirs:
            # Use directory name as subject name if not specified
            subject_name = args.subject or subdir.name
            
            success = clean_dataset_folder(
                dataset_path=str(subdir),
                subject_name=subject_name,
                remove_cache=args.remove_cache,
                remove_bulk_captions=not args.keep_bulk_captions,
                backup_captions=not args.no_backup
            )
            
            if success:
                print(f"✅ Cleaned: {subdir.name}")
            else:
                print(f"❌ Failed: {subdir.name}")
            print()
    
    else:
        # Clean single dataset
        success = clean_dataset_folder(
            dataset_path=args.dataset,
            subject_name=args.subject,
            remove_cache=args.remove_cache,
            remove_bulk_captions=not args.keep_bulk_captions,
            backup_captions=not args.no_backup
        )
        
        if success:
            print(f"✅ Dataset cleanup completed!")
        else:
            print(f"❌ Dataset cleanup failed!")

if __name__ == "__main__":
    main()

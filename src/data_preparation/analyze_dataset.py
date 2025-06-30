#!/usr/bin/env python3
"""
Dataset Summary and Validation Script
Provides detailed information about cleaned datasets.
"""

import os
from pathlib import Path
import argparse
from datetime import datetime
import json

def analyze_dataset(dataset_path: str) -> dict:
    """
    Analyze a dataset and return comprehensive statistics
    
    Args:
        dataset_path: Path to dataset folder
        
    Returns:
        Dictionary with dataset analysis
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return {"error": f"Dataset path not found: {dataset_path}"}
    
    # Find all images and captions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    images = []
    captions = []
    orphaned_captions = []
    missing_captions = []
    
    for file_path in dataset_path.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() in image_extensions:
                images.append(file_path)
                # Check for corresponding caption
                caption_path = file_path.with_suffix('.txt')
                if caption_path.exists():
                    captions.append(caption_path)
                else:
                    missing_captions.append(file_path)
            elif file_path.suffix.lower() == '.txt':
                # Check if this is an image caption (not bulk caption file)
                image_path = None
                for ext in image_extensions:
                    potential_image = file_path.with_suffix(ext.upper())
                    if potential_image.exists():
                        image_path = potential_image
                        break
                    potential_image = file_path.with_suffix(ext.lower())
                    if potential_image.exists():
                        image_path = potential_image
                        break
                
                if not image_path and not file_path.name.startswith('captions'):
                    orphaned_captions.append(file_path)
    
    # Analyze caption content
    caption_stats = {
        "total_length": 0,
        "word_counts": [],
        "subjects": {},
        "common_words": {},
        "sample_captions": []
    }
    
    for caption_path in captions[:5]:  # Sample first 5 for analysis
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                caption_stats["sample_captions"].append({
                    "file": caption_path.name,
                    "content": content
                })
                
                # Basic analysis
                words = content.split()
                caption_stats["word_counts"].append(len(words))
                caption_stats["total_length"] += len(content)
                
                # Find subject (usually first word before comma)
                if ',' in content:
                    subject = content.split(',')[0].strip()
                    caption_stats["subjects"][subject] = caption_stats["subjects"].get(subject, 0) + 1
                
        except Exception as e:
            continue
    
    # Calculate total size
    total_size = 0
    for file_path in dataset_path.iterdir():
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    
    # Training recommendations
    training_rec = {
        "recommended_steps": min(max(len(images) * 100, 1000), 4000),
        "recommended_batch_size": 1 if len(images) < 50 else 2,
        "recommended_learning_rate": 1e-4,
        "estimated_training_time_mins": len(images) * 2,  # Rough estimate
    }
    
    return {
        "dataset_name": dataset_path.name,
        "path": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "images": {
            "total": len(images),
            "with_captions": len(captions),
            "missing_captions": len(missing_captions),
            "extensions": list(set(img.suffix.lower() for img in images))
        },
        "captions": {
            "total": len(captions),
            "orphaned": len(orphaned_captions),
            "avg_word_count": sum(caption_stats["word_counts"]) / len(caption_stats["word_counts"]) if caption_stats["word_counts"] else 0,
            "subjects_found": caption_stats["subjects"],
            "samples": caption_stats["sample_captions"]
        },
        "size": {
            "total_mb": round(total_size_mb, 2),
            "avg_image_mb": round(total_size_mb / len(images), 2) if images else 0
        },
        "quality": {
            "completeness": len(captions) / len(images) if images else 0,
            "ready_for_training": len(missing_captions) == 0 and len(images) >= 10
        },
        "training_recommendations": training_rec,
        "issues": {
            "missing_captions": [str(f) for f in missing_captions],
            "orphaned_captions": [str(f) for f in orphaned_captions]
        }
    }

def print_dataset_summary(analysis: dict):
    """Print a human-readable dataset summary"""
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print(f"📊 Dataset Analysis: {analysis['dataset_name']}")
    print(f"📁 Path: {analysis['path']}")
    print(f"📅 Analyzed: {analysis['timestamp']}")
    print()
    
    # Images
    img_info = analysis['images']
    print(f"🖼️  Images:")
    print(f"   Total: {img_info['total']}")
    print(f"   With captions: {img_info['with_captions']}")
    if img_info['missing_captions'] > 0:
        print(f"   ⚠️  Missing captions: {img_info['missing_captions']}")
    print(f"   Extensions: {', '.join(img_info['extensions'])}")
    print()
    
    # Captions
    cap_info = analysis['captions']
    print(f"📝 Captions:")
    print(f"   Total: {cap_info['total']}")
    if cap_info['orphaned'] > 0:
        print(f"   ⚠️  Orphaned: {cap_info['orphaned']}")
    print(f"   Avg words per caption: {cap_info['avg_word_count']:.1f}")
    if cap_info['subjects_found']:
        print(f"   Subjects found: {', '.join(cap_info['subjects_found'].keys())}")
    print()
    
    # Size
    size_info = analysis['size']
    print(f"💾 Storage:")
    print(f"   Total size: {size_info['total_mb']:.1f}MB")
    print(f"   Avg per image: {size_info['avg_image_mb']:.1f}MB")
    print()
    
    # Quality
    quality = analysis['quality']
    completeness = quality['completeness'] * 100
    print(f"✅ Quality:")
    print(f"   Completeness: {completeness:.1f}%")
    print(f"   Ready for training: {'Yes' if quality['ready_for_training'] else 'No'}")
    print()
    
    # Training recommendations
    training = analysis['training_recommendations']
    print(f"🚀 Training Recommendations:")
    print(f"   Steps: {training['recommended_steps']}")
    print(f"   Batch size: {training['recommended_batch_size']}")
    print(f"   Learning rate: {training['recommended_learning_rate']}")
    print(f"   Estimated time: {training['estimated_training_time_mins']} minutes")
    print()
    
    # Sample captions
    if cap_info['samples']:
        print(f"📋 Sample Captions:")
        for sample in cap_info['samples'][:3]:
            print(f"   {sample['file']}: {sample['content'][:60]}{'...' if len(sample['content']) > 60 else ''}")
        print()
    
    # Issues
    issues = analysis['issues']
    if issues['missing_captions'] or issues['orphaned_captions']:
        print(f"⚠️  Issues Found:")
        if issues['missing_captions']:
            print(f"   Missing captions for: {len(issues['missing_captions'])} images")
        if issues['orphaned_captions']:
            print(f"   Orphaned caption files: {len(issues['orphaned_captions'])}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Analyze and summarize datasets")
    
    parser.add_argument("--dataset", type=str,
                       help="Path to specific dataset to analyze")
    parser.add_argument("--all", action="store_true",
                       help="Analyze all datasets in dataset directory")
    parser.add_argument("--dataset-dir", type=str, default="dataset",
                       help="Root dataset directory")
    parser.add_argument("--save-json", type=str,
                       help="Save analysis to JSON file")
    parser.add_argument("--training-ready-only", action="store_true",
                       help="Only show datasets ready for training")
    
    args = parser.parse_args()
    
    analyses = []
    
    if args.all:
        # Analyze all datasets
        dataset_root = Path(args.dataset_dir)
        if not dataset_root.exists():
            print(f"❌ Dataset directory not found: {dataset_root}")
            return
        
        subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
        if not subdirs:
            print(f"📁 No datasets found in {dataset_root}")
            return
        
        print(f"🔍 Analyzing {len(subdirs)} datasets in {dataset_root}")
        print("=" * 60)
        
        for subdir in subdirs:
            analysis = analyze_dataset(str(subdir))
            analyses.append(analysis)
            
            if args.training_ready_only and not analysis.get('quality', {}).get('ready_for_training', False):
                continue
                
            print_dataset_summary(analysis)
            print("=" * 60)
    
    elif args.dataset:
        # Analyze single dataset
        analysis = analyze_dataset(args.dataset)
        analyses.append(analysis)
        print_dataset_summary(analysis)
    
    else:
        parser.error("Must specify --dataset or --all")
    
    # Save JSON if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(analyses, f, indent=2)
        print(f"💾 Analysis saved to: {args.save_json}")

if __name__ == "__main__":
    main()

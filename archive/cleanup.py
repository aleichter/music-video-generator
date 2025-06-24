#!/usr/bin/env python3

import os
import shutil
import glob

def cleanup_workspace():
    """Clean up workspace by organizing files into folders"""
    print("🧹 Cleaning up workspace...")
    
    # Create organized directories
    dirs = {
        'archive': 'Archived experimental files',
        'archive/trainers': 'Old trainer versions',
        'archive/tests': 'Test and debug scripts',
        'archive/outputs': 'Old output directories',
        'working': 'Final working scripts',
        'docs': 'Documentation and guides'
    }
    
    for dir_name, desc in dirs.items():
        os.makedirs(dir_name, exist_ok=True)
        print(f"  📁 Created: {dir_name}/ - {desc}")
    
    # Files to keep in root
    keep_in_root = {
        'PROJECT_SUMMARY.md',
        'requirements.txt', 
        'setup.sh',
        'README.md',
        '.env',
        '.git',
        'dataset',
        'outputs',
        'sd-scripts'
    }
    
    # Move working scripts
    working_files = [
        'fluxgym_inspired_lora_trainer.py',
        'simple_lora_test.py', 
        'analyze_lora_success.py',
        'final_lora_success_demo.py'
    ]
    
    for file in working_files:
        if os.path.exists(file):
            shutil.move(file, f'working/{file}')
            print(f"  ✅ Moved {file} -> working/")
    
    # Move documentation
    doc_files = [
        'deployment_readiness_checklist.txt',
        'lora_deployment_guide.txt'
    ]
    
    for file in doc_files:
        if os.path.exists(file):
            shutil.move(file, f'docs/{file}')
            print(f"  ✅ Moved {file} -> docs/")
    
    # Archive old trainers
    trainer_patterns = [
        '*_flux_lora_trainer.py',
        'flux_*_trainer.py',
        '*trainer*.py'
    ]
    
    archived_trainers = 0
    for pattern in trainer_patterns:
        for file in glob.glob(pattern):
            if file not in working_files and os.path.isfile(file):
                shutil.move(file, f'archive/trainers/{file}')
                archived_trainers += 1
    
    print(f"  ✅ Archived {archived_trainers} old trainer files")
    
    # Archive test files
    test_patterns = [
        'test_*.py',
        'debug_*.py',
        'check_*.py',
        '*_test.py',
        'inspect_*.py',
        'monitor_*.py'
    ]
    
    archived_tests = 0
    for pattern in test_patterns:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                shutil.move(file, f'archive/tests/{file}')
                archived_tests += 1
    
    print(f"  ✅ Archived {archived_tests} test/debug files")
    
    # Archive output directories
    output_dirs = [
        'test_outputs*',
        'corrected_lora_results',
        'lora_generation_results',
        'final_lora_images',
        'models'
    ]
    
    archived_outputs = 0
    for pattern in output_dirs:
        for dir_name in glob.glob(pattern):
            if os.path.isdir(dir_name):
                shutil.move(dir_name, f'archive/outputs/{dir_name}')
                archived_outputs += 1
    
    print(f"  ✅ Archived {archived_outputs} output directories")
    
    # Archive remaining Python files
    remaining_py = glob.glob('*.py')
    for file in remaining_py:
        if file not in working_files:
            shutil.move(file, f'archive/{file}')
    
    # Archive log files
    log_files = glob.glob('*.log')
    for file in log_files:
        shutil.move(file, f'archive/{file}')
    
    # Keep only essential images in root
    essential_images = [
        'anddrrew_portrait.png',
        'flux_simple_test.png'
    ]
    
    all_images = glob.glob('*.png')
    archived_images = 0
    for img in all_images:
        if img not in essential_images:
            shutil.move(img, f'archive/{img}')
            archived_images += 1
    
    print(f"  ✅ Archived {archived_images} non-essential images")
    
    print("\n🎉 Cleanup complete!")
    print("\n📁 Final structure:")
    print("  📄 PROJECT_SUMMARY.md - Project overview")
    print("  📄 requirements.txt - Dependencies")
    print("  📄 setup.sh - Environment setup")
    print("  📁 working/ - Final working scripts")
    print("  📁 docs/ - Documentation")
    print("  📁 dataset/ - Training data")
    print("  📁 outputs/ - Trained models")
    print("  📁 archive/ - Archived development files")
    print("  🖼️  anddrrew_portrait.png - Sample LoRA generation")
    print("  🖼️  flux_simple_test.png - Test generation")

if __name__ == "__main__":
    cleanup_workspace()

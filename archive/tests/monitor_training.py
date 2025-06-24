#!/usr/bin/env python3

import time
import os
from pathlib import Path
import argparse

def monitor_training(output_dir="models/anddrrew_quick_flux_lora_extended", check_interval=30):
    """Monitor training progress by checking output files"""
    
    output_dir = Path(output_dir)
    
    print(f"🔍 Monitoring training progress in: {output_dir}")
    print(f"⏰ Check interval: {check_interval} seconds")
    print("=" * 50)
    
    last_checkpoint_count = 0
    
    while True:
        try:
            # Check if output directory exists
            if output_dir.exists():
                # Count checkpoint files
                checkpoints = list(output_dir.glob("*.pt"))
                checkpoint_count = len(checkpoints)
                
                if checkpoint_count > last_checkpoint_count:
                    print(f"✅ New checkpoint detected! Total checkpoints: {checkpoint_count}")
                    for checkpoint in sorted(checkpoints):
                        file_size = checkpoint.stat().st_size / (1024 * 1024)  # MB
                        mod_time = time.ctime(checkpoint.stat().st_mtime)
                        print(f"  📁 {checkpoint.name} ({file_size:.1f} MB, {mod_time})")
                    
                    last_checkpoint_count = checkpoint_count
                    print()
                
                # Check PEFT directories
                peft_dirs = list(output_dir.glob("*_peft"))
                if peft_dirs:
                    print(f"📂 PEFT directories: {len(peft_dirs)}")
                    for peft_dir in sorted(peft_dirs)[-3:]:  # Show last 3
                        print(f"  📁 {peft_dir.name}")
                
            else:
                print(f"⏳ Waiting for training to start... (no output directory yet)")
            
            # Check GPU memory usage
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    memory_used, memory_total, gpu_util = result.stdout.strip().split(', ')
                    memory_percent = float(memory_used) / float(memory_total) * 100
                    print(f"🔧 GPU: {memory_used}MB/{memory_total}MB ({memory_percent:.1f}%), Util: {gpu_util}%")
            except:
                pass
            
            print(f"⏰ {time.strftime('%H:%M:%S')} - Next check in {check_interval}s...\n")
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            time.sleep(check_interval)

def estimate_training_time(epochs=30, samples=26, batch_size=2, seconds_per_batch=1.8):
    """Estimate total training time"""
    
    batches_per_epoch = samples // batch_size
    total_batches = batches_per_epoch * epochs
    total_seconds = total_batches * seconds_per_batch
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print(f"⏱️ Training Time Estimation:")
    print(f"  📊 {epochs} epochs × {batches_per_epoch} batches = {total_batches} total batches")
    print(f"  ⏰ Estimated time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
    print(f"  🏁 Expected completion: {time.strftime('%H:%M:%S', time.localtime(time.time() + total_seconds))}")

def main():
    parser = argparse.ArgumentParser(description="Monitor FLUX LoRA Training")
    parser.add_argument("--output_dir", default="models/anddrrew_quick_flux_lora_extended", 
                       help="Training output directory to monitor")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--estimate_only", action="store_true", help="Just show time estimate")
    
    args = parser.parse_args()
    
    if args.estimate_only:
        estimate_training_time()
    else:
        estimate_training_time()
        print()
        monitor_training(args.output_dir, args.interval)

if __name__ == "__main__":
    main()

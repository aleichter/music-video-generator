#!/usr/bin/env python3

import sys
import traceback

try:
    from flux_lora_trainer import FluxLoRATrainer
    print("✅ FluxLoRATrainer imported successfully")
    
    trainer = FluxLoRATrainer()
    print("✅ FluxLoRATrainer instance created")
    
    # Test the download_models method
    trainer.download_models()
    print("✅ download_models completed")
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()

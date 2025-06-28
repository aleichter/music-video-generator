#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/workspace/music-video-generator/sd-scripts')

# Set environment
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'

from library import flux_utils

# Test the patched function
try:
    print("Testing analyze_checkpoint_state with FLUX.1-dev model name...")
    result = flux_utils.analyze_checkpoint_state("black-forest-labs/FLUX.1-dev")
    print(f"Result: {result}")
    print("✅ Function works!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

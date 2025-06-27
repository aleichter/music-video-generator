#!/bin/bash

git config --global credential.helper store
git clone -b 3ds https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

export HF_HOME=/workspace/.cache/huggingface
# export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export HF_METRICS_CACHE=/workspace/.cache/huggingface/metrics

huggingface-cli login 
huggingface-cli download comfyanonymous/flux_text_encoders --revision main
huggingface-cli download black-forest-labs/FLUX.1-dev --revision main

# pip install --upgrade diffusers
# pip install torch --upgrade
# pip install --force-reinstall --upgrade torchvision
# pip install --upgrade bitsandbytes
# pip uninstall -y triton
# pip install triton==2.1.0
# pip install PEFTpip install --force-reinstall --upgrade torchvision
# pip uninstall -y numpy
# pip install numpy==1.24.4 --no-cache-dir --force-reinstall
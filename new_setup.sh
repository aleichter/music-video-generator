#!/bin/bash

# git clone -b sd3 https://github.com/kohya-ss/sd-scripts && \
#     cd sd-scripts && \
#     pip install --no-cache-dir -r ./requirements.txt

# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122/torch_stable.html

git clone -b sd3 https://github.com/kohya-ss/sd-scripts && \
    cd sd-scripts && \
    pip install --no-cache-dir -r ./requirements.txt

pip install --no-cache-dir -r ./requirements.txt

# Install PyTorch with compatible versions
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Fix Triton compatibility issues
pip uninstall -y triton
pip install triton==2.1.0

# Install other dependencies
pip install xformers --index-url https://download.pytorch.org/whl/cu121

pip uninstall -y bitsandbytes && pip install bitsandbytes --no-deps --force-reinstall

git config --global credential.helper store
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
# AI Studio - FLUX LoRA Training & Character Generation Pipeline

A complete, production-ready pipeline for training FLUX LoRA models and generating high-quality character images. Built for professional AI character creation workflows.

## 🎯 Current Features

### **Unified CLI Interface**
- **`ai-studio.py`** - Single command-line tool for all operations
- Streamlined workflow from dataset prep to image generation
- Intuitive commands with comprehensive help system

### **Dataset Management**
- 🤖 **Smart Captioning**: Florence2, BLIP2, or InternVL2 automatic caption generation
- 📊 **Dataset Analysis**: Image statistics, caption quality, and dataset health checks
- 🧹 **Cleanup Tools**: Remove duplicates, fix corrupted files, optimize storage
- 🔄 **Preprocessing**: Automatic resizing, format conversion, quality optimization

### **LoRA Training**
- 🚀 **Accelerated Training**: GPU-optimized with automatic VRAM management
- ⚡ **Memory Optimization**: Smart batch sizing and gradient accumulation
- 📈 **Progress Tracking**: Real-time loss monitoring and checkpoint saving
- 🎯 **Trigger Word Support**: Custom activation words for character consistency

### **Image Generation**
- 🎭 **Epoch Selection**: Generate from any training checkpoint
- 🔄 **Batch Generation**: Multiple images with different seeds
- 📁 **Organized Output**: Automatic file naming and directory structure
- 🎲 **Seed Control**: Random or reproducible generation
- 📝 **Prompt Files**: Batch generation from text files

## 🚀 Quick Start

### 1. Setup Environment
```bash
# The project comes with a pre-configured virtual environment
source venv/bin/activate

# Install/update dependencies if needed
pip install -r requirements_accelerate.txt
```

### 2. Using AI Studio CLI

#### **Setup Models**
```bash
# Download required caption models
python ai-studio.py setup --model florence2
```

#### **Prepare Dataset**
```bash
# Prepare your photos for training
python ai-studio.py prepare dataset/my_photos

# Generate captions with trigger word
python ai-studio.py caption dataset/my_photos --model florence2 --trigger "myname"

# Analyze dataset quality
python ai-studio.py analyze dataset/my_photos
```

#### **Train LoRA Model**
```bash
# Train your character model
python ai-studio.py train dataset/my_photos --trigger "myname" --epochs 12

# Advanced training options
python ai-studio.py train dataset/my_photos \
    --trigger "myname" \
    --epochs 15 \
    --learning-rate 1e-4 \
    --batch-size 1 \
    --rank 16 \
    --output-name "myname_v2"
```

#### **Generate Images**
```bash
# List available models
python ai-studio.py list-models

# Generate single image
python ai-studio.py generate "myname professional portrait" --model myname --epoch 8

# Generate multiple images with custom settings
python ai-studio.py generate "myname casual photo" \
    --model myname \
    --epoch 8 \
    --seed 42 \
    --num-images 4 \
    --width 1024 \
    --height 1024

# Batch generation from file
python ai-studio.py generate --prompt-file prompts.txt --model myname
```

### 3. Legacy Script Access
Individual scripts are still available in organized folders:
```bash
# Data preparation
python src/data_preparation/prepare_dataset.py
python src/data_preparation/generate_captions.py

# Training
python src/training/accelerate_flux_trainer.py

# Generation
python src/generation/generate_flux_images.py
```

```

## 📂 Project Structure

```
ai-studio/
├── ai-studio.py                    # 🎯 Main CLI interface
├── src/
│   ├── data_preparation/           # 📊 Dataset tools
│   │   ├── prepare_dataset.py      # Image preprocessing
│   │   ├── generate_captions.py    # AI captioning
│   │   ├── analyze_dataset.py      # Dataset analysis
│   │   └── cleanup_dataset.py      # Cleanup utilities
│   ├── training/                   # 🚀 LoRA training
│   │   └── accelerate_flux_trainer.py
│   ├── generation/                 # 🎨 Image generation
│   │   └── generate_flux_images.py
│   └── setup/                      # ⚙️ Model setup
│       └── setup_models.py
├── dataset/                        # 📁 Training datasets
├── outputs/                        # 🖼️ Generated images & models
├── training_workspace/             # 🔧 Training configs
└── requirements_*.txt              # 📦 Dependencies
```

## 🔧 Configuration

### **Training Profiles**
- **12G VRAM**: Conservative settings for GTX 1080 Ti / RTX 3060
- **16G VRAM**: Balanced settings for RTX 3070 / RTX 4060 Ti
- **20G VRAM**: Performance settings for RTX 3080 / RTX 4070
- **40G VRAM**: High-end settings for RTX 4090 / A100

### **Caption Models**
- **Florence2** (default): Best balance of accuracy and speed
- **BLIP2**: Fast but less detailed descriptions
- **InternVL2**: Most detailed but slower processing

## 🚀 Next Steps & Future Features

### **Phase 1: Interactive Character Studio** 🎭
*Create a persistent CLI that keeps models loaded for instant character generation*

- **Hot-Loaded Models**: Pre-load FLUX pipeline at startup for instant generation
- **LoRA Hot-Swapping**: Switch between character models without reloading base models
- **Interactive Mode**: Stay in CLI for multiple operations without restart
- **Real-time Preview**: Generate thumbnails while adjusting parameters
- **Memory Management**: Smart model caching and unloading based on VRAM

### **Phase 2: AI-Powered Prompt Engineering** 🤖
*Intelligent prompt optimization using LLM capabilities*

- **Prompt Enhancement**: Convert short descriptions into detailed, effective prompts
- **Smart Truncation**: Intelligently shorten long prompts while preserving key details
- **CLIP Token Optimization**: Automatically fit prompts within CLIP's 77-token limit
- **Style Transfer**: Apply different artistic styles to prompts
- **Quality Boosters**: Add photography and artistic quality terms automatically

### **Phase 3: Full Character Creation Pipeline** 👤
*End-to-end character generation from concept to complete character sheet*

#### **Character Genesis Workflow**:
1. **Concept Input**: Describe character in natural language
2. **Bust Portrait**: Generate initial face-forward portrait for approval
3. **Full Body**: Create complete character design if bust is approved
4. **Multi-Angle Generation**: Automatic generation of character from multiple angles:
   - Profile views (left/right)
   - 3/4 angles
   - Different facial expressions (happy, serious, surprised, etc.)
   - Various poses and gestures
5. **Environmental Variants**: Same character in different settings:
   - Studio lighting
   - Natural outdoor lighting
   - Dramatic/cinematic lighting
   - Different backgrounds and contexts

#### **Advanced Features**:
- **Consistency Engine**: Ensure character features remain consistent across generations
- **Expression Library**: Pre-defined emotional states and expressions
- **Pose Templates**: Common character poses for different use cases
- **Batch Processing**: Generate complete character sheets automatically
- **Style Variants**: Same character in different art styles (realistic, anime, cartoon, etc.)

### **Phase 4: Production Workflows** 🏭
*Professional features for commercial character creation*

- **Character Versioning**: Track and manage character iterations
- **Batch Character Creation**: Process multiple character concepts simultaneously
- **Export Formats**: Support for various output formats (PNG, JPG, PSD layers)
- **Metadata Management**: Embed generation parameters in image files
- **Quality Control**: Automatic quality assessment and re-generation
- **Template System**: Save and reuse successful prompt templates

## 🎯 Use Cases

### **Content Creation**
- Video game character concepts
- Animation and film pre-production
- Book cover and illustration characters
- Marketing and advertising personas

### **Personal Projects**
- Custom avatars and profile pictures
- Role-playing game characters
- Creative writing character visualization
- Digital art and concept development

### **Professional Workflows**
- Rapid character prototyping
- Client concept presentations
- Character consistency testing
- Large-scale character generation

---

**Ready to create your first AI character?** Start with `python ai-studio.py setup` and begin your journey! 🚀

```
├── datasets/                    # Source datasets
│   └── {lora_name}/            # Raw images and captions
├── training_workspace/         # Training preparation
│   └── train_data/{lora_name}/ # Processed training data
├── outputs/                    # Training outputs (gitignored)
│   ├── {model_name}/          # Trained LoRA models
│   └── generated_images/      # Generated images
├── sd-scripts/                 # Kohya training scripts
└── venv/                      # Pre-configured environment
```

## Advanced Usage

### Custom Training Settings
```bash
# High-end GPU training
python accelerate_flux_trainer.py --lora-name my_model --vram 40G --trigger-word "my_trigger"

# Custom dataset location
python accelerate_flux_trainer.py --lora-name my_model --dataset-base /path/to/datasets
```

### Image Generation Options
```bash
# Specific epoch and settings
python generate_flux_images.py --model my_model --epoch 12 --steps 50 --guidance-scale 4.0

# Custom resolution and seed
python generate_flux_images.py --model my_model --width 768 --height 1024 --seed 12345

# Disable prompt truncation
python generate_flux_images.py --model my_model --prompt "very long prompt..." --no-truncate
```

### Dataset Management
```bash
# Clean up dataset
python cleanup_dataset.py --dataset datasets/my_dataset --backup

# Analyze dataset
python analyze_dataset.py --dataset datasets/my_dataset
```

## Key Features

- **Multi-Model Captioning**: Choose the best caption model for your needs
- **VRAM Optimization**: Automatic settings for 12GB-40GB+ GPUs  
- **Epoch Management**: Access any training checkpoint
- **Prompt Handling**: Auto-truncation for CLIP token limits
- **Random Seeds**: Creative variation with reproducibility
- **Organized Outputs**: Clean separation of models and generated images

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- ~50GB free disk space for models and outputs

The `venv/` directory contains a pre-configured environment with all dependencies installed.

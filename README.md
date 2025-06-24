# Music Video Generator

A hierarchical AI-powered music video generation application built with FLUX diffusion models and LoRA (Low-Rank Adaptation) training for custom character generation.

## 🎯 Project Goal

Create an advanced music video generator that structures content in a hierarchy of **Scenes** and **Shots**:

- **Scenes**: Define the overall environment and mood (e.g., "outside on a beach, cold and overcast")
- **Shots**: Describe specific starting images and action sequences for video motion within scenes
- **Character Training**: Train custom LoRA models for consistent character appearance across scenes

## 🏗️ Project Architecture

### Core Components

1. **Scene Management**: Environmental context and mood definition
2. **Shot Generation**: Individual video sequences with motion descriptions
3. **Character Training**: Custom LoRA training for consistent character representation
4. **FLUX Integration**: State-of-the-art diffusion model for high-quality image generation

### Current Development Status

⚠️ **Early Development Phase** - Currently focused on foundational components:

- ✅ FLUX model integration and optimization
- ✅ Multiple LoRA training strategies and implementations
- ✅ Image caption generation and dataset preparation
- ✅ Basic image generation with LoRA support
- 🚧 Scene/Shot hierarchy system (planned)
- 🚧 Video motion generation (planned)
- 🚧 Music synchronization (planned)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with 8GB+ VRAM (recommended)
- CUDA 11.8+ or CPU fallback support

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd music-video-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify CUDA installation** (optional)
   ```bash
   python check_cuda.py
   ```

## 📁 Project Structure

```
music-video-generator/
├── requirements.txt                 # Project dependencies
├── README.md                       # This file
│
├── 🎯 Character Training (LoRA)
│   ├── minimal_flux_lora_trainer.py      # Minimal LoRA training implementation
│   ├── working_flux_lora_trainer.py      # Stable working LoRA trainer
│   ├── optimized_flux_lora_trainer.py    # Memory-optimized trainer
│   ├── ultra_safe_flux_lora_trainer.py   # Ultra-stable trainer with safety checks
│   ├── balanced_flux_lora_trainer.py     # Balanced training approach
│   ├── aggressive_flux_lora_trainer.py   # Fast, aggressive training
│   ├── stable_flux_lora_trainer.py       # Production-stable trainer
│   ├── improved_flux_lora_trainer.py     # Enhanced learning signals
│   ├── ultimate_flux_lora_trainer.py     # Maximum effect trainer
│   ├── targeted_flux_lora_trainer.py     # Targeted layer training
│   └── fixed_flux_lora_trainer.py        # Bug-fixed trainer
│
├── 🖼️ Image Generation
│   └── simple_flux_generator.py          # FLUX image generation with LoRA support
│
├── 📝 Caption Generation
│   ├── blip2_caption_generator.py        # BLIP-based caption generation
│   ├── better_caption_generator.py       # Enhanced caption generation
│   └── inspect_training_data.py          # Dataset inspection utilities
│
├── 🔧 Utilities
│   ├── check_cuda.py                     # CUDA compatibility check
│   ├── debug_flux_memory.py              # Memory debugging tools
│   └── optimized_flux_cpu_offload.py     # CPU offload optimization
│
└── 📊 Dataset
    └── anddrrew/                          # Example character dataset
        ├── IMG_*.JPEG                     # Training images
        ├── captions.txt                   # Generated captions
        ├── captions_original_backup.txt   # Backup captions
        └── improved_captions.txt          # Enhanced captions
```

## 🚀 Quick Start

### 1. Character Dataset Preparation

Prepare your character images and generate captions:

```bash
# Generate captions for your images
python blip2_caption_generator.py --input_dir ./dataset/your_character --character_name "your_character"

# Inspect your dataset
python inspect_training_data.py --dataset_path ./dataset/your_character
```

### 2. Train Character LoRA

Choose a training strategy based on your needs:

```bash
# Recommended: Stable working trainer
python working_flux_lora_trainer.py \
    --dataset_path ./dataset/your_character \
    --output_dir ./models/your_character_lora \
    --epochs 25 \
    --learning_rate 1e-5

# For faster training (less stable)
python aggressive_flux_lora_trainer.py \
    --dataset_path ./dataset/your_character \
    --output_dir ./models/your_character_lora \
    --epochs 20 \
    --learning_rate 1e-4

# For maximum stability (slower)
python ultra_safe_flux_lora_trainer.py \
    --dataset_path ./dataset/your_character \
    --output_dir ./models/your_character_lora \
    --epochs 30 \
    --learning_rate 5e-7
```

### 3. Generate Images

Test your trained LoRA model:

```bash
# Generate with your custom character
python simple_flux_generator.py \
    --prompt "your_character, walking on a beach at sunset" \
    --lora_path ./models/your_character_lora/working_flux_lora_final.pt \
    --lora_scale 1.0 \
    --width 512 \
    --height 512 \
    --num_images 4

# Generate comparison (with/without LoRA)
python simple_flux_generator.py \
    --prompt "your_character, portrait in studio lighting" \
    --lora_path ./models/your_character_lora/working_flux_lora_final.pt \
    --comparison \
    --num_images 4
```

## 🎛️ LoRA Training Strategies

| Trainer | Speed | Stability | Memory Usage | Recommended For |
|---------|-------|-----------|--------------|-----------------|
| `minimal_flux_lora_trainer.py` | ⚡⚡⚡ | ⭐⭐ | 🟢 Low | Quick tests |
| `working_flux_lora_trainer.py` | ⚡⚡ | ⭐⭐⭐⭐ | 🟡 Medium | **Production** |
| `ultra_safe_flux_lora_trainer.py` | ⚡ | ⭐⭐⭐⭐⭐ | 🟢 Low | Critical projects |
| `optimized_flux_lora_trainer.py` | ⚡⚡ | ⭐⭐⭐ | 🟢 Low | Limited VRAM |
| `aggressive_flux_lora_trainer.py` | ⚡⚡⚡ | ⭐⭐ | 🔴 High | Fast iteration |
| `ultimate_flux_lora_trainer.py` | ⚡ | ⭐⭐⭐ | 🔴 High | Maximum effect |

## 🎬 Planned Features

### Scene System
- **Environment Definition**: Weather, lighting, location, mood
- **Scene Transitions**: Smooth transitions between different environments
- **Template Library**: Pre-built scene templates for common music video styles

### Shot System
- **Shot Types**: Close-up, wide shot, tracking shot, etc.
- **Motion Descriptions**: Camera movement and subject action
- **Timing Synchronization**: Beat-matched shot changes

### Advanced Features
- **Music Analysis**: Automatic beat detection and tempo analysis
- **Style Transfer**: Apply artistic styles to match music genre
- **Multi-Character**: Support for multiple characters in scenes
- **Video Export**: Export final music video in various formats

## 💡 Usage Examples

### Character Training Workflow

```bash
# 1. Prepare dataset
mkdir -p dataset/my_artist
# Copy 20-30 images of your character to dataset/my_artist/

# 2. Generate captions
python blip2_caption_generator.py \
    --input_dir dataset/my_artist \
    --character_name "my_artist"

# 3. Train LoRA
python working_flux_lora_trainer.py \
    --dataset_path dataset/my_artist \
    --output_dir models/my_artist_lora \
    --epochs 25

# 4. Test generation
python simple_flux_generator.py \
    --prompt "my_artist, performing on stage with dramatic lighting" \
    --lora_path models/my_artist_lora/working_flux_lora_final.pt \
    --num_images 4
```

### Scene-Based Generation (Future)

```python
# Planned API for scene/shot hierarchy
from music_video_generator import MusicVideoGenerator

generator = MusicVideoGenerator()

# Define scenes
beach_scene = generator.create_scene(
    environment="beach at sunset, golden hour lighting",
    mood="romantic, warm, peaceful"
)

concert_scene = generator.create_scene(
    environment="concert stage, dramatic lighting, crowd",
    mood="energetic, powerful, exciting"
)

# Define shots within scenes
beach_scene.add_shot(
    starting_image="my_artist, walking towards camera",
    motion="slow zoom in, hair flowing in wind",
    duration=4.0  # seconds
)

# Generate video
video = generator.create_music_video(
    scenes=[beach_scene, concert_scene],
    music_file="song.mp3",
    character_lora="models/my_artist_lora/final.pt"
)
```

## 🛡️ Memory Management

The project includes several memory optimization strategies:

- **CPU Offloading**: Automatically manages GPU/CPU memory usage
- **Gradient Checkpointing**: Reduces memory during training
- **Attention Slicing**: Reduces VRAM requirements for generation
- **VAE Tiling**: Enables higher resolution generation on limited VRAM

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU offload trainer
   python ultra_safe_flux_lora_trainer.py --dataset_path your_dataset
   
   # Or reduce batch size and resolution
   python working_flux_lora_trainer.py --batch_size 1 --dataset_path your_dataset
   ```

2. **LoRA Not Loading**
   ```bash
   # Check LoRA compatibility
   python simple_flux_generator.py --prompt "test" --lora_path your_lora.pt
   ```

3. **Training Instability**
   ```bash
   # Use ultra-safe trainer with conservative settings
   python ultra_safe_flux_lora_trainer.py \
       --learning_rate 5e-7 \
       --epochs 50 \
       --dataset_path your_dataset
   ```

## 📊 Performance Requirements

### Minimum System Requirements
- **GPU**: 6GB VRAM (GTX 1060 6GB / RTX 2060)
- **RAM**: 16GB system RAM
- **Storage**: 10GB free space

### Recommended System Requirements
- **GPU**: 12GB+ VRAM (RTX 3080 / RTX 4070 or better)
- **RAM**: 32GB system RAM
- **Storage**: 50GB+ free space for models and outputs

## 🤝 Contributing

This project is in early development. Contributions are welcome for:

- Scene/Shot hierarchy implementation
- Video generation pipeline
- Music synchronization features
- Performance optimizations
- Bug fixes and improvements

## 📄 License

[License information to be added]

## 🔗 Dependencies

Key dependencies:
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face diffusion models library
- **Transformers**: Model loading and text processing
- **Accelerate**: Training acceleration
- **Pillow**: Image processing
- **NumPy**: Numerical computations

See `requirements.txt` for complete dependency list.

---

**Note**: This project is under active development. The Scene/Shot hierarchy system and video generation features are planned for future releases. Current focus is on perfecting character training and image generation capabilities.

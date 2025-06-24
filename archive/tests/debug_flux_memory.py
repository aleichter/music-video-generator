import torch
import psutil
import GPUtil
from diffusers import FluxPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_system_resources():
    """Check current system resources"""
    
    # GPU Memory
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        logger.info(f"🎮 GPU: {gpu.name}")
        logger.info(f"💾 Total VRAM: {gpu.memoryTotal}MB")
        logger.info(f"💾 Used VRAM: {gpu.memoryUsed}MB") 
        logger.info(f"💾 Free VRAM: {gpu.memoryFree}MB")
        
        # PyTorch CUDA memory
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"🔥 PyTorch Allocated: {allocated:.2f}GB")
        logger.info(f"🔥 PyTorch Reserved: {reserved:.2f}GB")
    
    # System RAM
    ram = psutil.virtual_memory()
    logger.info(f"🧠 Total RAM: {ram.total / 1024**3:.2f}GB")
    logger.info(f"🧠 Used RAM: {ram.used / 1024**3:.2f}GB")
    logger.info(f"🧠 Available RAM: {ram.available / 1024**3:.2f}GB")

def test_flux_loading():
    """Test FLUX loading and basic operations"""
    
    logger.info("🚀 Testing FLUX pipeline loading...")
    
    check_system_resources()
    
    try:
        # Load with minimal settings
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16,
        )
        
        logger.info("✅ Pipeline loaded successfully")
        check_system_resources()
        
        # Try moving to GPU
        pipe = pipe.to("cuda")
        logger.info("✅ Moved to GPU")
        check_system_resources()
        
        # Test a simple generation
        logger.info("🎨 Testing simple generation...")
        with torch.inference_mode():
            result = pipe(
                "a simple test image",
                width=512,
                height=512,
                num_inference_steps=1,  # Minimal steps
                guidance_scale=0.0,
            )
        
        logger.info("✅ Generation successful!")
        check_system_resources()
        
        return pipe
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        check_system_resources()
        raise

def test_lora_weight_behavior():
    """Test LoRA weight initialization and basic math"""
    
    logger.info("🧮 Testing LoRA weight behavior...")
    
    # Test basic LoRA math
    rank = 4
    dim = 3072
    
    # Initialize like our trainer
    lora_A = torch.zeros(rank, dim, dtype=torch.float16, device='cuda')
    lora_B = torch.zeros(dim, rank, dtype=torch.float16, device='cuda')
    
    # Test 1: Pure zeros
    product = lora_B @ lora_A
    logger.info(f"Pure zeros product: min={product.min():.15f}, max={product.max():.15f}")
    
    # Test 2: Tiny values like our trainer
    with torch.no_grad():
        lora_A[0, 0] = 1e-6
        lora_B[0, 0] = 1e-6
    
    product = lora_B @ lora_A
    logger.info(f"Tiny values product: min={product.min():.15f}, max={product.max():.15f}")
    logger.info(f"Product[0,0]: {product[0,0]:.15f}")
    
    # Test 3: Check for float16 precision issues
    logger.info(f"lora_A[0,0]: {lora_A[0,0]:.15f}")
    logger.info(f"lora_B[0,0]: {lora_B[0,0]:.15f}")
    logger.info(f"Are they actually non-zero? A: {lora_A[0,0] != 0}, B: {lora_B[0,0] != 0}")
    
    # Test 4: Larger values
    with torch.no_grad():
        lora_A[0, 0] = 0.001
        lora_B[0, 0] = 0.001
    
    product = lora_B @ lora_A
    logger.info(f"Larger values product: min={product.min():.15f}, max={product.max():.15f}")
    logger.info(f"Product[0,0]: {product[0,0]:.15f}")
    
    # Test 5: Check gradients
    lora_A.requires_grad = True
    lora_B.requires_grad = True
    
    loss = torch.sum(product)
    loss.backward()
    
    logger.info(f"Gradient A max: {lora_A.grad.max():.15f}")
    logger.info(f"Gradient B max: {lora_B.grad.max():.15f}")
    logger.info(f"Any NaN in gradients? A: {torch.isnan(lora_A.grad).any()}, B: {torch.isnan(lora_B.grad).any()}")

def test_transformer_access():
    """Test accessing transformer layers"""
    
    logger.info("🔍 Testing transformer access...")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    transformer = pipe.transformer
    logger.info(f"Transformer type: {type(transformer)}")
    
    # Find the target layer
    target_layer = None
    for name, module in transformer.named_modules():
        if 'transformer_blocks.0.attn.to_q' in name:
            target_layer = module
            logger.info(f"Found target layer: {name}")
            logger.info(f"Layer type: {type(module)}")
            logger.info(f"Weight shape: {module.weight.shape}")
            logger.info(f"Weight dtype: {module.weight.dtype}")
            logger.info(f"Weight device: {module.weight.device}")
            break
    
    if target_layer is None:
        logger.error("❌ Could not find target layer!")
        # List all available layers
        logger.info("Available layers:")
        for name, module in transformer.named_modules():
            if isinstance(module, torch.nn.Linear) and 'attn' in name:
                logger.info(f"  {name}: {module.weight.shape}")
    
    return pipe

if __name__ == "__main__":
    print("🔬 FLUX Debug Analysis on RTX 4090")
    print("=" * 50)
    
    try:
        # Step 1: Check resources
        check_system_resources()
        print()
        
        # Step 2: Test FLUX loading
        pipe = test_flux_loading()
        print()
        
        # Step 3: Test LoRA math
        test_lora_weight_behavior()
        print()
        
        # Step 4: Test transformer access
        test_transformer_access()
        print()
        
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
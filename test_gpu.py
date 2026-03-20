import torch

print("="*60)
print("GPU VERIFICATION TEST")
print("="*60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"\nGPU Device Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Get GPU memory info
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {gpu_props.total_memory / 1e9:.2f} GB")
    print(f"GPU Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Test GPU with a simple tensor operation
    print("\n" + "="*60)
    print("Testing GPU with tensor operations...")
    print("="*60)
    
    # Create tensor on GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    
    # Perform operation
    z = torch.matmul(x, y)
    
    print(f"✓ Tensor created on GPU: {x.device}")
    print(f"✓ Matrix multiplication successful!")
    
    print("\n" + "="*60)
    print(" GPU IS WORKING!")
    print("="*60)
    print("\nYour RTX 3050 Laptop GPU is ready for training!")
    print("Expected training time: ~15-30 minutes (vs 2-3 hours on CPU)")
    
else:
    print("\n" + "="*60)
    print(" GPU NOT DETECTED")
    print("="*60)
    print("\nPossible issues:")
    print("1. PyTorch CPU version is still installed")
    print("2. CUDA toolkit not properly installed")
    print("3. Driver version mismatch")
    print("\nYou can still train on CPU, but it will be slower.")

print("\n" + "="*60)
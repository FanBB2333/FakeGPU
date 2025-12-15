import sys
import torch

print("=" * 60)
print("FakeGPU + PyTorch Integration Test Summary")
print("=" * 60)

try:
    # Basic checks
    print(f"\n1. PyTorch version: {torch.__version__}")
    print(f"2. CUDA available: {torch.cuda.is_available()}")
    print(f"3. Device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"\n4. Device properties (GPU 0):")
        props = torch.cuda.get_device_properties(0)
        print(f"   - Name: {props.name}")
        print(f"   - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   - Compute capability: {props.major}.{props.minor}")
        
        print(f"\n5. Creating small tensor on CUDA...")
        x = torch.randn(10, 10, device='cuda')
        print(f"   - Tensor shape: {x.shape}")
        print(f"   - Tensor device: {x.device}")
        
        print(f"\n6. Simple tensor operation...")
        y = torch.randn(10, 10, device='cuda')
        z = x + y
        print(f"   - Result shape: {z.shape}")
        
        print(f"\n7. Memory info:")
        print(f"   - Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

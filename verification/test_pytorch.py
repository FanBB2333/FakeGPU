#!/usr/bin/env python3
"""
Test fake GPU with PyTorch (requires CUDA Runtime API interception)
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fakegpu

result = fakegpu.init(force=True, update_env=True)
print(f"[Python] FakeGPU initialized from: {result.lib_dir}")

try:
    import torch
except ImportError as e:
    print(f"ERROR: PyTorch not installed or failed to import: {e}")
    print("Install with: pip install torch")
    sys.exit(1)

try:
    print("=== Testing GPU Detection with PyTorch ===")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"[PyTorch] CUDA Available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"[PyTorch] Device Count: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n[GPU {i}]")
            print(f"  Name: {props.name}")
            print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"  Major: {props.major}, Minor: {props.minor}")

        # Try to allocate memory
        print("\n[PyTorch] Testing memory allocation...")
        device = torch.device("cuda:0")
        tensor = torch.zeros(1000, 1000, device=device)
        print(f"[PyTorch] Allocated tensor of shape {tensor.shape} on {tensor.device}")
        print(f"[PyTorch] Memory allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

        del tensor
        torch.cuda.empty_cache()
        print("[PyTorch] Memory freed")

        print("\n[PyTorch] Test completed successfully!")
    else:
        print("[PyTorch] CUDA not available - this is expected if PyTorch has no CUDA support or FakeGPU was initialized too late")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

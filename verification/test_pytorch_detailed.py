#!/usr/bin/env python3
"""
Detailed PyTorch test to check which operations use fake GPU
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fakegpu

result = fakegpu.init(force=True, update_env=True)
print(f"[Test] FakeGPU initialized from: {result.lib_dir}")

try:
    import torch
except ImportError as e:
    print(f"ERROR: PyTorch not installed: {e}")
    sys.exit(1)

print("=" * 70)
print("PyTorch Detailed GPU Test")
print("=" * 70)
print()

# Test 1: Check CUDA availability
print("Test 1: CUDA Availability")
print("-" * 70)
cuda_available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {cuda_available}")
print()

if not cuda_available:
    print("CUDA not available. This might mean:")
    print("1. PyTorch was built without CUDA support")
    print("2. No CUDA runtime library was found")
    print("3. The fake GPU library is not being used")
    sys.exit(0)

# Test 2: Device count
print("Test 2: Device Count")
print("-" * 70)
device_count = torch.cuda.device_count()
print(f"torch.cuda.device_count(): {device_count}")
print()

# Test 3: Device properties
print("Test 3: Device Properties")
print("-" * 70)
for i in range(min(device_count, 3)):
    props = torch.cuda.get_device_properties(i)
    print(f"Device {i}:")
    print(f"  Name: {props.name}")
    print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Multi Processor Count: {props.multi_processor_count}")
    print()

# Test 4: Memory allocation
print("Test 4: Memory Allocation")
print("-" * 70)
try:
    device = torch.device("cuda:0")
    print(f"Creating tensor on {device}...")

    # Small tensor
    tensor1 = torch.zeros(100, 100, device=device)
    print(f"✓ Created tensor of shape {tensor1.shape}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # Larger tensor
    tensor2 = torch.randn(1000, 1000, device=device)
    print(f"✓ Created tensor of shape {tensor2.shape}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # Free memory
    del tensor1, tensor2
    torch.cuda.empty_cache()
    print(f"✓ Freed tensors")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    print()

except Exception as e:
    print(f"✗ Memory allocation failed: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 5: Simple computation
print("Test 5: Simple Computation")
print("-" * 70)
try:
    device = torch.device("cuda:0")

    # Matrix multiplication
    a = torch.randn(100, 100, device=device)
    b = torch.randn(100, 100, device=device)
    print(f"Created two 100x100 matrices on {device}")

    c = torch.matmul(a, b)
    print(f"✓ Matrix multiplication completed")
    print(f"  Result shape: {c.shape}")
    print(f"  Result device: {c.device}")
    print()

except Exception as e:
    print(f"✗ Computation failed: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 6: Data transfer
print("Test 6: Data Transfer (CPU <-> GPU)")
print("-" * 70)
try:
    # CPU to GPU
    cpu_tensor = torch.randn(50, 50)
    print(f"Created tensor on CPU: {cpu_tensor.device}")

    gpu_tensor = cpu_tensor.to('cuda:0')
    print(f"✓ Transferred to GPU: {gpu_tensor.device}")

    # GPU to CPU
    cpu_tensor2 = gpu_tensor.to('cpu')
    print(f"✓ Transferred back to CPU: {cpu_tensor2.device}")
    print()

except Exception as e:
    print(f"✗ Data transfer failed: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 7: Check current device
print("Test 7: Device Management")
print("-" * 70)
try:
    current_device = torch.cuda.current_device()
    print(f"Current device: {current_device}")

    torch.cuda.set_device(0)
    print(f"✓ Set device to 0")

    current_device = torch.cuda.current_device()
    print(f"Current device after set: {current_device}")
    print()

except Exception as e:
    print(f"✗ Device management failed: {e}")
    print()

# Test 8: Memory info
print("Test 8: Memory Information")
print("-" * 70)
try:
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    print(f"Memory allocated: {allocated / (1024**2):.2f} MB")
    print(f"Memory reserved: {reserved / (1024**2):.2f} MB")
    print()

except Exception as e:
    print(f"✗ Memory info failed: {e}")
    print()

print("=" * 70)
print("Test Summary")
print("=" * 70)
print()
print("If you see fake GPU information above (e.g., 'Fake NVIDIA A100-SXM4-80GB'),")
print("then PyTorch is successfully using the fake GPU library.")
print()
print("If you see real GPU information (e.g., 'RTX 3090 Ti'), then PyTorch is")
print("using the real GPU drivers, but some operations (like cudaMalloc) may")
print("still be intercepted by the fake GPU library.")
print()
print("Check the console output for [FakeCUDA] messages to see which operations")
print("were intercepted by the fake GPU library.")
print()

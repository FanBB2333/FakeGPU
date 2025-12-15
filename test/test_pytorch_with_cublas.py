#!/usr/bin/env python3
"""
Test PyTorch with FakeGPU including cuBLAS stub for matrix operations.
This test should now succeed with matrix multiplication operations.
"""

import torch
import sys

def test_basic_operations():
    """Test basic GPU detection and operations"""
    print("=" * 60)
    print("Test 1: GPU Detection")
    print("=" * 60)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("ERROR: CUDA not detected!")
        return False

    # Get device count
    device_count = torch.cuda.device_count()
    print(f"GPU Count: {device_count}")

    # Print device properties
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessor Count: {props.multi_processor_count}")

    return True

def test_tensor_operations():
    """Test basic tensor creation and operations"""
    print("\n" + "=" * 60)
    print("Test 2: Tensor Operations")
    print("=" * 60)

    device = torch.device('cuda:0')

    # Create tensors
    print("\nCreating tensors...")
    x = torch.randn(10, 10, device=device)
    y = torch.randn(10, 10, device=device)

    print(f"Tensor x shape: {x.shape}, device: {x.device}")
    print(f"Tensor y shape: {y.shape}, device: {y.device}")

    # Element-wise addition
    print("\nElement-wise addition...")
    z = x + y
    print(f"Result shape: {z.shape}")
    print(f"Sample values: {z[0, :5]}")

    return True

def test_matrix_multiplication():
    """Test matrix multiplication (requires cuBLAS)"""
    print("\n" + "=" * 60)
    print("Test 3: Matrix Multiplication (cuBLAS)")
    print("=" * 60)

    device = torch.device('cuda:0')

    try:
        # Small matrix multiplication
        print("\nSmall matrix multiplication (10x10)...")
        a = torch.randn(10, 10, device=device)
        b = torch.randn(10, 10, device=device)
        c = torch.matmul(a, b)

        print(f"Result shape: {c.shape}")
        print(f"Sample values: {c[0, :5]}")

        # Larger matrix multiplication
        print("\nLarger matrix multiplication (100x100)...")
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        c = torch.matmul(a, b)

        print(f"Result shape: {c.shape}")
        print(f"Sample values: {c[0, :5]}")

        # Batch matrix multiplication
        print("\nBatch matrix multiplication (32x50x50)...")
        a = torch.randn(32, 50, 50, device=device)
        b = torch.randn(32, 50, 50, device=device)
        c = torch.matmul(a, b)

        print(f"Result shape: {c.shape}")
        print(f"Sample values: {c[0, 0, :5]}")

        print("\n✓ Matrix multiplication tests PASSED!")
        return True

    except Exception as e:
        print(f"\n✗ Matrix multiplication FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_operations():
    """Test memory transfer operations"""
    print("\n" + "=" * 60)
    print("Test 4: Memory Operations")
    print("=" * 60)

    device = torch.device('cuda:0')

    # CPU to GPU
    print("\nCPU -> GPU transfer...")
    x_cpu = torch.randn(100, 100)
    x_gpu = x_cpu.to(device)
    print(f"Transferred tensor shape: {x_gpu.shape}, device: {x_gpu.device}")

    # GPU to CPU
    print("\nGPU -> CPU transfer...")
    y_cpu = x_gpu.cpu()
    print(f"Transferred tensor shape: {y_cpu.shape}, device: {y_cpu.device}")

    return True

def test_memory_tracking():
    """Test memory allocation tracking"""
    print("\n" + "=" * 60)
    print("Test 5: Memory Tracking")
    print("=" * 60)

    device = torch.device('cuda:0')

    # Allocate some memory
    print("\nAllocating memory...")
    tensors = []
    for i in range(5):
        size = (i+1) * 100
        t = torch.randn(size, size, device=device)
        tensors.append(t)

        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"  Tensor {i+1} ({size}x{size}): Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")

    # Clear memory
    print("\nClearing memory...")
    tensors.clear()
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"After cleanup: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")

    return True

def main():
    """Run all tests"""
    print("FakeGPU + cuBLAS Integration Test")
    print("=" * 60)

    tests = [
        ("GPU Detection", test_basic_operations),
        ("Tensor Operations", test_tensor_operations),
        ("Matrix Multiplication", test_matrix_multiplication),
        ("Memory Operations", test_memory_operations),
        ("Memory Tracking", test_memory_tracking),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())

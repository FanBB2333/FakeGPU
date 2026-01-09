#!/usr/bin/env python3
"""
Test script for FakeGPU passthrough mode.

This script tests that passthrough mode correctly forwards CUDA calls to the real GPU.
It compares results between:
1. Direct CUDA execution (no FakeGPU)
2. FakeGPU passthrough mode
3. FakeGPU simulate mode

Usage:
    # Test passthrough mode (requires real GPU)
    FAKEGPU_MODE=passthrough python test_passthrough.py

    # Test simulate mode (no GPU required)
    FAKEGPU_MODE=simulate python test_passthrough.py
"""

import os
import sys
import json

def test_cuda_available():
    """Test if CUDA is available and get device info."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available")
            return None

        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")

        devices = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "index": i,
                "name": props.name,
                "total_memory": props.total_memory,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
            devices.append(device_info)
            print(f"  Device {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    SM count: {props.multi_processor_count}")

        return devices
    except ImportError:
        print("PyTorch not installed")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_memory_allocation():
    """Test memory allocation and deallocation."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available for memory test")
            return False

        print("\nTesting memory allocation...")

        # Get initial memory
        torch.cuda.synchronize()
        initial_allocated = torch.cuda.memory_allocated()
        initial_reserved = torch.cuda.memory_reserved()
        print(f"  Initial allocated: {initial_allocated / (1024**2):.2f} MB")
        print(f"  Initial reserved: {initial_reserved / (1024**2):.2f} MB")

        # Allocate some memory
        sizes = [1024*1024, 10*1024*1024, 100*1024*1024]  # 1MB, 10MB, 100MB
        tensors = []

        for size in sizes:
            num_elements = size // 4  # float32 = 4 bytes
            t = torch.zeros(num_elements, dtype=torch.float32, device='cuda')
            tensors.append(t)
            allocated = torch.cuda.memory_allocated()
            print(f"  After allocating {size / (1024**2):.1f} MB: {allocated / (1024**2):.2f} MB allocated")

        # Free memory
        del tensors
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_allocated = torch.cuda.memory_allocated()
        print(f"  After freeing: {final_allocated / (1024**2):.2f} MB allocated")

        return True
    except Exception as e:
        print(f"Memory test error: {e}")
        return False


def test_simple_computation():
    """Test simple GPU computation."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available for computation test")
            return None

        print("\nTesting simple computation...")

        # Create tensors
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')

        # Matrix multiplication
        torch.cuda.synchronize()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Get result checksum
        checksum = c.sum().item()
        print(f"  Matrix multiplication result checksum: {checksum:.6f}")

        # Test with fixed seed for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        a_fixed = torch.randn(100, 100, device='cuda')
        b_fixed = torch.randn(100, 100, device='cuda')
        c_fixed = torch.matmul(a_fixed, b_fixed)
        fixed_checksum = c_fixed.sum().item()
        print(f"  Fixed seed result checksum: {fixed_checksum:.6f}")

        return {
            "random_checksum": checksum,
            "fixed_checksum": fixed_checksum,
        }
    except Exception as e:
        print(f"Computation test error: {e}")
        return None


def test_memcpy():
    """Test memory copy operations."""
    try:
        import torch
        import numpy as np

        if not torch.cuda.is_available():
            print("CUDA not available for memcpy test")
            return False

        print("\nTesting memory copy...")

        # Host to Device
        host_data = np.random.randn(1000000).astype(np.float32)
        device_tensor = torch.from_numpy(host_data).cuda()
        print(f"  H2D: Copied {host_data.nbytes / (1024**2):.2f} MB")

        # Device to Host
        host_result = device_tensor.cpu().numpy()
        print(f"  D2H: Copied {host_result.nbytes / (1024**2):.2f} MB")

        # Verify data integrity
        if np.allclose(host_data, host_result):
            print("  Data integrity: PASS")
            return True
        else:
            print("  Data integrity: FAIL")
            return False
    except Exception as e:
        print(f"Memcpy test error: {e}")
        return False


def check_report():
    """Check the generated FakeGPU report."""
    report_path = os.environ.get("FAKEGPU_REPORT_PATH", "fake_gpu_report.json")

    if not os.path.exists(report_path):
        print(f"\nReport file not found: {report_path}")
        return None

    print(f"\nChecking report: {report_path}")
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)

        print(f"  Report version: {report.get('report_version', 'unknown')}")
        print(f"  Mode: {report.get('mode', 'unknown')}")

        if 'oom_policy' in report:
            print(f"  OOM policy: {report['oom_policy']}")

        if 'hybrid_stats' in report:
            stats = report['hybrid_stats']
            print("  Hybrid stats:")
            print(f"    Real alloc: {stats.get('real_alloc', {}).get('count', 0)} calls, "
                  f"{stats.get('real_alloc', {}).get('bytes', 0) / (1024**2):.2f} MB")
            print(f"    Managed alloc: {stats.get('managed_alloc', {}).get('count', 0)} calls")
            print(f"    Spilled alloc: {stats.get('spilled_alloc', {}).get('count', 0)} calls")

        if 'backing_gpus' in report:
            print("  Backing GPUs:")
            for gpu in report['backing_gpus']:
                print(f"    GPU {gpu['index']}: {gpu['total_memory'] / (1024**3):.2f} GB total, "
                      f"{gpu['used_memory'] / (1024**3):.2f} GB used")

        summary = report.get('summary', {})
        print(f"  Device count: {summary.get('device_count', 0)}")

        return report
    except Exception as e:
        print(f"Error reading report: {e}")
        return None


def main():
    mode = os.environ.get("FAKEGPU_MODE", "simulate")
    print(f"FakeGPU Mode: {mode}")
    print("=" * 60)

    # Run tests
    devices = test_cuda_available()
    if devices is None and mode == "passthrough":
        print("\nWARNING: Passthrough mode requires a real GPU!")
        sys.exit(1)

    if devices:
        test_memory_allocation()
        test_simple_computation()
        test_memcpy()

    # Check report
    check_report()

    print("\n" + "=" * 60)
    print("Tests completed!")


if __name__ == "__main__":
    main()

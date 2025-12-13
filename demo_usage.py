#!/usr/bin/env python3
"""
Demo: run fakeGPU to emulate NVIDIA GPUs.

Usage (run with LD_PRELOAD):
    LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test all
    LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test nvml --max-devices 2
    LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test cuda --alloc-size 256
    LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test pytorch

What it does (simple view):
1) NVML via pynvml: report device count and a few device names.
2) CUDA Runtime via ctypes: allocate and free a small buffer.
3) Optional PyTorch check: show CUDA availability and device count.
"""

import os
import sys
import argparse
from ctypes import CDLL, c_int, c_void_p, c_size_t, POINTER, byref


def load_fake_gpu_library(lib_path):
    """Load the fake GPU shared library.

    Note: When using LD_PRELOAD, the library is already loaded.
    We only need to get a handle to it for direct CUDA calls.
    """
    if not os.path.exists(lib_path):
        print(f"Error: library not found at {lib_path}")
        print("Build first: cmake --build build")
        sys.exit(1)

    # Check if library is already loaded via LD_PRELOAD
    if 'LD_PRELOAD' in os.environ and lib_path in os.environ['LD_PRELOAD']:
        print(f"Fake GPU library already loaded via LD_PRELOAD: {lib_path}")
        # Return None to indicate we don't need to load it
        # The CUDA functions will be available via LD_PRELOAD
        return None

    print(f"Loading fake GPU library: {lib_path}")
    # If not preloaded, load it normally
    return CDLL(lib_path)


def test_pynvml(max_devices=None):
    """Scenario 1: list GPUs using pynvml (NVML API)."""
    print("Scenario 1: NVML (pynvml)")
    print("-" * 70)

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"Detected {device_count} GPU device(s)")

        display_count = min(device_count, max_devices) if max_devices else device_count
        for i in range(display_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            # Handle both bytes (older pynvml) and str (newer pynvml)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            print(f"GPU {i}: {name}")

        if max_devices and device_count > max_devices:
            print(f"... {device_count - max_devices} more device(s) not shown")

        pynvml.nvmlShutdown()
        print("✓ NVML check finished")
        return True

    except ImportError:
        print("pynvml not installed; skip (pip install nvidia-ml-py3)")
        return False
    except Exception as e:
        print(f"✗ NVML check failed: {e}")
        return False
    finally:
        print()


def test_cuda_runtime(fake_gpu, alloc_size_mb=100):
    """Scenario 2: basic CUDA runtime calls via ctypes."""
    print("Scenario 2: CUDA Runtime (ctypes)")
    print("-" * 70)

    try:
        # If fake_gpu is None, the library was loaded via LD_PRELOAD
        # We need to load it explicitly for direct function calls
        if fake_gpu is None:
            import ctypes
            fake_gpu = ctypes.CDLL(None)  # Load the main program's symbols

        cudaGetDeviceCount = fake_gpu.cudaGetDeviceCount
        cudaGetDeviceCount.argtypes = [POINTER(c_int)]
        cudaGetDeviceCount.restype = c_int

        cudaMalloc = fake_gpu.cudaMalloc
        cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        cudaMalloc.restype = c_int

        cudaFree = fake_gpu.cudaFree
        cudaFree.argtypes = [c_void_p]
        cudaFree.restype = c_int

        device_count = c_int()
        result = cudaGetDeviceCount(byref(device_count))
        print(f"cudaGetDeviceCount -> {device_count.value} device(s)")

        size = 1024 * 1024 * alloc_size_mb
        device_ptr = c_void_p()
        result = cudaMalloc(byref(device_ptr), size)
        if result == 0:
            print(f"✓ cudaMalloc allocated {size / (1024**2):.2f} MB")
            print(f"  device pointer: 0x{device_ptr.value:x}")

            result = cudaFree(device_ptr)
            if result == 0:
                print("✓ cudaFree released memory")
        else:
            print(f"✗ cudaMalloc failed, error code: {result}")

        print("✓ CUDA runtime check finished")
        return True

    except Exception as e:
        print(f"✗ CUDA runtime check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print()


def test_pytorch():
    """Scenario 3: quick PyTorch CUDA check."""
    print("Scenario 3: PyTorch")
    print("-" * 70)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print("Note: detection may use real drivers; allocations are intercepted by fakeGPU.")
            return True
        else:
            print("PyTorch did not detect CUDA support.")
            return False

    except ImportError:
        print("PyTorch not installed; skip (pip install torch)")
        return False
    except Exception as e:
        print(f"PyTorch check failed: {e}")
        return False
    finally:
        print()


def print_usage_summary():
    """Print a brief usage guide."""
    print("=" * 70)
    print("How to run")
    print("=" * 70)
    print("LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test all")
    print("LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test nvml --max-devices 2")
    print("LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test cuda --alloc-size 256")
    print("LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test pytorch")
    print("Report file: fake_gpu_report.json (written at program exit)")
    print()


def main():
    """Entry point for running the simplified demo."""
    parser = argparse.ArgumentParser(
        description='FakeGPU demo - simple NVML, CUDA, and PyTorch checks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test all
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test nvml --max-devices 2
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test cuda --alloc-size 256
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test pytorch
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --no-summary
        """
    )

    parser.add_argument(
        '--lib-path',
        default=os.environ.get('FAKE_GPU_LIB', './build/libfake_gpu.so'),
        help='Path to libfake_gpu.so (default: ./build/libfake_gpu.so or $FAKE_GPU_LIB)'
    )

    parser.add_argument(
        '--test',
        choices=['all', 'nvml', 'cuda', 'pytorch'],
        default='all',
        help='Select which test to run (default: all)'
    )

    parser.add_argument(
        '--max-devices',
        type=int,
        metavar='N',
        help='Max devices to display for NVML test'
    )

    parser.add_argument(
        '--alloc-size',
        type=int,
        default=100,
        metavar='MB',
        help='Allocation size in MB for CUDA test (default: 100)'
    )

    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Do not print usage guide at startup'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode: only show test results'
    )

    args = parser.parse_args()

    if not args.quiet:
        print_usage_summary()
        print("=" * 70)
        print("FakeGPU Demo")
        print("=" * 70)
        print()

    fake_gpu = load_fake_gpu_library(args.lib_path)
    if not args.quiet:
        print()

    results = {}

    if args.test in ['all', 'nvml']:
        results['nvml'] = test_pynvml(max_devices=args.max_devices)

    if args.test in ['all', 'cuda']:
        results['cuda'] = test_cuda_runtime(fake_gpu, alloc_size_mb=args.alloc_size)

    if args.test in ['all', 'pytorch']:
        results['pytorch'] = test_pytorch()

    if not args.quiet and len(results) > 1:
        print("=" * 70)
        print("Test summary")
        print("=" * 70)
        for test_name, success in results.items():
            status = "✓ pass" if success else "✗ fail/skip"
            print(f"{test_name.upper()}: {status}")
        print()


if __name__ == '__main__':
    main()

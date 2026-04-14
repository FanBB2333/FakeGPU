#!/usr/bin/env python3
"""
Demo: run FakeGPU to emulate NVIDIA GPUs.

Recommended usage:
    ./fgpu python demo_usage.py --test all
    ./fgpu python demo_usage.py --test nvml --max-devices 2
    ./fgpu python demo_usage.py --test cuda --alloc-size 256
    ./fgpu python demo_usage.py --test pytorch
    python3 demo_usage.py --test transformer

What it does (simple view):
1) NVML via pynvml: report device count and a few device names.
2) CUDA Runtime via ctypes: allocate and free a small buffer.
3) Optional PyTorch check: show CUDA availability and device count.
4) Optional tiny Transformer training loop using fake CUDA semantics.
"""

import os
import sys
import argparse
from pathlib import Path
from ctypes import c_int, c_void_p, c_size_t, POINTER, byref

import fakegpu


def _default_lib_path() -> str:
    return './build/libfake_gpu.dylib' if sys.platform == 'darwin' else './build/libfake_gpu.so'


def _cudart_lib_name() -> str:
    return 'libcudart.dylib' if sys.platform == 'darwin' else 'libcudart.so.12'


def _nvml_lib_name() -> str:
    return 'libnvidia-ml.dylib' if sys.platform == 'darwin' else 'libnvidia-ml.so.1'


def load_fake_gpu_library(lib_path):
    """Initialize FakeGPU and return the init result plus a CUDA Runtime handle."""
    candidate = Path(lib_path).resolve()
    lib_dir = candidate.parent if candidate.suffix else candidate

    if not lib_dir.exists():
        print(f"Error: library directory not found at {lib_dir}")
        print("Build first: cmake --build build")
        sys.exit(1)

    result = fakegpu.init(lib_dir=lib_dir, force=True, update_env=True)
    print(f"FakeGPU initialized from: {result.lib_dir}")
    return result, result.handles[_cudart_lib_name()]


def test_pynvml(nvml_handle, max_devices=None):
    """Scenario 1: list GPUs using pynvml (NVML API)."""
    print("Scenario 1: NVML (pynvml)")
    print("-" * 70)

    try:
        import pynvml

        def _load_fake_nvml_library():
            pynvml.nvmlLib = nvml_handle
            return pynvml.nvmlLib

        pynvml.nvmlLib = None
        pynvml._LoadNvmlLibrary = _load_fake_nvml_library

        pynvml.nvmlInit()

        # Query version information
        print("\nVersion Information:")
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            print(f"  Driver Version: {driver_version}")
        except Exception as e:
            print(f"  Driver Version: (error: {e})")

        try:
            nvml_version = pynvml.nvmlSystemGetNVMLVersion()
            print(f"  NVML Version: {nvml_version}")
        except Exception as e:
            print(f"  NVML Version: (error: {e})")

        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            # CUDA version is returned as an integer (e.g., 12080 for CUDA 12.8)
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            print(f"  CUDA Driver Version: {major}.{minor} (raw: {cuda_version})")
        except Exception as e:
            print(f"  CUDA Driver Version: (error: {e})")

        print()
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


def test_transformer_training():
    """Scenario 4: run a tiny Transformer training loop."""
    print("Scenario 4: Tiny Transformer training")
    print("-" * 70)

    try:
        from fakegpu.torch_patch import patch

        patch()

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.manual_seed(0)

        device = torch.device("cuda:0")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

        class TinyTransformerLM(nn.Module):
            def __init__(self, vocab_size=256, d_model=64, nhead=4, num_layers=2, max_len=32):
                super().__init__()
                self.token_emb = nn.Embedding(vocab_size, d_model)
                self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=128,
                    dropout=0.0,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
                self.ln = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, input_ids):
                x = self.token_emb(input_ids) + self.pos_emb[: input_ids.size(1)]
                x = self.encoder(x)
                x = self.ln(x)
                return self.head(x)

        model = TinyTransformerLM().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size = 4
        seq_len = 16
        steps = 5

        for step in range(1, steps + 1):
            input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
            labels = torch.roll(input_ids, shifts=-1, dims=1)

            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"step={step} loss={loss.item():.4f} "
                f"logits_device={logits.device} is_cuda={getattr(logits, 'is_cuda', False)}"
            )

        print("Transformer training simulation finished")
        return True

    except ImportError:
        print("PyTorch not installed; skip (pip install torch)")
        return False
    except Exception as e:
        print(f"Transformer training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print()


def print_usage_summary():
    """Print a brief usage guide."""
    print("=" * 70)
    print("How to run")
    print("=" * 70)
    print("./fgpu python demo_usage.py --test all")
    print("./fgpu python demo_usage.py --test nvml --max-devices 2")
    print("./fgpu python demo_usage.py --test cuda --alloc-size 256")
    print("./fgpu python demo_usage.py --test pytorch")
    print("python3 demo_usage.py --test transformer")
    print("Report file: fake_gpu_report.json (written at program exit)")
    print()


def main():
    """Entry point for running the simplified demo."""
    parser = argparse.ArgumentParser(
        description='FakeGPU demo - simple NVML, CUDA, and PyTorch checks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ./fgpu python demo_usage.py --test all
    ./fgpu python demo_usage.py --test nvml --max-devices 2
    ./fgpu python demo_usage.py --test cuda --alloc-size 256
    ./fgpu python demo_usage.py --test pytorch
    python3 demo_usage.py --test transformer
    ./fgpu python demo_usage.py --no-summary
        """
    )

    parser.add_argument(
        '--lib-path',
                default=os.environ.get('FAKE_GPU_LIB', _default_lib_path()),
                help='Path to a FakeGPU library or library directory (default: build output or $FAKE_GPU_LIB)'
    )

    parser.add_argument(
        '--test',
        choices=['all', 'nvml', 'cuda', 'pytorch', 'transformer'],
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

    init_result, fake_gpu = load_fake_gpu_library(args.lib_path)
    if not args.quiet:
        print()

    results = {}

    if args.test in ['all', 'nvml']:
        results['nvml'] = test_pynvml(init_result.handles[_nvml_lib_name()], max_devices=args.max_devices)

    if args.test in ['all', 'cuda']:
        results['cuda'] = test_cuda_runtime(fake_gpu, alloc_size_mb=args.alloc_size)

    if args.test in ['all', 'pytorch']:
        results['pytorch'] = test_pytorch()

    if args.test == 'transformer':
        results['transformer'] = test_transformer_training()

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

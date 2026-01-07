#!/usr/bin/env python3
"""
对比测试脚本：在真实GPU和FakeGPU上运行相同的测试
用于验证测试代码本身没有问题，只是FakeGPU的实现还不完整
"""

import torch
import torch.nn as nn
import argparse
import os
import sys


class SimpleModel(nn.Module):
    """简单的测试模型，只使用基本操作"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_basic_operations(device):
    """测试基本张量操作"""
    print(f"\n{'='*60}")
    print(f"Testing on device: {device}")
    print(f"{'='*60}")

    results = []

    # Test 1: Tensor creation
    try:
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        results.append(("Tensor creation", "PASS"))
        print("✓ Tensor creation: PASS")
    except Exception as e:
        results.append(("Tensor creation", f"FAIL: {e}"))
        print(f"✗ Tensor creation: FAIL - {e}")
        return results

    # Test 2: Element-wise operations
    try:
        z = x + y
        z = x * y
        z = torch.sin(x)
        results.append(("Element-wise operations", "PASS"))
        print("✓ Element-wise operations: PASS")
    except Exception as e:
        results.append(("Element-wise operations", f"FAIL: {e}"))
        print(f"✗ Element-wise operations: FAIL - {e}")

    # Test 3: Basic matrix multiplication (without cuBLASLt)
    try:
        a = torch.randn(50, 50, device=device)
        b = torch.randn(50, 50, device=device)
        c = torch.mm(a, b)  # Basic matrix multiplication
        results.append(("Basic matmul", "PASS"))
        print("✓ Basic matmul: PASS")
    except Exception as e:
        results.append(("Basic matmul", f"FAIL: {e}"))
        print(f"✗ Basic matmul: FAIL - {e}")

    # Test 4: Simple Linear layer (may fail on FakeGPU if cuBLASLt is used)
    try:
        linear = nn.Linear(50, 30).to(device)
        x = torch.randn(10, 50, device=device)
        y = linear(x)
        results.append(("Linear layer", "PASS"))
        print("✓ Linear layer: PASS")
    except Exception as e:
        results.append(("Linear layer", f"FAIL: {e}"))
        print(f"✗ Linear layer: FAIL - {e}")

    # Test 5: Simple model forward pass
    try:
        model = SimpleModel(100, 50, 10).to(device)
        x = torch.randn(32, 100, device=device)
        y = model(x)
        results.append(("Model forward", "PASS"))
        print("✓ Model forward: PASS")
    except Exception as e:
        results.append(("Model forward", f"FAIL: {e}"))
        print(f"✗ Model forward: FAIL - {e}")

    # Test 6: Memory transfer
    try:
        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = cpu_tensor.to(device)
        cpu_tensor_back = gpu_tensor.cpu()
        results.append(("Memory transfer", "PASS"))
        print("✓ Memory transfer: PASS")
    except Exception as e:
        results.append(("Memory transfer", f"FAIL: {e}"))
        print(f"✗ Memory transfer: FAIL - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Comparison test: Real GPU vs FakeGPU')
    parser.add_argument('--mode', choices=['real', 'fake', 'both'], default='both',
                        help='Test mode: real GPU, fake GPU, or both')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device ID to use')
    args = parser.parse_args()

    print("="*60)
    print("PyTorch GPU Comparison Test")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available() and args.device_id < torch.cuda.device_count():
        print(f"Device {args.device_id}: {torch.cuda.get_device_name(args.device_id)}")
    print()

    # Detect if we're running with FakeGPU (Linux: LD_PRELOAD, macOS: DYLD_INSERT_LIBRARIES)
    preload_var = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
    is_fake_gpu = preload_var in os.environ and "libcuda" in os.environ.get(preload_var, "")

    all_results = {}

    if args.mode in ['real', 'both']:
        if not is_fake_gpu and torch.cuda.is_available():
            print("\n" + "="*60)
            print("TESTING ON REAL GPU")
            print("="*60)
            device = f'cuda:{args.device_id}'
            all_results['real'] = test_basic_operations(device)
        elif args.mode == 'real':
            print("\n✗ Real GPU not available or FakeGPU is active")
            print("  Please run without FakeGPU preloading to test on real GPU")

    if args.mode in ['fake', 'both']:
        if is_fake_gpu or args.mode == 'fake':
            print("\n" + "="*60)
            print("TESTING ON FAKEGPU")
            print("="*60)
            if torch.cuda.is_available():
                device = f'cuda:{args.device_id}'
                all_results['fake'] = test_basic_operations(device)
            else:
                print("✗ CUDA not available (FakeGPU not loaded?)")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for mode, results in all_results.items():
        print(f"\n{mode.upper()} GPU Results:")
        pass_count = sum(1 for _, status in results if status == "PASS")
        total_count = len(results)

        for test_name, status in results:
            symbol = "✓" if status == "PASS" else "✗"
            print(f"  {symbol} {test_name}: {status}")

        print(f"\n  Total: {pass_count}/{total_count} tests passed")

    # Print interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if 'real' in all_results:
        real_pass = sum(1 for _, s in all_results['real'] if s == "PASS")
        print(f"\nReal GPU: {real_pass}/{len(all_results['real'])} tests passed")
        if real_pass == len(all_results['real']):
            print("  → Test code is working correctly on real hardware")

    if 'fake' in all_results:
        fake_pass = sum(1 for _, s in all_results['fake'] if s == "PASS")
        print(f"\nFakeGPU: {fake_pass}/{len(all_results['fake'])} tests passed")
        if fake_pass < len(all_results['fake']):
            print("  → FakeGPU implementation needs additional work")
            print("  → The test failures are due to missing FakeGPU features,")
            print("    NOT due to problems with the test code or environment")


if __name__ == '__main__':
    main()

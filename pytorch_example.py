#!/usr/bin/env python3
"""
PyTorch中使用fakeGPU的实用示例

这个脚本展示如何在PyTorch中使用虚拟GPU进行：
1. 内存分配追踪
2. 模型训练调试
3. 多GPU模拟
"""

import os
import sys
from ctypes import CDLL

# 在导入torch之前加载fake GPU库
fake_gpu_lib = os.environ.get('FAKE_GPU_LIB', './build/libfake_gpu.so')
if os.path.exists(fake_gpu_lib):
    print(f"[预加载] 加载虚拟GPU库: {fake_gpu_lib}")
    CDLL(fake_gpu_lib, mode=os.RTLD_GLOBAL)
else:
    print(f"警告: 找不到虚拟GPU库 {fake_gpu_lib}")

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"错误: PyTorch未安装: {e}")
    print("安装命令: pip install torch")
    sys.exit(1)

print("=" * 70)
print("PyTorch + FakeGPU 使用示例")
print("=" * 70)
print()

# ============================================================================
# 示例1: 基本的GPU内存分配
# ============================================================================
print("示例1: 基本的GPU内存分配")
print("-" * 70)

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")

    # 创建张量
    device = torch.device("cuda:0")
    print(f"\n在 {device} 上分配张量...")

    # 小张量
    x = torch.randn(100, 100, device=device)
    print(f"✓ 创建 100x100 张量")
    print(f"  PyTorch报告的内存: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # 大张量
    y = torch.randn(1000, 1000, device=device)
    print(f"✓ 创建 1000x1000 张量")
    print(f"  PyTorch报告的内存: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # 释放内存
    del x, y
    torch.cuda.empty_cache()
    print(f"✓ 释放张量")
    print(f"  PyTorch报告的内存: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
else:
    print("CUDA不可用，跳过GPU测试")

print()

# ============================================================================
# 示例2: 简单的神经网络
# ============================================================================
print("示例2: 简单的神经网络")
print("-" * 70)

if torch.cuda.is_available():
    # 定义一个简单的网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # 创建模型并移到GPU
    device = torch.device("cuda:0")
    model = SimpleNet().to(device)
    print(f"✓ 创建神经网络并移到 {device}")
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"  PyTorch报告的内存: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # 创建输入数据
    batch_size = 32
    input_data = torch.randn(batch_size, 784, device=device)
    print(f"\n✓ 创建输入数据 (batch_size={batch_size})")
    print(f"  PyTorch报告的内存: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # 前向传播
    output = model(input_data)
    print(f"\n✓ 前向传播完成")
    print(f"  输出形状: {output.shape}")
    print(f"  PyTorch报告的内存: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # 注意：虚拟GPU不会真正执行计算，但会追踪内存分配
    print("\n注意: 虚拟GPU拦截了内存分配，但不执行实际计算")
    print("      这对于调试内存问题和测试代码逻辑很有用")

    # 清理
    del model, input_data, output
    torch.cuda.empty_cache()

print()

# ============================================================================
# 示例3: 批量数据处理
# ============================================================================
print("示例3: 批量数据处理")
print("-" * 70)

if torch.cuda.is_available():
    device = torch.device("cuda:0")

    print("模拟批量处理数据...")
    total_memory_used = 0

    for i in range(5):
        # 创建批次数据
        batch = torch.randn(64, 3, 224, 224, device=device)
        batch_size_mb = batch.element_size() * batch.nelement() / (1024**2)
        total_memory_used += batch_size_mb

        print(f"  批次 {i+1}: {batch.shape}, 大小: {batch_size_mb:.2f} MB")

        # 处理完后释放
        del batch

    torch.cuda.empty_cache()
    print(f"\n✓ 处理了5个批次，总共使用约 {total_memory_used:.2f} MB")

print()

# ============================================================================
# 示例4: 数据传输（CPU <-> GPU）
# ============================================================================
print("示例4: 数据传输（CPU <-> GPU）")
print("-" * 70)

if torch.cuda.is_available():
    # CPU上的数据
    cpu_tensor = torch.randn(500, 500)
    print(f"CPU张量: {cpu_tensor.shape}, 设备: {cpu_tensor.device}")

    # 传输到GPU
    gpu_tensor = cpu_tensor.to('cuda:0')
    print(f"✓ 传输到GPU: {gpu_tensor.shape}, 设备: {gpu_tensor.device}")
    print(f"  PyTorch报告的内存: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

    # 传输回CPU
    cpu_tensor2 = gpu_tensor.to('cpu')
    print(f"✓ 传输回CPU: {cpu_tensor2.shape}, 设备: {cpu_tensor2.device}")

    # 验证数据一致性
    if torch.allclose(cpu_tensor, cpu_tensor2):
        print("✓ 数据传输前后一致")
    else:
        print("✗ 数据传输前后不一致（这是预期的，因为虚拟GPU不执行实际计算）")

print()

# ============================================================================
# 总结和使用建议
# ============================================================================
print("=" * 70)
print("使用总结")
print("=" * 70)
print()
print("虚拟GPU的功能:")
print("✓ 拦截CUDA Runtime API调用（cudaMalloc, cudaFree等）")
print("✓ 追踪GPU内存分配和使用情况")
print("✓ 生成内存使用报告（fake_gpu_report.json）")
print("✓ 模拟多GPU环境")
print()
print("限制:")
print("✗ 不执行实际的GPU计算（kernel launches是no-op）")
print("✗ PyTorch可能仍使用真实GPU驱动进行设备检测")
print("✗ 计算结果可能不正确（因为没有实际执行）")
print()
print("适用场景:")
print("• 在没有GPU的机器上测试代码逻辑")
print("• 调试GPU内存泄漏和OOM问题")
print("• 开发GPU相关工具和框架")
print("• 教学和演示GPU编程概念")
print("• CI/CD环境中的基本测试")
print()
print("运行方式:")
print("  LD_PRELOAD=./build/libfake_gpu.so python pytorch_example.py")
print()
print("查看内存报告:")
print("  cat fake_gpu_report.json")
print()

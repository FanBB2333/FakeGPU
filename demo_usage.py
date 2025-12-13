#!/usr/bin/env python3
"""
演示如何使用fakeGPU库来模拟NVIDIA GPU

这个脚本展示了三种使用场景：
1. 使用pynvml库检测GPU（NVML API）
2. 使用ctypes直接调用CUDA Runtime API
3. 尝试使用PyTorch（可能需要额外配置）
"""

import os
import sys
from ctypes import CDLL, c_int, c_void_p, c_char_p, c_size_t, POINTER, byref

print("=" * 70)
print("FakeGPU 使用演示")
print("=" * 70)
print()

# 加载fake GPU库
fake_gpu_lib_path = os.environ.get('FAKE_GPU_LIB', './build/libfake_gpu.so')
if not os.path.exists(fake_gpu_lib_path):
    print(f"错误: 找不到库文件 {fake_gpu_lib_path}")
    print("请先构建库: cmake --build build")
    sys.exit(1)

print(f"加载虚拟GPU库: {fake_gpu_lib_path}")
fake_gpu = CDLL(fake_gpu_lib_path)
print()

# ============================================================================
# 场景1: 使用pynvml检测GPU（NVML API）
# ============================================================================
print("场景1: 使用pynvml检测GPU")
print("-" * 70)

try:
    import pynvml

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"检测到 {device_count} 个GPU设备")
    print()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"GPU {i}:")
        print(f"  名称: {name}")
        print(f"  UUID: {uuid}")
        print(f"  总内存: {memory_info.total / (1024**3):.2f} GB")
        print(f"  已用内存: {memory_info.used / (1024**2):.2f} MB")
        print()

    pynvml.nvmlShutdown()
    print("✓ pynvml测试成功")

except ImportError:
    print("pynvml未安装，跳过此测试")
    print("安装命令: pip install nvidia-ml-py3")
except Exception as e:
    print(f"✗ pynvml测试失败: {e}")

print()

# ============================================================================
# 场景2: 直接使用CUDA Runtime API
# ============================================================================
print("场景2: 直接使用CUDA Runtime API")
print("-" * 70)

try:
    # 定义CUDA Runtime API函数
    cudaGetDeviceCount = fake_gpu.cudaGetDeviceCount
    cudaGetDeviceCount.argtypes = [POINTER(c_int)]
    cudaGetDeviceCount.restype = c_int

    cudaMalloc = fake_gpu.cudaMalloc
    cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
    cudaMalloc.restype = c_int

    cudaFree = fake_gpu.cudaFree
    cudaFree.argtypes = [c_void_p]
    cudaFree.restype = c_int

    # 获取设备数量
    device_count = c_int()
    result = cudaGetDeviceCount(byref(device_count))
    print(f"cudaGetDeviceCount 返回: {device_count.value} 个设备")

    # 分配内存
    size = 1024 * 1024 * 100  # 100 MB
    device_ptr = c_void_p()
    result = cudaMalloc(byref(device_ptr), size)
    if result == 0:
        print(f"✓ cudaMalloc 成功分配 {size / (1024**2):.2f} MB")
        print(f"  设备指针: 0x{device_ptr.value:x}")

        # 释放内存
        result = cudaFree(device_ptr)
        if result == 0:
            print(f"✓ cudaFree 成功释放内存")
    else:
        print(f"✗ cudaMalloc 失败，错误码: {result}")

    print()
    print("✓ CUDA Runtime API测试成功")

except Exception as e:
    print(f"✗ CUDA Runtime API测试失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 场景3: 使用PyTorch（可能需要额外配置）
# ============================================================================
print("场景3: 使用PyTorch")
print("-" * 70)

try:
    import torch

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print()
        print("注意: PyTorch可能使用真实的GPU驱动进行设备检测")
        print("但内存分配和计算操作会被虚拟GPU拦截")
    else:
        print()
        print("PyTorch未检测到CUDA支持")
        print("这可能是因为:")
        print("1. PyTorch版本不支持CUDA")
        print("2. 需要在没有真实NVIDIA驱动的环境中运行")
        print("3. 需要额外的Driver API拦截")

except ImportError:
    print("PyTorch未安装，跳过此测试")
    print("安装命令: pip install torch")
except Exception as e:
    print(f"PyTorch测试遇到问题: {e}")

print()

# ============================================================================
# 总结
# ============================================================================
print("=" * 70)
print("使用说明")
print("=" * 70)
print()
print("1. 使用LD_PRELOAD运行程序:")
print("   LD_PRELOAD=./build/libfake_gpu.so python your_script.py")
print()
print("2. 或者在Python中预加载库:")
print("   from ctypes import CDLL")
print("   CDLL('./build/libfake_gpu.so', mode=os.RTLD_GLOBAL)")
print()
print("3. 查看内存使用报告:")
print("   程序结束后会生成 fake_gpu_report.json 文件")
print()
print("4. 适用场景:")
print("   - 在没有GPU的机器上测试GPU代码")
print("   - 调试GPU内存分配问题")
print("   - 模拟多GPU环境")
print("   - 开发和测试GPU相关工具")
print()

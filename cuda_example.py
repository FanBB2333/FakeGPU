#!/usr/bin/env python3
"""
使用fakeGPU库的纯CUDA示例

这个脚本展示如何直接使用CUDA Runtime API和NVML API
不依赖PyTorch，可以完全使用虚拟GPU
"""

import os
import sys
import json
from ctypes import (
    CDLL, c_int, c_void_p, c_char_p, c_size_t, c_uint, c_ulonglong,
    POINTER, byref, Structure, create_string_buffer
)

print("=" * 70)
print("CUDA + FakeGPU 完整示例")
print("=" * 70)
print()

# 加载虚拟GPU库
fake_gpu_lib_path = os.environ.get('FAKE_GPU_LIB', './build/libfake_gpu.so')
if not os.path.exists(fake_gpu_lib_path):
    print(f"错误: 找不到库文件 {fake_gpu_lib_path}")
    sys.exit(1)

print(f"加载虚拟GPU库: {fake_gpu_lib_path}")
libfake = CDLL(fake_gpu_lib_path)
print()

# ============================================================================
# 示例1: CUDA Runtime API - 设备管理
# ============================================================================
print("示例1: CUDA Runtime API - 设备管理")
print("-" * 70)

# 定义函数
cudaGetDeviceCount = libfake.cudaGetDeviceCount
cudaGetDeviceCount.argtypes = [POINTER(c_int)]
cudaGetDeviceCount.restype = c_int

cudaSetDevice = libfake.cudaSetDevice
cudaSetDevice.argtypes = [c_int]
cudaSetDevice.restype = c_int

cudaGetDevice = libfake.cudaGetDevice
cudaGetDevice.argtypes = [POINTER(c_int)]
cudaGetDevice.restype = c_int

# 获取设备数量
device_count = c_int()
result = cudaGetDeviceCount(byref(device_count))
print(f"✓ cudaGetDeviceCount: {device_count.value} 个设备")

# 设置当前设备
result = cudaSetDevice(0)
print(f"✓ cudaSetDevice(0): 设置成功")

# 获取当前设备
current_device = c_int()
result = cudaGetDevice(byref(current_device))
print(f"✓ cudaGetDevice: 当前设备 {current_device.value}")

print()

# ============================================================================
# 示例2: CUDA Runtime API - 内存管理
# ============================================================================
print("示例2: CUDA Runtime API - 内存管理")
print("-" * 70)

# 定义函数
cudaMalloc = libfake.cudaMalloc
cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
cudaMalloc.restype = c_int

cudaFree = libfake.cudaFree
cudaFree.argtypes = [c_void_p]
cudaFree.restype = c_int

cudaMemcpy = libfake.cudaMemcpy
cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
cudaMemcpy.restype = c_int

# cudaMemcpyKind枚举
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# 分配设备内存
sizes = [
    (1024 * 1024, "1 MB"),
    (10 * 1024 * 1024, "10 MB"),
    (100 * 1024 * 1024, "100 MB"),
]

device_ptrs = []
for size, size_str in sizes:
    device_ptr = c_void_p()
    result = cudaMalloc(byref(device_ptr), size)
    if result == 0:
        print(f"✓ cudaMalloc: 分配 {size_str} 成功")
        print(f"  设备指针: 0x{device_ptr.value:x}")
        device_ptrs.append((device_ptr, size, size_str))
    else:
        print(f"✗ cudaMalloc: 分配 {size_str} 失败，错误码 {result}")

print()

# 数据传输示例
print("数据传输示例:")
host_data = b"Hello from CPU!" * 100  # 1500 bytes
device_buffer = c_void_p()
buffer_size = len(host_data)

result = cudaMalloc(byref(device_buffer), buffer_size)
if result == 0:
    print(f"✓ 分配 {buffer_size} 字节的设备缓冲区")

    # Host to Device
    result = cudaMemcpy(device_buffer, host_data, buffer_size, cudaMemcpyHostToDevice)
    if result == 0:
        print(f"✓ cudaMemcpy: Host -> Device 传输 {buffer_size} 字节")

    # Device to Host
    host_buffer = create_string_buffer(buffer_size)
    result = cudaMemcpy(host_buffer, device_buffer, buffer_size, cudaMemcpyDeviceToHost)
    if result == 0:
        print(f"✓ cudaMemcpy: Device -> Host 传输 {buffer_size} 字节")
        if host_buffer.raw == host_data:
            print(f"✓ 数据验证: 传输前后数据一致")
        else:
            print(f"✗ 数据验证: 传输前后数据不一致")

    # 释放缓冲区
    cudaFree(device_buffer)
    print(f"✓ cudaFree: 释放设备缓冲区")

print()

# 释放之前分配的内存
print("释放之前分配的内存:")
for device_ptr, size, size_str in device_ptrs:
    result = cudaFree(device_ptr)
    if result == 0:
        print(f"✓ cudaFree: 释放 {size_str}")

print()

# ============================================================================
# 示例3: NVML API - 设备信息查询
# ============================================================================
print("示例3: NVML API - 设备信息查询")
print("-" * 70)

try:
    # 定义NVML函数
    nvmlInit = libfake.nvmlInit_v2
    nvmlInit.argtypes = []
    nvmlInit.restype = c_int

    nvmlDeviceGetCount = libfake.nvmlDeviceGetCount_v2
    nvmlDeviceGetCount.argtypes = [POINTER(c_uint)]
    nvmlDeviceGetCount.restype = c_int

    nvmlDeviceGetHandleByIndex = libfake.nvmlDeviceGetHandleByIndex_v2
    nvmlDeviceGetHandleByIndex.argtypes = [c_uint, POINTER(c_void_p)]
    nvmlDeviceGetHandleByIndex.restype = c_int

    nvmlDeviceGetName = libfake.nvmlDeviceGetName
    nvmlDeviceGetName.argtypes = [c_void_p, c_char_p, c_uint]
    nvmlDeviceGetName.restype = c_int

    # 定义内存信息结构
    class nvmlMemory_t(Structure):
        _fields_ = [
            ("total", c_ulonglong),
            ("free", c_ulonglong),
            ("used", c_ulonglong),
        ]

    nvmlDeviceGetMemoryInfo = libfake.nvmlDeviceGetMemoryInfo
    nvmlDeviceGetMemoryInfo.argtypes = [c_void_p, POINTER(nvmlMemory_t)]
    nvmlDeviceGetMemoryInfo.restype = c_int

    # 初始化NVML
    result = nvmlInit()
    print(f"✓ nvmlInit: 初始化成功")

    # 获取设备数量
    nvml_device_count = c_uint()
    result = nvmlDeviceGetCount(byref(nvml_device_count))
    print(f"✓ nvmlDeviceGetCount: {nvml_device_count.value} 个设备")
    print()

    # 查询每个设备的信息
    for i in range(min(nvml_device_count.value, 3)):  # 只显示前3个设备
        handle = c_void_p()
        result = nvmlDeviceGetHandleByIndex(i, byref(handle))

        if result == 0:
            print(f"设备 {i}:")

            # 获取设备名称
            name_buffer = create_string_buffer(256)
            result = nvmlDeviceGetName(handle, name_buffer, 256)
            if result == 0:
                print(f"  名称: {name_buffer.value.decode('utf-8')}")

            # 获取内存信息
            memory_info = nvmlMemory_t()
            result = nvmlDeviceGetMemoryInfo(handle, byref(memory_info))
            if result == 0:
                total_gb = memory_info.total / (1024**3)
                used_mb = memory_info.used / (1024**2)
                free_gb = memory_info.free / (1024**3)
                print(f"  总内存: {total_gb:.2f} GB")
                print(f"  已用内存: {used_mb:.2f} MB")
                print(f"  可用内存: {free_gb:.2f} GB")
            print()

except Exception as e:
    print(f"NVML API调用失败: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 示例4: 模拟实际工作负载
# ============================================================================
print("示例4: 模拟实际工作负载")
print("-" * 70)

print("模拟机器学习训练过程的内存分配模式...")
print()

# 模型参数
model_size = 50 * 1024 * 1024  # 50 MB
model_ptr = c_void_p()
result = cudaMalloc(byref(model_ptr), model_size)
print(f"✓ 分配模型参数: {model_size / (1024**2):.2f} MB")

# 优化器状态
optimizer_size = 100 * 1024 * 1024  # 100 MB
optimizer_ptr = c_void_p()
result = cudaMalloc(byref(optimizer_ptr), optimizer_size)
print(f"✓ 分配优化器状态: {optimizer_size / (1024**2):.2f} MB")

# 模拟训练批次
batch_size = 32
batch_memory = 20 * 1024 * 1024  # 20 MB per batch

print(f"\n模拟训练 5 个批次:")
for epoch in range(5):
    batch_ptr = c_void_p()
    result = cudaMalloc(byref(batch_ptr), batch_memory)
    print(f"  Epoch {epoch + 1}: 分配批次数据 {batch_memory / (1024**2):.2f} MB")

    # 模拟前向和反向传播（实际上不执行）
    # ...

    # 释放批次数据
    cudaFree(batch_ptr)

print()

# 清理
cudaFree(model_ptr)
cudaFree(optimizer_ptr)
print("✓ 清理所有分配的内存")

print()

# ============================================================================
# 查看内存使用报告
# ============================================================================
print("=" * 70)
print("内存使用报告")
print("=" * 70)
print()

# 程序结束时会自动生成报告，这里我们手动读取之前的报告
report_file = "fake_gpu_report.json"
if os.path.exists(report_file):
    print(f"注意: 程序结束后会生成新的 {report_file}")
    print("      报告包含每个设备的内存使用统计")
    print()
    print("报告格式示例:")
    print(json.dumps({
        "devices": [
            {
                "name": "Fake NVIDIA A100-SXM4-80GB",
                "uuid": "GPU-00000000-abcd-ef01-2345-0000abcdef00",
                "total_memory": 85899345920,
                "used_memory_peak": 314572800,
                "used_memory_current": 0
            }
        ]
    }, indent=2))
else:
    print(f"报告文件 {report_file} 将在程序结束时生成")

print()

# ============================================================================
# 使用总结
# ============================================================================
print("=" * 70)
print("使用总结")
print("=" * 70)
print()
print("本示例展示了:")
print("✓ CUDA Runtime API的设备管理和内存操作")
print("✓ NVML API的设备信息查询")
print("✓ 数据传输（Host <-> Device）")
print("✓ 模拟实际工作负载的内存分配模式")
print()
print("虚拟GPU的优势:")
print("• 无需物理GPU即可测试CUDA代码")
print("• 追踪和分析内存使用情况")
print("• 模拟多GPU环境")
print("• 快速原型开发和调试")
print()
print("运行方式:")
print("  LD_PRELOAD=./build/libfake_gpu.so python cuda_example.py")
print()
print("查看详细日志:")
print("  在输出中查找 [FakeCUDA] 和 [FakeNVML] 标记")
print()

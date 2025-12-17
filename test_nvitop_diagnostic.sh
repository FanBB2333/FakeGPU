#!/bin/bash

echo "=========================================="
echo "nvitop 诊断测试"
echo "=========================================="
echo ""

echo "方法1: 使用 Python API 测试 nvitop"
echo "------------------------------------------"
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
python3 << 'PYTHON_EOF'
import os
os.environ['TERM'] = 'dumb'  # 使用简单终端模式

try:
    from nvitop import Device
    
    devices = Device.all()
    print(f"\n检测到 {len(devices)} 个GPU设备:\n")
    
    for i, dev in enumerate(devices):
        print(f"GPU {i}:")
        print(f"  名称: {dev.name()}")
        
        # 获取内存信息
        mem_info = dev.memory_info()
        print(f"  总内存: {mem_info.total / 1024**3:.2f} GB")
        print(f"  已用: {mem_info.used / 1024**3:.2f} GB")
        print(f"  空闲: {mem_info.free / 1024**3:.2f} GB")
        
        # 获取其他信息
        try:
            print(f"  温度: {dev.temperature()}°C")
        except:
            print(f"  温度: N/A")
            
        try:
            print(f"  功耗: {dev.power_usage() / 1000:.1f}W")
        except:
            print(f"  功耗: N/A")
            
        try:
            print(f"  利用率: {dev.gpu_utilization()}%")
        except:
            print(f"  利用率: N/A")
        
        print()
    
    print("✓ nvitop Python API 测试通过!")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
PYTHON_EOF

echo ""
echo "=========================================="
echo "方法2: 使用简化TUI模式"
echo "=========================================="
echo ""
echo "注意: nvitop的完整TUI可能需要额外的NVML函数"
echo "      如果TUI卡住，按 Ctrl+C 退出并运行 'reset' 恢复终端"
echo ""
echo "按Enter继续测试，或 Ctrl+C 跳过..."
read

# 尝试运行nvitop的简化模式
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
timeout 5 nvitop --once 2>&1 || {
    echo ""
    echo "nvitop TUI 模式失败"
    echo ""
    echo "如果终端光标消失，请运行: reset"
}

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="

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
import argparse
from ctypes import CDLL, c_int, c_void_p, c_char_p, c_size_t, POINTER, byref


def load_fake_gpu_library(lib_path):
    """加载虚拟GPU库"""
    if not os.path.exists(lib_path):
        print(f"错误: 找不到库文件 {lib_path}")
        print("请先构建库: cmake --build build")
        sys.exit(1)

    print(f"加载虚拟GPU库: {lib_path}")
    return CDLL(lib_path)


def test_pynvml(max_devices=None):
    """场景1: 使用pynvml检测GPU（NVML API）"""
    print("场景1: 使用pynvml检测GPU")
    print("-" * 70)

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"检测到 {device_count} 个GPU设备")
        print()

        display_count = min(device_count, max_devices) if max_devices else device_count
        for i in range(display_count):
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

        if max_devices and device_count > max_devices:
            print(f"... 还有 {device_count - max_devices} 个设备未显示")
            print()

        pynvml.nvmlShutdown()
        print("✓ pynvml测试成功")
        return True

    except ImportError:
        print("pynvml未安装，跳过此测试")
        print("安装命令: pip install nvidia-ml-py3")
        return False
    except Exception as e:
        print(f"✗ pynvml测试失败: {e}")
        return False
    finally:
        print()


def test_cuda_runtime(fake_gpu, alloc_size_mb=100):
    """场景2: 直接使用CUDA Runtime API"""
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
        size = 1024 * 1024 * alloc_size_mb
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
        return True

    except Exception as e:
        print(f"✗ CUDA Runtime API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print()


def test_pytorch():
    """场景3: 使用PyTorch（可能需要额外配置）"""
    print("场景3: 使用PyTorch")
    print("-" * 70)

    try:
        import torch

        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            # 打印具体GPU信息
            # for i in range(torch.cuda.device_count()):
            #     print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
            print()
            print("注意: PyTorch可能使用真实的GPU驱动进行设备检测")
            print("但内存分配和计算操作会被虚拟GPU拦截")
            return True
        else:
            print()
            print("PyTorch未检测到CUDA支持")
            print("这可能是因为:")
            print("1. PyTorch版本不支持CUDA")
            print("2. 需要在没有真实NVIDIA驱动的环境中运行")
            print("3. 需要额外的Driver API拦截")
            return False

    except ImportError:
        print("PyTorch未安装，跳过此测试")
        print("安装命令: pip install torch")
        return False
    except Exception as e:
        print(f"PyTorch测试遇到问题: {e}")
        return False
    finally:
        print()


def print_usage_summary():
    """打印使用说明"""
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='FakeGPU 使用演示 - 模拟NVIDIA GPU的三种使用场景',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有测试
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py

  # 只运行NVML测试，最多显示3个设备
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test nvml --max-devices 3

  # 运行CUDA测试，分配500MB内存
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test cuda --alloc-size 500

  # 只运行PyTorch测试
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --test pytorch

  # 不显示使用说明
  LD_PRELOAD=./build/libfake_gpu.so python demo_usage.py --no-summary
        """
    )

    parser.add_argument(
        '--lib-path',
        default=os.environ.get('FAKE_GPU_LIB', './build/libfake_gpu.so'),
        help='虚拟GPU库路径 (默认: ./build/libfake_gpu.so 或 $FAKE_GPU_LIB)'
    )

    parser.add_argument(
        '--test',
        choices=['all', 'nvml', 'cuda', 'pytorch'],
        default='all',
        help='选择要运行的测试 (默认: all)'
    )

    parser.add_argument(
        '--max-devices',
        type=int,
        metavar='N',
        help='NVML测试中最多显示的设备数量'
    )

    parser.add_argument(
        '--alloc-size',
        type=int,
        default=100,
        metavar='MB',
        help='CUDA测试中分配的内存大小(MB) (默认: 100)'
    )

    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='不显示使用说明'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，只显示测试结果'
    )

    args = parser.parse_args()

    # 打印标题
    if not args.quiet:
        print("=" * 70)
        print("FakeGPU 使用演示")
        print("=" * 70)
        print()

    # 加载库
    fake_gpu = load_fake_gpu_library(args.lib_path)
    if not args.quiet:
        print()

    # 运行测试
    results = {}

    if args.test in ['all', 'nvml']:
        results['nvml'] = test_pynvml(max_devices=args.max_devices)

    if args.test in ['all', 'cuda']:
        results['cuda'] = test_cuda_runtime(fake_gpu, alloc_size_mb=args.alloc_size)

    if args.test in ['all', 'pytorch']:
        results['pytorch'] = test_pytorch()

    # 打印测试结果摘要
    if not args.quiet and len(results) > 1:
        print("=" * 70)
        print("测试结果摘要")
        print("=" * 70)
        for test_name, success in results.items():
            status = "✓ 通过" if success else "✗ 失败/跳过"
            print(f"{test_name.upper()}: {status}")
        print()

    # 打印使用说明
    if not args.no_summary and not args.quiet:
        print_usage_summary()


if __name__ == '__main__':
    main()

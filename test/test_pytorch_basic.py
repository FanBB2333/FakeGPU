import torch
import sys

print("=== FakeGPU PyTorch 基础测试 ===")
print(f"PyTorch版本: {torch.__version__}")

try:
    print(f"\n1. 检查CUDA可用性...")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA可用: {cuda_available}")

    if not cuda_available:
        print("   错误: CUDA不可用")
        sys.exit(1)

    print(f"\n2. 获取GPU数量...")
    device_count = torch.cuda.device_count()
    print(f"   GPU数量: {device_count}")

    print(f"\n3. 获取当前设备...")
    current_device = torch.cuda.current_device()
    print(f"   当前设备: {current_device}")

    print(f"\n4. 测试设备属性...")
    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}:")
            print(f"     名称: {props.name}")
            print(f"     总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"     计算能力: {props.major}.{props.minor}")
            print(f"     多处理器数量: {props.multi_processor_count}")
        except Exception as e:
            print(f"   GPU {i}: 获取属性失败 - {e}")

    print(f"\n5. 测试张量创建...")
    device = torch.device('cuda:0')
    x = torch.randn(3, 3, device=device)
    print(f"   创建张量成功: shape={x.shape}, device={x.device}")

    print(f"\n6. 测试张量运算...")
    y = torch.randn(3, 3, device=device)
    z = x + y
    print(f"   张量加法成功: shape={z.shape}")

    print(f"\n7. 测试矩阵乘法...")
    a = torch.randn(100, 100, device=device)
    b = torch.randn(100, 100, device=device)
    c = torch.matmul(a, b)
    print(f"   矩阵乘法成功: shape={c.shape}")

    print(f"\n8. 测试内存分配...")
    large_tensor = torch.randn(1000, 1000, device=device)
    print(f"   大张量创建成功: shape={large_tensor.shape}")
    print(f"   已分配内存: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   已保留内存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    print(f"\n9. 测试设备切换...")
    if device_count > 1:
        torch.cuda.set_device(1)
        print(f"   切换到设备1成功")
        x1 = torch.randn(3, 3, device='cuda:1')
        print(f"   在设备1创建张量成功: device={x1.device}")
    else:
        print(f"   只有一个GPU，跳过设备切换测试")

    print(f"\n10. 测试CPU-GPU数据传输...")
    cpu_tensor = torch.randn(10, 10)
    gpu_tensor = cpu_tensor.to(device)
    cpu_tensor_back = gpu_tensor.cpu()
    print(f"   数据传输成功")
    print(f"   数据一致性: {torch.allclose(cpu_tensor, cpu_tensor_back)}")

    print(f"\n=== 所有测试通过 ===")

except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

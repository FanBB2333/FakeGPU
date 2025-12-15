import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 临时补丁解决 torchvision 问题
import torch
_original_has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key

def _patched_has_kernel(name, key):
    if "torchvision::" in name:
        return False
    return _original_has_kernel(name, key)

torch._C._dispatch_has_kernel_for_dispatch_key = _patched_has_kernel

print("Testing basic PyTorch CUDA operations...")

# 测试1: 检查CUDA是否可用
print(f"1. torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CUDA not available, exiting")
    exit(1)

# 测试2: 获取设备数量
print(f"2. torch.cuda.device_count(): {torch.cuda.device_count()}")

# 测试3: 获取设备名称
print(f"3. torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

# 测试4: 获取设备属性
props = torch.cuda.get_device_properties(0)
print(f"4. Device properties: total_memory={props.total_memory / 1e9:.2f}GB, major={props.major}, minor={props.minor}")

# 测试5: 创建简单tensor
print("\n5. Creating simple tensors...")
x = torch.randn(10, 10).cuda()
print(f"   Created tensor on device: {x.device}")

# 测试6: 简单计算
print("\n6. Testing basic operations...")
y = torch.randn(10, 10).cuda()
z = x + y
print(f"   Addition result shape: {z.shape}, device: {z.device}")

# 测试7: 矩阵乘法
print("\n7. Testing matrix multiplication...")
result = torch.matmul(x, y)
print(f"   Matmul result shape: {result.shape}, device: {result.device}")

print("\nAll basic tests passed!")

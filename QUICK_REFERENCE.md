# FakeGPU 快速参考

## 常用命令

### 编译库
```bash
# 带日志（调试用）
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=ON
cmake --build build

# 不带日志（生产用）
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=OFF
cmake --build build
```

### 运行Python程序
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
python3 your_script.py
```

### GPU监控

**使用nvitop（Python API）：**
```bash
./fakegpu_nvitop.sh
```

**一次性查询：**
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH \
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12 \
python3 -c "from nvitop import Device; [print(f'{i}: {d.name()}') for i, d in enumerate(Device.all())]"
```

### 测试

**Qwen2.5模型测试：**
```bash
# 完整对比（真实GPU vs FakeGPU）
./test/run_full_comparison.sh

# 仅FakeGPU
./test/run_qwen_comparison.sh

# 分析结果
python3 test/compare_results.py
```

**基础PyTorch测试：**
```bash
./test/run_comparison.sh
```

## 问题排查

### 终端光标消失
```bash
reset
# 或
stty sane
```

### 查看已实现的NVML函数
```bash
nm -D ./build/libnvidia-ml.so.1 | grep ' T nvml'
```

### 查看库依赖
```bash
ldd ./build/libnvidia-ml.so.1
ldd ./build/libcuda.so.1
ldd ./build/libcudart.so.12
ldd ./build/libcublas.so.12
```

## 环境变量

### 必需的环境变量
```bash
LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH
LD_PRELOAD=./build/libcudart.so.12:./build/libcuda.so.1:./build/libnvidia-ml.so.1:./build/libcublas.so.12
```

### 可选的环境变量
```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1     # 禁用PyTorch CUDA内存缓存
TRANSFORMERS_NO_TORCHVISION=1        # 禁用torchvision
TORCH_SDPA_KERNEL=math               # 使用数学kernel
CUDA_LAUNCH_BLOCKING=1               # 同步CUDA调用（调试用）
```

## 已测试的应用

✓ **深度学习框架：**
- PyTorch (基础操作、模型加载、推理)
- Transformers (Qwen2.5-0.5B)

✓ **监控工具：**
- nvitop (Python API)
- pynvml

✓ **基础操作：**
- CUDA内存分配/释放
- 设备查询
- 内存信息查询

⚠️ **已知限制：**
- nvitop TUI模式（会段错误）
- 实际计算结果（返回随机值）
- cuDNN操作（未实现）

## 文件结构

```
fakeGPU/
├── build/                      # 编译输出
│   ├── libnvidia-ml.so.1      # NVML库
│   ├── libcuda.so.1           # CUDA Driver库
│   ├── libcudart.so.12        # CUDA Runtime库
│   └── libcublas.so.12        # cuBLAS库
├── src/                       # 源代码
│   ├── core/                  # 核心实现
│   ├── nvml/                  # NVML stubs
│   ├── cuda/                  # CUDA stubs
│   ├── cublas/                # cuBLAS stubs
│   └── monitor/               # 监控和报告
├── test/                      # 测试脚本
│   ├── test_load_qwen2_5.py  # Qwen2.5测试
│   ├── run_full_comparison.sh # 完整对比测试
│   └── output/                # 测试输出
├── fakegpu_nvitop.sh         # nvitop便捷脚本
├── test_nvitop_wrapper.py    # nvitop Python包装
├── NVITOP_SOLUTION.md        # nvitop解决方案
└── fake_gpu_report.json      # 运行时生成的报告
```

## 相关文档

- [CLAUDE.md](CLAUDE.md) - 项目架构和设计
- [NVITOP_SOLUTION.md](NVITOP_SOLUTION.md) - nvitop使用指南
- [test/README_QWEN_TEST.md](test/README_QWEN_TEST.md) - Qwen2.5测试说明
- [test/output/TEST_SUMMARY.md](test/output/TEST_SUMMARY.md) - 测试结果总结

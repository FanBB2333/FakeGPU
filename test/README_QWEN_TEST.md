# Qwen2.5 模型测试说明

## 概述

这个目录包含了用于测试FakeGPU库运行Qwen2.5模型推理的脚本和结果。

## 文件说明

### 测试脚本

- **test_load_qwen2_5.py** - 主测试脚本，加载Qwen2.5模型并进行单token推理
- **run_qwen_comparison.sh** - 仅运行FakeGPU测试（不需要真实GPU）
- **run_full_comparison.sh** - 完整对比测试，同时运行真实GPU和FakeGPU测试
- **compare_results.py** - 分析和对比测试结果

### 输出文件

- **output/real_gpu_output.txt** - 真实GPU测试的完整输出
- **output/fakegpu_output.txt** - FakeGPU测试的完整输出
- **output/TEST_SUMMARY.md** - 测试结果总结
- **../fake_gpu_report.json** - FakeGPU生成的内存使用报告

## 快速开始

### 1. 编译FakeGPU库（关闭日志）

```bash
cd /path/to/FakeGPU
cmake -S . -B build -DENABLE_FAKEGPU_LOGGING=OFF
cmake --build build
```

### 2. 运行测试

#### 选项A：完整对比测试（推荐，需要真实GPU）

```bash
./test/run_full_comparison.sh
```

这将：
1. 在真实GPU上运行模型推理
2. 在FakeGPU上运行模型推理
3. 对比两次运行的结果
4. 生成详细的分析报告

#### 选项B：仅FakeGPU测试（无需真实GPU）

```bash
./test/run_qwen_comparison.sh
```

这将仅使用FakeGPU运行测试。

### 3. 分析结果

```bash
python3 test/compare_results.py
```

这将显示详细的对比分析，包括：
- 设备信息对比
- 生成token对比
- 测试状态
- 内存使用情况

## 测试结果说明

### 预期结果

✓ **成功标准**：
- 模型成功加载
- Forward pass执行完成
- 生成一个token（任意内容）
- 程序正常退出（TEST PASSED）

⚠️ **预期差异**：
- FakeGPU生成的token与真实GPU不同
- 这是正常现象，因为FakeGPU不执行实际计算

### 测试输出示例

真实GPU:
```
Using device: cuda
GPU: NVIDIA CUDA GPU
Model loaded successfully!
Generated token: Hello
=== TEST PASSED ===
```

FakeGPU:
```
Using device: cuda
GPU: Fake NVIDIA A100-SXM4-80GB
Model loaded successfully!
Generated token: !
=== TEST PASSED ===
```

## 常见问题

### Q: 为什么FakeGPU生成的token不正确？

A: FakeGPU的目标是让代码能够运行，而不是产生正确的计算结果。cuBLAS函数返回随机值，所以生成的token是随机的。

### Q: 为什么内存使用显示为0？

A: PyTorch可能使用了自己的内存池管理机制，绕过了cudaMalloc。FakeGPU仍然能够正常工作，只是无法追踪这部分内存分配。

### Q: 如何在没有真实GPU的机器上测试？

A: 使用 `./test/run_qwen_comparison.sh`，这个脚本只运行FakeGPU测试。

### Q: 编译时的warning可以忽略吗？

A: 是的，UUID格式化的warning不影响功能。

## 性能说明

- **模型大小**: Qwen2.5-0.5B (~500M参数)
- **测试时间**: 通常5-30秒（取决于模型加载时间）
- **内存需求**: 至少2GB系统RAM

## 进一步测试

如果想测试其他模型，可以修改 `test_load_qwen2_5.py` 中的 `model_name` 变量：

```python
model_name = os.path.join(model_path_base, "your-model-name")
```

## 相关文档

- [项目CLAUDE.md](../CLAUDE.md) - FakeGPU项目架构说明
- [测试结果总结](output/TEST_SUMMARY.md) - 详细测试结果

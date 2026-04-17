# 错误模拟

FakeGPU 能够复现常见的真实 GPU 运行时错误，使开发者可以在没有物理 GPU 的机器上验证错误处理代码路径。所有错误模拟功能属于 Python 层的 `torch_patch`，默认启用。

## 错误类别

| 编码 | 类别 | 捕获内容 |
|------|------|----------|
| E1 | 跨设备 | 混合不同 CUDA 设备上张量的操作 |
| E2 | 显存不足 | 分配超过单设备显存限制 |
| E3 | 无效设备索引 | 引用超出已配置数量的设备 ID |
| E4 | dtype / autocast | 在不支持 bfloat16 的设备上使用 bfloat16 autocast（计算能力 < 8.0） |
| E5 | 检查点加载 | 加载为不同 GPU 架构保存的检查点 |
| E6 | 分布式 | （尚未实现） |
| E7 | 梯度 | 对非叶子张量或已 detach 的张量进行梯度计算 |

## E1：跨设备操作

对张量运算（算术、matmul、`torch.cat`、`F.linear`）和模块前向传播进行检查。当操作数位于不同的 fake CUDA 设备时，抛出 `RuntimeError`，提示两个设备索引。

错误信息示例：

```
RuntimeError: cross-device operation: tensor on cuda:0, other on cuda:1
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
a = torch.randn(3, device="cuda:0")
b = torch.randn(3, device="cuda:1")
a + b  # RuntimeError: cross-device operation: tensor on cuda:0, other on cuda:1
```

## E2：显存不足

基于可配置 profile 的单设备显存跟踪。当分配超过设备的 `total_memory` 时，抛出 `torch.cuda.OutOfMemoryError`。

错误信息示例：

```
torch.cuda.OutOfMemoryError: CUDA out of memory on device cuda:0. Tried to allocate 37.25 GiB.
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
# 默认 A100 profile：每设备 80 GB
torch.randn(100000, 100000, device="cuda:0")  # 可能超出显存限制
```

## E3：无效设备索引

在 `cudaSetDevice`、`torch.device` 及张量创建时验证设备索引。引用超出已配置设备数量的序号时抛出 `RuntimeError`。

错误信息示例：

```
RuntimeError: invalid device ordinal 99
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
torch.cuda.set_device(99)  # RuntimeError: invalid device ordinal 99
```

## E4：dtype / autocast 兼容性

在启用 bfloat16 autocast 前检查计算能力。计算能力低于 8.0 的设备（如 T4，计算能力 7.5）不支持 bfloat16，抛出 `RuntimeError`。

错误信息示例：

```
RuntimeError: bfloat16 autocast requires compute capability >= 8.0 (current device: T4, compute 7.5)
```

```python
# FAKEGPU_PROFILE=t4 python script.py
import fakegpu; fakegpu.patch_torch()
import torch
with torch.autocast("cuda", dtype=torch.bfloat16):  # 在 T4 上抛出 RuntimeError
    ...
```

## E5：检查点加载兼容性

验证检查点元数据是否与当前 fake GPU profile 匹配。检测架构不一致的情况。

错误信息示例：

```
RuntimeError: checkpoint was saved on A100 (compute 8.0) but current device is T4 (compute 7.5)
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
# 在 "A100" profile 下保存，在 "T4" profile 下加载时抛出 RuntimeError
```

## E7：梯度错误

捕获对非叶子张量访问 `grad` 以及对已 detach 张量调用 `backward()` 的情况，抛出 `RuntimeError`。

错误信息示例：

```
RuntimeError: cannot access grad of non-leaf tensor
```

```python
import fakegpu; fakegpu.patch_torch()
import torch
x = torch.randn(3, device="cuda", requires_grad=True)
y = x * 2
y.grad  # 非叶子张量；误用时抛出 RuntimeError
```

## 环境变量

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `FAKEGPU_CROSS_DEVICE_CHECK` | `1`（启用） | 启用跨设备操作检查 (E1) |
| `FAKEGPU_MEMORY_TRACKING` | `1`（启用） | 启用单设备显存跟踪和 OOM 模拟 (E2) |
| `FAKEGPU_STRICT_COMPAT` | `1`（启用） | 启用严格的 dtype / 架构兼容性检查 (E4, E5) |

将任一变量设为 `0` 可禁用对应的检查。

## 运行错误模拟测试套件

```bash
python test/run_error_simulation_suite.py
```

运行全部 23 个错误模拟测试，并在 `test/report.html` 生成统一的 HTML 报告。报告包含分页导航，覆盖 Phase 1（设备发现）、Phase 2（训练流程）、Phase 3（MoE）和 Phase 4（错误模拟）。

各测试文件：

```bash
python test/test_error_cross_device.py      # E1：5 个测试
python test/test_error_oom.py               # E2：5 个测试
python test/test_error_device_index.py      # E3：4 个测试
python test/test_error_dtype_autocast.py    # E4：3 个测试
python test/test_error_checkpoint_load.py   # E5：3 个测试
python test/test_error_gradient.py          # E7：3 个测试
```

## 限制

- 错误模拟仅在 Python 层（`torch_patch`）实现；C stub 层仅有部分跨设备支持。
- E6（分布式错误）尚未实现。
- `tensor.device` 仍然显示 `cpu`——fake 设备索引在内部跟踪。
- 不支持 stream 和 event 的错误模拟。

## 相关页面

- [快速入门](getting-started.zh.md)
- [快速参考](quick-reference.zh.md)
- [报告与验证](reports-and-validation.zh.md)
